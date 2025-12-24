#!/usr/bin/env python3
"""
Benchmark-Enabled Disaggregated Proxy Server

This proxy captures detailed timing metrics for each request phase:
- P prefill time (time spent on prefill server)
- D decode time (time spent on decode server)
- Overhead time (proxy routing, network latency)

Usage:
    python disagg_proxy_benchmark.py --mode ppd --http-port 10001
"""

import argparse
import hashlib
import json
import os
import socket
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request, jsonify

# Global state
count = 0
prefill_instances: dict[str, Any] = {}
decode_instances: dict[str, Any] = {}

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

# Conversation tracking
conversation_state: dict[str, tuple[int, float]] = {}
conversation_lock = threading.Lock()

# Configuration
ROUTING_MODE = "pd"
DEFAULT_PING_SECONDS = 5
CONVERSATION_TTL = 3600


@dataclass
class RequestMetrics:
    """Detailed metrics for a single request."""
    request_id: str
    mode: str  # "pd" or "ppd_turn1" or "ppd_d_direct"
    turn: int
    conv_hash: str

    # Timing (ms)
    total_time_ms: float = 0.0
    p_prefill_time_ms: float = 0.0  # Time on P server (PD mode only)
    d_time_ms: float = 0.0  # Time on D server
    proxy_overhead_ms: float = 0.0

    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Estimated KV transfer (based on prompt tokens)
    estimated_kv_size_mb: float = 0.0

    timestamp: float = field(default_factory=time.time)


# Metrics storage
metrics_storage: list[RequestMetrics] = []
metrics_lock = threading.Lock()


def store_metrics(metrics: RequestMetrics):
    """Store metrics for later retrieval."""
    with metrics_lock:
        metrics_storage.append(metrics)
        # Keep last 10000 requests
        if len(metrics_storage) > 10000:
            metrics_storage.pop(0)


def calculate_kv_size_mb(num_tokens: int) -> float:
    """
    Calculate KV cache size for Llama-3.1-8B.
    KV size per token = 32 layers * 2 (K+V) * 8 heads * 128 dim * 2 bytes = 128 KB
    """
    kv_bytes_per_token = 32 * 2 * 8 * 128 * 2  # 131072 bytes = 128 KB
    return (num_tokens * kv_bytes_per_token) / (1024 * 1024)


def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["type"] == "P":
                global prefill_instances, prefill_cv
                with prefill_cv:
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(prefill_instances)
            elif data["type"] == "D":
                global decode_instances, decode_cv
                with decode_cv:
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    _listener_thread = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    _listener_thread.start()
    return _listener_thread


def get_conversation_hash(data: dict) -> str:
    if "messages" in data:
        messages = data.get("messages", [])
        if len(messages) > 1:
            history = messages[:-1]
            history_str = str(history)
        else:
            history_str = ""
    else:
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)
        prefix_len = int(len(prompt) * 0.8)
        history_str = prompt[:prefix_len]
    return hashlib.md5(history_str.encode()).hexdigest()


def get_conversation_turn(conv_hash: str) -> int:
    with conversation_lock:
        if conv_hash in conversation_state:
            return conversation_state[conv_hash][0]
        return 0


def update_conversation_turn(conv_hash: str) -> int:
    with conversation_lock:
        current_time = time.time()
        expired = [k for k, v in conversation_state.items()
                   if current_time - v[1] > CONVERSATION_TTL]
        for k in expired:
            del conversation_state[k]

        if conv_hash in conversation_state:
            turn_count, _ = conversation_state[conv_hash]
            turn_count += 1
        else:
            turn_count = 1
        conversation_state[conv_hash] = (turn_count, current_time)
        return turn_count


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


async def forward_request_with_timing(url, data, request_id) -> tuple[bytes, float, dict]:
    """Forward request and return (response_bytes, time_ms, usage_dict)."""
    start_time = time.perf_counter()
    response_bytes = b""
    usage = {}

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk in response.content.iter_chunked(1024):
                    response_bytes += chunk

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Parse response for usage info
    try:
        resp_json = json.loads(response_bytes.decode())
        usage = resp_json.get("usage", {})
    except:
        pass

    return response_bytes, elapsed_ms, usage


async def forward_request_streaming(url, data, request_id):
    """Forward request with streaming response."""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(1024):
                    yield chunk_bytes


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        request_start = time.perf_counter()
        original_request_data = await request.get_json()

        global count, prefill_instances, decode_instances

        # Get instances
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            if not prefill_list:
                return {"error": "No prefill instances available"}, 503
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            prefill_zmq_addr = prefill_zmq_addr[0]

        with decode_cv:
            decode_list = list(decode_instances.items())
            if not decode_list:
                return {"error": "No decode instances available"}, 503
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        # Determine routing
        conv_hash = get_conversation_hash(original_request_data)
        turn_number = update_conversation_turn(conv_hash)
        use_d_direct = (ROUTING_MODE == "ppd" and turn_number > 1)

        # Initialize metrics
        metrics = RequestMetrics(
            request_id="",
            mode="",
            turn=turn_number,
            conv_hash=conv_hash[:8],
        )

        if use_d_direct:
            # PPD D-Direct Mode
            request_id = f"d_direct_{random_uuid()}"
            metrics.request_id = request_id[:20]
            metrics.mode = "ppd_d_direct"

            count += 1

            # Send directly to D
            d_start = time.perf_counter()
            response_bytes, d_time_ms, usage = await forward_request_with_timing(
                f"http://{decode_addr}{request.path}",
                original_request_data,
                request_id
            )

            metrics.d_time_ms = d_time_ms
            metrics.prompt_tokens = usage.get("prompt_tokens", 0)
            metrics.completion_tokens = usage.get("completion_tokens", 0)
            metrics.total_time_ms = (time.perf_counter() - request_start) * 1000
            metrics.proxy_overhead_ms = metrics.total_time_ms - metrics.d_time_ms

            store_metrics(metrics)

            return response_bytes, 200, {"Content-Type": "application/json"}

        else:
            # PD Mode (or PPD Turn 1)
            prefill_request = original_request_data.copy()
            prefill_request["max_tokens"] = 1
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1

            request_id = (
                f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
                f"{decode_zmq_addr}_{random_uuid()}"
            )

            metrics.request_id = request_id[:20]
            metrics.mode = "pd" if ROUTING_MODE == "pd" else "ppd_turn1"

            count += 1

            # Step 1: Prefill on P
            _, p_time_ms, p_usage = await forward_request_with_timing(
                f"http://{prefill_addr}{request.path}",
                prefill_request,
                request_id
            )
            metrics.p_prefill_time_ms = p_time_ms

            # Estimate KV size based on prefill tokens
            prefill_tokens = p_usage.get("prompt_tokens", 0)
            metrics.estimated_kv_size_mb = calculate_kv_size_mb(prefill_tokens)

            # Step 2: Decode on D
            response_bytes, d_time_ms, d_usage = await forward_request_with_timing(
                f"http://{decode_addr}{request.path}",
                original_request_data,
                request_id
            )

            metrics.d_time_ms = d_time_ms
            metrics.prompt_tokens = d_usage.get("prompt_tokens", 0)
            metrics.completion_tokens = d_usage.get("completion_tokens", 0)
            metrics.total_time_ms = (time.perf_counter() - request_start) * 1000
            metrics.proxy_overhead_ms = metrics.total_time_ms - metrics.p_prefill_time_ms - metrics.d_time_ms

            store_metrics(metrics)

            return response_bytes, 200, {"Content-Type": "application/json"}

    except Exception as e:
        import traceback
        print(f"[PROXY] Error: {e}")
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.route("/mode", methods=["GET"])
async def get_mode():
    return {"mode": ROUTING_MODE, "conversations": len(conversation_state)}


@app.route("/mode/<new_mode>", methods=["POST"])
async def set_mode(new_mode):
    global ROUTING_MODE
    if new_mode not in ["pd", "ppd"]:
        return {"error": "Invalid mode. Use 'pd' or 'ppd'"}, 400
    old_mode = ROUTING_MODE
    ROUTING_MODE = new_mode
    return {"mode": ROUTING_MODE, "previous": old_mode}


@app.route("/conversations/clear", methods=["POST"])
async def clear_conversations():
    global conversation_state
    with conversation_lock:
        count = len(conversation_state)
        conversation_state = {}
    return {"cleared": count}


@app.route("/metrics", methods=["GET"])
async def get_metrics():
    """Get all stored metrics."""
    with metrics_lock:
        return jsonify([asdict(m) for m in metrics_storage])


@app.route("/metrics/clear", methods=["POST"])
async def clear_metrics():
    """Clear stored metrics."""
    global metrics_storage
    with metrics_lock:
        count = len(metrics_storage)
        metrics_storage = []
    return {"cleared": count}


@app.route("/metrics/summary", methods=["GET"])
async def get_metrics_summary():
    """Get summary statistics of metrics."""
    with metrics_lock:
        if not metrics_storage:
            return {"error": "No metrics available"}

        # Group by mode
        by_mode = defaultdict(list)
        for m in metrics_storage:
            by_mode[m.mode].append(m)

        summary = {}
        for mode, metrics_list in by_mode.items():
            n = len(metrics_list)
            summary[mode] = {
                "count": n,
                "avg_total_ms": sum(m.total_time_ms for m in metrics_list) / n,
                "avg_p_prefill_ms": sum(m.p_prefill_time_ms for m in metrics_list) / n,
                "avg_d_time_ms": sum(m.d_time_ms for m in metrics_list) / n,
                "avg_overhead_ms": sum(m.proxy_overhead_ms for m in metrics_list) / n,
                "avg_prompt_tokens": sum(m.prompt_tokens for m in metrics_list) / n,
                "avg_completion_tokens": sum(m.completion_tokens for m in metrics_list) / n,
                "avg_kv_size_mb": sum(m.estimated_kv_size_mb for m in metrics_list) / n,
            }

        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark-Enabled Disaggregated Proxy")
    parser.add_argument("--mode", choices=["pd", "ppd"], default="pd")
    parser.add_argument("--http-port", type=int, default=10001)
    parser.add_argument("--zmq-port", type=int, default=30001)
    parser.add_argument("--host", default="0.0.0.0")

    args = parser.parse_args()
    ROUTING_MODE = args.mode

    print("=" * 60)
    print("Benchmark-Enabled Disaggregated Proxy")
    print("=" * 60)
    print(f"Mode: {ROUTING_MODE}")
    print(f"HTTP Port: {args.http_port}")
    print(f"Metrics endpoints: /metrics, /metrics/summary, /metrics/clear")
    print("=" * 60)

    t = start_service_discovery(args.host, args.zmq_port)
    app.run(host=args.host, port=args.http_port)
    t.join()
