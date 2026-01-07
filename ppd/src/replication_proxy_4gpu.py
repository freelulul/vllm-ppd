#!/usr/bin/env python3
"""
Replication Mode Proxy with N-Worker Support

Extended version of replication_proxy.py to support arbitrary number of workers.
Uses conversation-aware routing for cache affinity.

Usage:
    python replication_proxy_4gpu.py --workers "host1:port1,host2:port2,host3:port3,host4:port4" --port 10002
"""

import argparse
import hashlib
import json
import re
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional

import aiohttp
from quart import Quart, request, jsonify, Response

# Configuration
WORKER_URLS: list[str] = []

# Conversation state tracking
conversation_to_worker: dict[str, int] = {}
conversation_lock = threading.Lock()


@dataclass
class RequestMetrics:
    """Detailed metrics for a single request."""
    request_id: str
    mode: str = "replication"
    worker_id: int = 0
    conv_id: str = ""
    turn: int = 1
    total_time_ms: float = 0.0
    worker_time_ms: float = 0.0
    proxy_overhead_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timestamp: float = field(default_factory=time.time)


metrics_storage: list[RequestMetrics] = []
metrics_lock = threading.Lock()
request_counter = 0
counter_lock = threading.Lock()


def store_metrics(metrics: RequestMetrics):
    with metrics_lock:
        metrics_storage.append(metrics)
        if len(metrics_storage) > 10000:
            metrics_storage.pop(0)


def extract_conversation_id(data: dict) -> tuple[str, int]:
    """Extract conversation ID from prompt for routing."""
    if "messages" in data:
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                conv_id = hashlib.md5(content[:200].encode()).hexdigest()[:16]
                turn = sum(1 for m in messages if m.get("role") == "user")
                return conv_id, turn
        return "default", 1
    else:
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)

        user_match = re.search(r'User:\s*(.+?)(?:\nAssistant:|$)', prompt, re.DOTALL)
        if user_match:
            first_user_content = user_match.group(1).strip()[:200]
            conv_id = hashlib.md5(first_user_content.encode()).hexdigest()[:16]
        else:
            conv_id = hashlib.md5(prompt[:200].encode()).hexdigest()[:16]

        turn = prompt.count("User:")
        if turn == 0:
            turn = 1

        return conv_id, turn


def get_worker_for_conversation(conv_id: str) -> int:
    """Get or assign worker for a conversation (hash-based for N workers)."""
    with conversation_lock:
        if conv_id not in conversation_to_worker:
            worker_idx = int(hashlib.md5(conv_id.encode()).hexdigest(), 16) % len(WORKER_URLS)
            conversation_to_worker[conv_id] = worker_idx
        return conversation_to_worker[conv_id]


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)


async def forward_request(url: str, data: dict, request_id: str) -> tuple[bytes, float, dict]:
    start_time = time.perf_counter()
    response_bytes = b""
    usage = {}

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"X-Request-Id": request_id}
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk in response.content.iter_any():
                    if chunk:
                        response_bytes += chunk

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    try:
        resp_json = json.loads(response_bytes.decode())
        usage = resp_json.get("usage", {})
    except:
        pass

    return response_bytes, elapsed_ms, usage


async def forward_request_streaming(url: str, data: dict, request_id: str):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"X-Request-Id": request_id}
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    global request_counter

    try:
        request_start = time.perf_counter()
        data = await request.get_json()

        with counter_lock:
            request_counter += 1
            req_id = f"repl_{request_counter}"

        conv_id, turn = extract_conversation_id(data)
        worker_idx = get_worker_for_conversation(conv_id)
        worker_url = WORKER_URLS[worker_idx]

        metrics = RequestMetrics(
            request_id=req_id,
            worker_id=worker_idx,
            conv_id=conv_id,
            turn=turn,
        )

        print(f"[PROXY] Request {req_id}: conv={conv_id[:8]}, turn={turn}, -> Worker {worker_idx} ({worker_url})")

        is_streaming = data.get("stream", False)
        target_url = f"http://{worker_url}{request.path}"

        if is_streaming:
            return Response(
                forward_request_streaming(target_url, data, req_id),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Transfer-Encoding": "chunked",
                }
            )
        else:
            response_bytes, worker_time_ms, usage = await forward_request(target_url, data, req_id)

            metrics.worker_time_ms = worker_time_ms
            metrics.prompt_tokens = usage.get("prompt_tokens", 0)
            metrics.completion_tokens = usage.get("completion_tokens", 0)
            metrics.total_time_ms = (time.perf_counter() - request_start) * 1000
            metrics.proxy_overhead_ms = metrics.total_time_ms - metrics.worker_time_ms

            store_metrics(metrics)

            return response_bytes, 200, {"Content-Type": "application/json"}

    except Exception as e:
        import traceback
        print(f"[PROXY] Error: {e}")
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.route("/status", methods=["GET"])
async def get_status():
    with conversation_lock:
        conv_count = len(conversation_to_worker)

    # Count requests per worker
    worker_counts = {i: 0 for i in range(len(WORKER_URLS))}
    with conversation_lock:
        for worker_idx in conversation_to_worker.values():
            worker_counts[worker_idx] = worker_counts.get(worker_idx, 0) + 1

    return {
        "mode": "replication",
        "num_workers": len(WORKER_URLS),
        "workers": WORKER_URLS,
        "total_requests": request_counter,
        "conversations": conv_count,
        "conversations_per_worker": worker_counts,
    }


@app.route("/conversations/clear", methods=["POST"])
async def clear_conversations():
    global conversation_to_worker
    with conversation_lock:
        count = len(conversation_to_worker)
        conversation_to_worker = {}
    return {"cleared": count}


@app.route("/metrics", methods=["GET"])
async def get_metrics():
    with metrics_lock:
        return jsonify([asdict(m) for m in metrics_storage])


@app.route("/metrics/clear", methods=["POST"])
async def clear_metrics():
    global metrics_storage, request_counter
    with metrics_lock:
        count = len(metrics_storage)
        metrics_storage = []
    with counter_lock:
        request_counter = 0
    return {"cleared": count}


@app.route("/metrics/summary", methods=["GET"])
async def get_metrics_summary():
    with metrics_lock:
        if not metrics_storage:
            return {"error": "No metrics available"}

        n = len(metrics_storage)
        worker_counts = {}
        for m in metrics_storage:
            worker_counts[m.worker_id] = worker_counts.get(m.worker_id, 0) + 1

        return {
            "total_requests": n,
            "requests_per_worker": worker_counts,
            "avg_total_ms": sum(m.total_time_ms for m in metrics_storage) / n,
            "avg_worker_ms": sum(m.worker_time_ms for m in metrics_storage) / n,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-Worker Replication Mode Proxy")
    parser.add_argument("--workers", required=True,
                        help="Comma-separated list of worker addresses (host:port)")
    parser.add_argument("--port", type=int, default=10002, help="Proxy port")
    parser.add_argument("--host", default="0.0.0.0", help="Proxy host")

    args = parser.parse_args()

    # Parse worker list
    WORKER_URLS.extend(args.workers.split(","))

    print("=" * 60)
    print(f"Replication Mode Proxy ({len(WORKER_URLS)} Workers)")
    print("=" * 60)
    for i, worker in enumerate(WORKER_URLS):
        print(f"  Worker {i}: {worker}")
    print(f"Listening on: http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port)
