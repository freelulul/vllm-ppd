#!/usr/bin/env python3
"""
Replication Mode Proxy with Conversation-Aware Routing

This is a simple proxy that distributes requests to 2 standalone vLLM workers
using conversation-aware routing for cache affinity.

Features:
- Conversation-aware routing: Same conversation always goes to same worker
- Turn 1 and Turn 2 of same conversation are routed to same GPU
- Metrics collection: Records timing and worker assignment for each request
- Simple architecture: No KV transfer, no complex disaggregation logic

Usage:
    python replication_proxy.py --worker0 localhost:8300 --worker1 localhost:8400 --port 10002
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
WORKER_URLS = []  # Will be set from args

# Conversation state tracking
conversation_to_worker: dict[str, int] = {}
conversation_lock = threading.Lock()

# Metrics storage
@dataclass
class RequestMetrics:
    """Detailed metrics for a single request."""
    request_id: str
    mode: str = "replication"
    worker_id: int = 0
    conv_id: str = ""
    turn: int = 1

    # Timing (ms)
    total_time_ms: float = 0.0
    worker_time_ms: float = 0.0
    proxy_overhead_ms: float = 0.0

    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0

    timestamp: float = field(default_factory=time.time)


metrics_storage: list[RequestMetrics] = []
metrics_lock = threading.Lock()
request_counter = 0
counter_lock = threading.Lock()


def store_metrics(metrics: RequestMetrics):
    """Store metrics for later retrieval."""
    with metrics_lock:
        metrics_storage.append(metrics)
        if len(metrics_storage) > 10000:
            metrics_storage.pop(0)


def extract_conversation_id(data: dict) -> tuple[str, int]:
    """
    Extract conversation ID from prompt for routing.

    For multi-turn conversations, we extract the first "User: ..." content
    which identifies the conversation. This ensures Turn 1 and Turn 2 go to same GPU.

    Returns: (conv_id, turn_number)
    """
    if "messages" in data:
        # Chat format - use first user message as conversation identifier
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Hash first user message for conversation ID
                conv_id = hashlib.md5(content[:200].encode()).hexdigest()[:16]
                turn = sum(1 for m in messages if m.get("role") == "user")
                return conv_id, turn
        return "default", 1
    else:
        # Completion format - extract first "User: ..." segment
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)

        # Parse conversation structure
        # Format: "User: {turn1_prompt}\nAssistant: {turn1_response}\nUser: {turn2_prompt}\nAssistant:"

        # Extract first User segment for conversation ID
        user_match = re.search(r'User:\s*(.+?)(?:\nAssistant:|$)', prompt, re.DOTALL)
        if user_match:
            first_user_content = user_match.group(1).strip()[:200]
            conv_id = hashlib.md5(first_user_content.encode()).hexdigest()[:16]
        else:
            conv_id = hashlib.md5(prompt[:200].encode()).hexdigest()[:16]

        # Count turns by counting "User:" occurrences
        turn = prompt.count("User:")
        if turn == 0:
            turn = 1

        return conv_id, turn


def get_worker_for_conversation(conv_id: str) -> int:
    """
    Get or assign worker for a conversation.
    Same conversation always goes to same worker for cache affinity.
    """
    with conversation_lock:
        if conv_id not in conversation_to_worker:
            # Assign based on hash for consistent distribution
            worker_idx = int(hashlib.md5(conv_id.encode()).hexdigest(), 16) % len(WORKER_URLS)
            conversation_to_worker[conv_id] = worker_idx
        return conversation_to_worker[conv_id]


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


async def forward_request(url: str, data: dict, request_id: str) -> tuple[bytes, float, dict]:
    """Forward request to worker and return (response_bytes, time_ms, usage_dict)."""
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

    # Parse response for usage info
    try:
        resp_json = json.loads(response_bytes.decode())
        usage = resp_json.get("usage", {})
    except:
        pass

    return response_bytes, elapsed_ms, usage


async def forward_request_streaming(url: str, data: dict, request_id: str):
    """Forward request with true SSE streaming."""
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

        # Get unique request ID
        with counter_lock:
            request_counter += 1
            req_id = f"repl_{request_counter}"

        # Extract conversation ID and turn number
        conv_id, turn = extract_conversation_id(data)

        # Get worker for this conversation (same conversation always goes to same GPU)
        worker_idx = get_worker_for_conversation(conv_id)
        worker_url = WORKER_URLS[worker_idx]

        # Initialize metrics
        metrics = RequestMetrics(
            request_id=req_id,
            worker_id=worker_idx,
            conv_id=conv_id,
            turn=turn,
        )

        is_streaming = data.get("stream", False)
        target_url = f"http://{worker_url}{request.path}"

        if is_streaming:
            # Return proper SSE streaming response
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
            # Non-streaming: collect full response
            response_bytes, worker_time_ms, usage = await forward_request(
                target_url, data, req_id
            )

            # Record metrics
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
    return {
        "mode": "replication",
        "workers": WORKER_URLS,
        "total_requests": request_counter,
        "conversations": conv_count,
    }


@app.route("/conversations/clear", methods=["POST"])
async def clear_conversations():
    """Clear conversation state."""
    global conversation_to_worker
    with conversation_lock:
        count = len(conversation_to_worker)
        conversation_to_worker = {}
    return {"cleared": count}


@app.route("/metrics", methods=["GET"])
async def get_metrics():
    """Get all stored metrics."""
    with metrics_lock:
        return jsonify([asdict(m) for m in metrics_storage])


@app.route("/metrics/clear", methods=["POST"])
async def clear_metrics():
    """Clear stored metrics."""
    global metrics_storage, request_counter
    with metrics_lock:
        count = len(metrics_storage)
        metrics_storage = []
    with counter_lock:
        request_counter = 0
    return {"cleared": count}


@app.route("/metrics/summary", methods=["GET"])
async def get_metrics_summary():
    """Get summary statistics of metrics."""
    with metrics_lock:
        if not metrics_storage:
            return {"error": "No metrics available"}

        n = len(metrics_storage)
        worker0_count = sum(1 for m in metrics_storage if m.worker_id == 0)
        worker1_count = sum(1 for m in metrics_storage if m.worker_id == 1)
        turn1_count = sum(1 for m in metrics_storage if m.turn == 1)
        turn2_count = sum(1 for m in metrics_storage if m.turn >= 2)

        return {
            "total_requests": n,
            "worker0_requests": worker0_count,
            "worker1_requests": worker1_count,
            "turn1_requests": turn1_count,
            "turn2_requests": turn2_count,
            "avg_total_ms": sum(m.total_time_ms for m in metrics_storage) / n,
            "avg_worker_ms": sum(m.worker_time_ms for m in metrics_storage) / n,
            "avg_overhead_ms": sum(m.proxy_overhead_ms for m in metrics_storage) / n,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replication Mode Proxy")
    parser.add_argument("--worker0", required=True, help="Worker 0 address (host:port)")
    parser.add_argument("--worker1", required=True, help="Worker 1 address (host:port)")
    parser.add_argument("--port", type=int, default=10002, help="Proxy port")
    parser.add_argument("--host", default="0.0.0.0", help="Proxy host")

    args = parser.parse_args()

    WORKER_URLS.append(args.worker0)
    WORKER_URLS.append(args.worker1)

    print("=" * 60)
    print("Replication Mode Proxy (Conversation-Aware Routing)")
    print("=" * 60)
    print(f"Worker 0: {args.worker0}")
    print(f"Worker 1: {args.worker1}")
    print(f"Listening on: http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port)
