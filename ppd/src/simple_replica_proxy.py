#!/usr/bin/env python3
"""
Simple Replica Proxy - Round-robin load balancer with conversation affinity
"""

import argparse
import asyncio
import hashlib
import aiohttp
from quart import Quart, request, Response

app = Quart(__name__)

workers = []
conversation_map = {}  # conversation_hash -> worker_index
request_count = 0


def get_conversation_hash(data: dict) -> str:
    """Generate stable hash to identify a conversation across turns."""
    # Option 1: Explicit session_id (best)
    if "session_id" in data:
        return hashlib.md5(data["session_id"].encode()).hexdigest()

    # Option 2: Chat format - use first user message
    if "messages" in data:
        messages = data.get("messages", [])
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user:
            content = first_user.get("content", "")[:500]
            return hashlib.md5(content.encode()).hexdigest()

    # Option 3: Completion format - extract first User: block
    prompt = data.get("prompt", "")
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)

    if "User:" in prompt:
        # Extract ONLY the first "User:" message content (stable across turns)
        start = prompt.find("User:") + 5  # skip "User:"
        # Find where this user message ends (next "Assistant:" or "User:" or end)
        end_assistant = prompt.find("Assistant:", start)
        end_user = prompt.find("User:", start)
        if end_assistant > 0 and end_user > 0:
            end = min(end_assistant, end_user)
        elif end_assistant > 0:
            end = end_assistant
        elif end_user > 0:
            end = end_user
        else:
            end = min(start + 500, len(prompt))
        first_user_content = prompt[start:end].strip()
        return hashlib.md5(first_user_content.encode()).hexdigest()

    # Option 4: Fallback - use first 500 chars (for long stable contexts)
    return hashlib.md5(prompt[:500].encode()).hexdigest()


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=3600)


async def forward_request(url, data):
    """Forward request to worker"""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(url, json=data) as response:
            if response.status == 200:
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    global request_count

    data = await request.get_json()
    conv_hash = get_conversation_hash(data)

    # Conversation affinity
    if conv_hash in conversation_map:
        worker_idx = conversation_map[conv_hash]
    else:
        worker_idx = request_count % len(workers)
        conversation_map[conv_hash] = worker_idx
        request_count += 1

    worker_url = f"http://{workers[worker_idx]}{request.path}"

    is_streaming = data.get("stream", False)
    if is_streaming:
        return Response(
            forward_request(worker_url, data),
            mimetype="text/event-stream"
        )
    else:
        response_bytes = b""
        async for chunk in forward_request(worker_url, data):
            response_bytes += chunk
        return response_bytes, 200, {"Content-Type": "application/json"}


@app.route("/status", methods=["GET"])
async def get_status():
    return {
        "workers": workers,
        "request_count": request_count,
        "conversations": len(conversation_map)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", required=True, help="Comma-separated worker addresses")
    parser.add_argument("--port", type=int, default=10002)
    args = parser.parse_args()

    workers = args.workers.split(",")
    print(f"Replica Proxy starting on port {args.port}")
    print(f"Workers: {workers}")

    app.run(host="0.0.0.0", port=args.port)
