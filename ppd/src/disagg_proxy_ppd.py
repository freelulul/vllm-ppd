#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
PPD-Aware Disaggregated Proxy Server

This proxy supports two routing modes for PD separation:

1. PD Mode (--mode pd):
   - Always routes requests through Prefill first (P→D)
   - Every request: Proxy → P (prefill) → [KV Transfer] → D (decode)
   - Original behavior, good for single-turn requests

2. PPD Mode (--mode ppd):
   - First turn of conversation: P→D (normal PD flow)
   - Subsequent turns: Direct to D (D uses local prefix cache)
   - Optimized for multi-turn dialogues

The key difference:
- PD mode: Uses special request_id format that triggers KV transfer
- PPD mode (subsequent turns): Uses normal request_id, D uses local prefix cache

Usage:
    # Start with PD mode (original behavior)
    python disagg_proxy_ppd.py --mode pd

    # Start with PPD mode (optimized for multi-turn)
    python disagg_proxy_ppd.py --mode ppd
"""

import argparse
import hashlib
import os
import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request, Response

# Global state
count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}   # http_address: (zmq_address, stamp)

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

# Conversation tracking for PPD mode
# Key: conversation_hash, Value: (turn_count, last_access_time)
conversation_state: dict[str, tuple[int, float]] = {}
conversation_lock = threading.Lock()

# Configuration
ROUTING_MODE = "pd"  # "pd" or "ppd"
DEFAULT_PING_SECONDS = 5
CONVERSATION_TTL = 3600  # 1 hour TTL for conversation state


def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"[PROXY] Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["type"] == "P":
                global prefill_instances
                global prefill_cv
                with prefill_cv:
                    node = prefill_instances.get(data["http_address"], None)
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(prefill_instances)

            elif data["type"] == "D":
                global decode_instances
                global decode_cv
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
            else:
                print(f"[PROXY] Unexpected message from {remote_address}, data: {data}")
                return

            if node is None:
                print(f"[PROXY] Add [{data['type']}] HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}")


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

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
    """
    Generate a hash to identify a conversation.

    For chat format: Hash the first user message (conversation starter)
    For completion format: Hash the first "User: xxx" segment (first turn)

    The key insight: we want to identify the SAME conversation across turns,
    so we hash the first user message which is constant across all turns.
    """
    if "messages" in data:
        # Chat format - hash the first user message
        messages = data.get("messages", [])
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user:
            history_str = first_user.get("content", "")
        else:
            history_str = ""
    else:
        # Completion format - extract first "User: xxx" segment
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)

        # Find the first user turn: "User: xxx\nAssistant:"
        # This stays constant across all turns of the same conversation
        if "User:" in prompt:
            # Get content from first "User:" to first "Assistant:"
            start = prompt.find("User:")
            end = prompt.find("Assistant:", start)
            if end > start:
                history_str = prompt[start:end].strip()
            else:
                # No Assistant yet, use everything after User:
                history_str = prompt[start:start+200]  # Limit to 200 chars
        else:
            # Fallback: use first 200 chars
            history_str = prompt[:200]

    return hashlib.md5(history_str.encode()).hexdigest()


def get_conversation_turn(conv_hash: str) -> int:
    """Get the current turn number for a conversation."""
    with conversation_lock:
        if conv_hash in conversation_state:
            turn_count, _ = conversation_state[conv_hash]
            return turn_count
        return 0


def update_conversation_turn(conv_hash: str) -> int:
    """Increment and return the turn number for a conversation."""
    with conversation_lock:
        # Clean up old conversations
        current_time = time.time()
        expired = [k for k, v in conversation_state.items()
                   if current_time - v[1] > CONVERSATION_TTL]
        for k in expired:
            del conversation_state[k]

        # Update turn count
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


async def forward_request(url, data, request_id):
    """Forward request to vLLM with true SSE streaming."""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                # Use iter_any() for true streaming - yields data as soon as available
                async for chunk_bytes in response.content.iter_any():
                    if chunk_bytes:
                        yield chunk_bytes


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        global count
        global prefill_instances
        global prefill_cv
        global decode_instances
        global decode_cv

        # Get instance addresses
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

        # Determine routing based on mode
        conv_hash = get_conversation_hash(original_request_data)
        turn_number = update_conversation_turn(conv_hash)

        # PPD mode: subsequent turns go directly to D
        use_d_direct = (ROUTING_MODE == "ppd" and turn_number > 1)

        if use_d_direct:
            # D-direct mode: Send directly to D without PD flow
            # Use a normal request_id (no special format)
            request_id = f"d_direct_{random_uuid()}"

            print(
                f"[PROXY] PPD Mode - D-Direct: count={count}, turn={turn_number}, "
                f"conv={conv_hash[:8]}, -> D:{decode_addr}"
            )
            count += 1

            # Send directly to D
            is_streaming = original_request_data.get("stream", False)
            target_url = f"http://{decode_addr}{request.path}"

            if is_streaming:
                # Return proper SSE streaming response
                return Response(
                    forward_request(target_url, original_request_data, request_id),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Transfer-Encoding": "chunked",
                    }
                )
            else:
                # Non-streaming: collect full response and return as JSON
                response_bytes = b""
                async for chunk in forward_request(target_url, original_request_data, request_id):
                    response_bytes += chunk
                return response_bytes, 200, {"Content-Type": "application/json"}

        else:
            # PD mode: Normal P→D flow
            prefill_request = original_request_data.copy()
            prefill_request["max_tokens"] = 1
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1

            # Create special request_id for PD flow
            request_id = (
                f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
                f"{decode_zmq_addr}_{random_uuid()}"
            )

            mode_str = "PD" if ROUTING_MODE == "pd" else f"PPD-Turn1"
            print(
                f"[PROXY] {mode_str}: count={count}, turn={turn_number}, "
                f"conv={conv_hash[:8]}, P:{prefill_addr} -> D:{decode_addr}"
            )
            count += 1

            # Step 1: Prefill on P
            async for _ in forward_request(
                f"http://{prefill_addr}{request.path}",
                prefill_request,
                request_id
            ):
                continue

            # Step 2: Decode on D (with KV from P)
            is_streaming = original_request_data.get("stream", False)
            decode_url = f"http://{decode_addr}{request.path}"

            if is_streaming:
                # Return proper SSE streaming response
                return Response(
                    forward_request(decode_url, original_request_data, request_id),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Transfer-Encoding": "chunked",
                    }
                )
            else:
                # Non-streaming: collect full response and return as JSON
                response_bytes = b""
                async for chunk in forward_request(decode_url, original_request_data, request_id):
                    response_bytes += chunk
                return response_bytes, 200, {"Content-Type": "application/json"}

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print(f"[PROXY] Error: {e}")
        print("".join(traceback.format_exception(*exc_info)))
        return {"error": str(e)}, 500


@app.route("/mode", methods=["GET"])
async def get_mode():
    """Get current routing mode."""
    return {"mode": ROUTING_MODE, "conversations": len(conversation_state)}


@app.route("/mode/<new_mode>", methods=["POST"])
async def set_mode(new_mode):
    """Set routing mode (pd or ppd)."""
    global ROUTING_MODE
    if new_mode not in ["pd", "ppd"]:
        return {"error": "Invalid mode. Use 'pd' or 'ppd'"}, 400
    old_mode = ROUTING_MODE
    ROUTING_MODE = new_mode
    print(f"[PROXY] Mode changed: {old_mode} -> {new_mode}")
    return {"mode": ROUTING_MODE, "previous": old_mode}


@app.route("/conversations", methods=["GET"])
async def get_conversations():
    """Get conversation state (for debugging)."""
    with conversation_lock:
        return {
            "count": len(conversation_state),
            "conversations": {
                k: {"turns": v[0], "last_access": v[1]}
                for k, v in list(conversation_state.items())[:10]  # Limit to 10
            }
        }


@app.route("/conversations/clear", methods=["POST"])
async def clear_conversations():
    """Clear conversation state."""
    global conversation_state
    with conversation_lock:
        count = len(conversation_state)
        conversation_state = {}
    return {"cleared": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPD-Aware Disaggregated Proxy")
    parser.add_argument("--mode", choices=["pd", "ppd"], default="pd",
                        help="Routing mode: pd (always P->D) or ppd (first turn P->D, rest direct to D)")
    parser.add_argument("--http-port", type=int, default=10001,
                        help="HTTP server port")
    parser.add_argument("--zmq-port", type=int, default=30001,
                        help="ZMQ service discovery port")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind")

    args = parser.parse_args()

    ROUTING_MODE = args.mode

    print("=" * 60)
    print("PPD-Aware Disaggregated Proxy Server")
    print("=" * 60)
    print(f"Routing Mode: {ROUTING_MODE}")
    print(f"  - pd:  Always route P -> D (original behavior)")
    print(f"  - ppd: First turn P -> D, subsequent turns direct to D")
    print(f"HTTP Port: {args.http_port}")
    print(f"ZMQ Port: {args.zmq_port}")
    print("=" * 60)

    t = start_service_discovery(args.host, args.zmq_port)
    app.run(host=args.host, port=args.http_port)
    t.join()
