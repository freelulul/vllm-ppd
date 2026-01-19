#!/usr/bin/env python3
"""
Comprehensive Proxy Server for All 17 GPU Configurations.

This proxy handles routing for:
- Pure Replica (4R)
- Pure Disaggregated (1P+3D, 2P+2D, etc.)
- Hybrid (1R+1P+2D, etc.)

Key features:
- Automatic instance discovery via ZMQ
- Cache affinity for multi-turn conversations
- Flexible routing based on server types (P, D, pD, R)
- Statistics collection

Usage:
    python comprehensive_proxy.py --config 2P_2D --http-port 10001 --zmq-port 30001
"""

import argparse
import hashlib
import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import msgpack
import zmq
from quart import Quart, request, Response


# Global state - instances by type
prefill_instances: Dict[str, Tuple[str, float]] = {}  # http_addr -> (zmq_addr, stamp)
decode_instances: Dict[str, Tuple[str, float]] = {}   # D instances
ppd_decode_instances: Dict[str, Tuple[str, float]] = {}  # pD instances
replica_instances: Dict[str, float] = {}  # http_addr -> stamp

# Locks
prefill_cv = threading.Condition()
decode_cv = threading.Condition()
ppd_decode_cv = threading.Condition()
replica_cv = threading.Condition()

# Conversation tracking
# Key: conv_hash, Value: (turn_count, last_access, assigned_servers)
# assigned_servers: {"prefill": addr, "decode": addr, "replica": addr}
conversation_state: Dict[str, Tuple[int, float, Dict[str, str]]] = {}
conversation_lock = threading.Lock()

# Statistics
stats = {
    "total_requests": 0,
    "turn1_requests": 0,
    "turn2plus_requests": 0,
    "pd_requests": 0,
    "ppd_requests": 0,
    "ppd_direct_requests": 0,
    "replica_requests": 0,
    "cache_affinity_hits": 0,
    "cache_affinity_misses": 0,
}
stats_lock = threading.Lock()

# Configuration
CONFIG_NAME = "2P_2D"
CONVERSATION_TTL = 3600
DEFAULT_PING_SECONDS = 5

# Static configuration for replica ports (replicas don't use ZMQ registration)
REPLICA_PORTS_BY_CONFIG = {
    "4R": [8300, 8400, 8500, 8600],
    "1R_1P_2D": [8300],
    "1R_1P_1D_1pD": [8300],
    "1R_1P_2pD": [8300],
    "1R_2P_1D": [8300],
    "1R_2P_1pD": [8300],
    "2R_1P_1D": [8300, 8400],
    "2R_1P_1pD": [8300, 8400],
}

# Explicit decode port mapping: config_name -> {"D": [ports], "pD": [ports]}
# This ensures precise D vs pD distinction for mixed configurations
DECODE_PORTS_BY_CONFIG = {
    # Pure PD mode (all decode servers are D)
    "1P_3D": {"D": [8200, 8201, 8202], "pD": []},
    "2P_2D": {"D": [8200, 8201], "pD": []},
    "3P_1D": {"D": [8200], "pD": []},
    "1R_1P_2D": {"D": [8200, 8201], "pD": []},
    "1R_2P_1D": {"D": [8200], "pD": []},
    "2R_1P_1D": {"D": [8200], "pD": []},

    # Pure PPD mode (all decode servers are pD)
    "1P_3pD": {"D": [], "pD": [8200, 8201, 8202]},
    "2P_2pD": {"D": [], "pD": [8200, 8201]},
    "3P_1pD": {"D": [], "pD": [8200]},
    "1R_1P_2pD": {"D": [], "pD": [8200, 8201]},
    "1R_2P_1pD": {"D": [], "pD": [8200]},
    "2R_1P_1pD": {"D": [], "pD": [8200]},

    # Mixed D/pD configurations
    "1P_2D_1pD": {"D": [8200, 8201], "pD": [8202]},
    "1P_1D_2pD": {"D": [8200], "pD": [8201, 8202]},
    "2P_1D_1pD": {"D": [8200], "pD": [8201]},
    "1R_1P_1D_1pD": {"D": [8200], "pD": [8201]},
}

app = Quart(__name__)


def _remove_oldest_instances(instances: dict, is_replica: bool = False) -> None:
    """Remove expired instances from the registry."""
    current_time = time.time()
    expired = []
    for key, value in instances.items():
        stamp = value if is_replica else value[1]
        if stamp < current_time:
            expired.append(key)
    for key in expired:
        print(f"[PROXY] Remove expired: {key}")
        instances.pop(key, None)


def _get_port_from_addr(addr: str) -> int:
    """Extract port number from address string (ip:port format)."""
    try:
        return int(addr.split(":")[-1])
    except (ValueError, IndexError):
        return 0


def _is_ppd_port(port: int) -> bool:
    """Check if the given port is configured as pD (prefill-capable decode).

    Uses explicit port mapping from DECODE_PORTS_BY_CONFIG for precise D/pD distinction.
    """
    config = DECODE_PORTS_BY_CONFIG.get(CONFIG_NAME, {})
    ppd_ports = config.get("pD", [])
    return port in ppd_ports


def _listen_for_register(poller, router_socket):
    """Listen for instance registration messages."""
    while True:
        try:
            socks = dict(poller.poll(timeout=1000))
            if router_socket not in socks:
                continue

            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            http_addr = data.get("http_address", "")
            zmq_addr = data.get("zmq_address", "")
            server_type = data.get("type", "")

            stamp = time.time() + DEFAULT_PING_SECONDS

            if server_type == "P":
                with prefill_cv:
                    if http_addr not in prefill_instances:
                        print(f"[PROXY] Add [P] HTTP:{http_addr}, ZMQ:{zmq_addr}")
                    prefill_instances[http_addr] = (zmq_addr, stamp)
                    _remove_oldest_instances(prefill_instances)

            elif server_type == "D":
                # Use explicit port mapping to determine D vs pD
                port = _get_port_from_addr(http_addr)
                is_ppd = _is_ppd_port(port)

                if is_ppd:
                    with ppd_decode_cv:
                        if http_addr not in ppd_decode_instances:
                            print(f"[PROXY] Add [pD] HTTP:{http_addr}, ZMQ:{zmq_addr}")
                        ppd_decode_instances[http_addr] = (zmq_addr, stamp)
                        _remove_oldest_instances(ppd_decode_instances)
                else:
                    with decode_cv:
                        if http_addr not in decode_instances:
                            print(f"[PROXY] Add [D] HTTP:{http_addr}, ZMQ:{zmq_addr}")
                        decode_instances[http_addr] = (zmq_addr, stamp)
                        _remove_oldest_instances(decode_instances)

            elif server_type == "R":
                with replica_cv:
                    if http_addr not in replica_instances:
                        print(f"[PROXY] Add [R] HTTP:{http_addr}")
                    replica_instances[http_addr] = stamp
                    _remove_oldest_instances(replica_instances, is_replica=True)

        except Exception as e:
            print(f"[PROXY] Registration error: {e}")


def start_service_discovery(hostname: str, port: int):
    """Start the ZMQ service discovery listener."""
    if not hostname:
        hostname = socket.gethostname()

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    thread = threading.Thread(
        target=_listen_for_register,
        args=[poller, router_socket],
        daemon=True
    )
    thread.start()
    return thread


def register_static_replicas():
    """Register replica instances based on static configuration.

    Replicas don't use KV transfer, so they don't register via ZMQ.
    We register them statically based on the config name.
    """
    replica_ports = REPLICA_PORTS_BY_CONFIG.get(CONFIG_NAME, [])
    if not replica_ports:
        return

    # Get local IP
    local_ip = socket.gethostbyname(socket.gethostname())

    with replica_cv:
        for port in replica_ports:
            http_addr = f"{local_ip}:{port}"
            # Use a far-future timestamp so they don't expire
            replica_instances[http_addr] = time.time() + 86400 * 365
            print(f"[PROXY] Add [R] HTTP:{http_addr} (static)")


def get_conversation_hash(data: dict) -> str:
    """Generate a hash to identify a conversation."""
    if "messages" in data:
        messages = data.get("messages", [])
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user:
            history_str = first_user.get("content", "")
        else:
            history_str = ""
    else:
        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)

        if "User:" in prompt:
            start = prompt.find("User:")
            end = prompt.find("Assistant:", start)
            if end > start:
                history_str = prompt[start:end].strip()
            else:
                history_str = prompt[start:start+200]
        else:
            history_str = prompt[:200]

    return hashlib.md5(history_str.encode()).hexdigest()


def get_conversation_info(conv_hash: str) -> Tuple[int, Dict[str, str]]:
    """Get turn number and assigned servers for a conversation."""
    with conversation_lock:
        if conv_hash in conversation_state:
            turn_count, _, assigned = conversation_state[conv_hash]
            return turn_count, assigned.copy()
        return 0, {}


def update_conversation(
    conv_hash: str,
    assigned: Optional[Dict[str, str]] = None
) -> Tuple[int, Dict[str, str]]:
    """Update conversation state and return new turn number."""
    with conversation_lock:
        # Clean up expired conversations
        current_time = time.time()
        expired = [k for k, v in conversation_state.items()
                   if current_time - v[1] > CONVERSATION_TTL]
        for k in expired:
            del conversation_state[k]

        if conv_hash in conversation_state:
            turn_count, _, existing_assigned = conversation_state[conv_hash]
            turn_count += 1
            # Merge new assignments with existing
            if assigned:
                existing_assigned.update(assigned)
            final_assigned = existing_assigned
        else:
            turn_count = 1
            final_assigned = assigned or {}

        conversation_state[conv_hash] = (turn_count, current_time, final_assigned)
        return turn_count, final_assigned


def select_servers(
    conv_hash: str,
    current_turn: int,
    assigned: Dict[str, str]
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    Select servers based on configuration and cache affinity.

    Returns: (prefill_addr, decode_addr, replica_addr, routing_mode)
    routing_mode: "pd", "ppd", "ppd_direct", "replica"

    Routing Strategy:
    1. Pure Replica (4R): All requests to replica, hash-based selection
    2. Pure Disagg (1P_3D, 2P_2pD, etc.): All requests through P->D/pD
    3. Hybrid (1R_1P_2pD, etc.): Distribute requests between replica and disagg
       - Use conversation hash to deterministically assign mode
       - Once assigned, cache affinity keeps same mode for Turn 2+
    """
    prefill_addr = None
    decode_addr = None
    replica_addr = None
    routing_mode = "pd"

    # Check available instances
    with prefill_cv:
        prefill_list = list(prefill_instances.keys())
    with decode_cv:
        decode_list = list(decode_instances.keys())
    with ppd_decode_cv:
        ppd_list = list(ppd_decode_instances.keys())
    with replica_cv:
        replica_list = list(replica_instances.keys())

    has_prefill = len(prefill_list) > 0
    has_decode = len(decode_list) > 0 or len(ppd_list) > 0
    has_replica = len(replica_list) > 0

    # Check if this conversation already has an assigned mode
    if assigned:
        if assigned.get("replica") and assigned["replica"] in replica_list:
            # Continue with replica mode
            routing_mode = "replica"
            replica_addr = assigned["replica"]
            return prefill_addr, decode_addr, replica_addr, routing_mode

        if assigned.get("decode"):
            # Continue with disagg mode
            all_decode = decode_list + ppd_list
            if assigned["decode"] in all_decode:
                decode_addr = assigned["decode"]
                if assigned.get("prefill") and assigned["prefill"] in prefill_list:
                    prefill_addr = assigned["prefill"]
                else:
                    prefill_addr = prefill_list[hash(conv_hash) % len(prefill_list)] if prefill_list else None

                if decode_addr in ppd_list:
                    routing_mode = "ppd_direct" if current_turn > 1 else "ppd"
                else:
                    routing_mode = "pd"
                return prefill_addr, decode_addr, replica_addr, routing_mode

    # New conversation - determine routing mode
    if has_replica and not has_prefill and not has_decode:
        # Pure replica mode (4R)
        routing_mode = "replica"
        idx = hash(conv_hash) % len(replica_list)
        replica_addr = replica_list[idx]

    elif has_prefill and has_decode and not has_replica:
        # Pure disaggregated mode (1P_3D, 2P_2pD, 1P_2D_1pD, etc.)
        # Select prefill
        idx = hash(conv_hash) % len(prefill_list)
        prefill_addr = prefill_list[idx]

        # Select decode - distribute across ALL decode servers (D + pD) by capacity
        # This ensures mixed configs like 1P_2D_1pD use all 3 servers, not just pD
        all_decode = decode_list + ppd_list
        decode_idx = hash(conv_hash) % len(all_decode)
        decode_addr = all_decode[decode_idx]

        if decode_addr in ppd_list:
            routing_mode = "ppd"
        else:
            routing_mode = "pd"

    elif has_prefill and has_decode and has_replica:
        # Hybrid mode (1R_1P_2pD, 2R_1P_1D, etc.)
        # Distribute requests between replica and disagg based on capacity
        # Calculate total capacity slots
        num_replica_slots = len(replica_list)
        all_decode = decode_list + ppd_list
        num_disagg_slots = len(all_decode)  # One slot per decode server
        total_slots = num_replica_slots + num_disagg_slots

        # Use hash to select slot
        slot = hash(conv_hash) % total_slots

        if slot < num_replica_slots:
            # Route to replica
            routing_mode = "replica"
            replica_addr = replica_list[slot]
        else:
            # Route to disagg
            decode_idx = slot - num_replica_slots
            if decode_idx < len(ppd_list):
                decode_addr = ppd_list[decode_idx]
                routing_mode = "ppd"
            else:
                decode_addr = decode_list[decode_idx - len(ppd_list)]
                routing_mode = "pd"

            # Select prefill
            prefill_addr = prefill_list[hash(conv_hash) % len(prefill_list)]

    elif has_replica:
        # Only replica available
        routing_mode = "replica"
        idx = hash(conv_hash) % len(replica_list)
        replica_addr = replica_list[idx]

    # For PPD mode, Turn 2+ goes directly to pD
    if routing_mode == "ppd" and current_turn > 1:
        routing_mode = "ppd_direct"

    return prefill_addr, decode_addr, replica_addr, routing_mode


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


async def forward_request(url: str, data: dict, request_id: str):
    """Forward request to vLLM with streaming support."""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            # Always yield the response content, even for error status
            async for chunk_bytes in response.content.iter_any():
                if chunk_bytes:
                    yield chunk_bytes


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """Handle inference requests with automatic routing."""
    try:
        data = await request.get_json()
        conv_hash = get_conversation_hash(data)

        # Get conversation state
        current_turn, assigned = get_conversation_info(conv_hash)

        # Select servers
        prefill_addr, decode_addr, replica_addr, routing_mode = select_servers(
            conv_hash, current_turn + 1, assigned
        )

        # Update conversation state
        new_assigned = {}
        if prefill_addr:
            new_assigned["prefill"] = prefill_addr
        if decode_addr:
            new_assigned["decode"] = decode_addr
        if replica_addr:
            new_assigned["replica"] = replica_addr

        turn_number, final_assigned = update_conversation(conv_hash, new_assigned)

        # Update statistics
        with stats_lock:
            stats["total_requests"] += 1
            if turn_number == 1:
                stats["turn1_requests"] += 1
            else:
                stats["turn2plus_requests"] += 1
                if assigned:
                    stats["cache_affinity_hits"] += 1
                else:
                    stats["cache_affinity_misses"] += 1

            if routing_mode == "pd":
                stats["pd_requests"] += 1
            elif routing_mode == "ppd":
                stats["ppd_requests"] += 1
            elif routing_mode == "ppd_direct":
                stats["ppd_direct_requests"] += 1
            elif routing_mode == "replica":
                stats["replica_requests"] += 1

        is_streaming = data.get("stream", False)

        # Route based on mode
        if routing_mode == "replica":
            # Direct to replica
            request_id = f"replica_{uuid.uuid4().hex}"
            target_url = f"http://{replica_addr}{request.path}"

            print(f"[PROXY] Replica: turn={turn_number}, conv={conv_hash[:8]}, "
                  f"-> R:{replica_addr}")

        elif routing_mode == "ppd_direct":
            # Direct to pD (no prefill needed)
            request_id = f"ppd_direct_{uuid.uuid4().hex}"
            target_url = f"http://{decode_addr}{request.path}"

            print(f"[PROXY] PPD-Direct: turn={turn_number}, conv={conv_hash[:8]}, "
                  f"-> pD:{decode_addr}")

        else:
            # PD or PPD mode - need prefill first
            with prefill_cv:
                prefill_zmq = prefill_instances.get(prefill_addr, ("", 0))[0]
            with decode_cv:
                decode_zmq = decode_instances.get(decode_addr, ("", 0))[0]
            if not decode_zmq:
                with ppd_decode_cv:
                    decode_zmq = ppd_decode_instances.get(decode_addr, ("", 0))[0]

            request_id = (
                f"___prefill_addr_{prefill_zmq}___decode_addr_"
                f"{decode_zmq}_{uuid.uuid4().hex}"
            )

            # Prefill request
            prefill_request = data.copy()
            prefill_request["max_tokens"] = 1
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1

            print(f"[PROXY] {routing_mode.upper()}: turn={turn_number}, "
                  f"conv={conv_hash[:8]}, P:{prefill_addr} -> D:{decode_addr}")

            # Execute prefill
            async for _ in forward_request(
                f"http://{prefill_addr}{request.path}",
                prefill_request,
                request_id
            ):
                continue

            target_url = f"http://{decode_addr}{request.path}"

        # Execute decode/generation
        if is_streaming:
            return Response(
                forward_request(target_url, data, request_id),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Transfer-Encoding": "chunked",
                }
            )
        else:
            response_bytes = b""
            async for chunk in forward_request(target_url, data, request_id):
                response_bytes += chunk
            return response_bytes, 200, {"Content-Type": "application/json"}

    except Exception as e:
        import traceback
        print(f"[PROXY] Error: {e}")
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.route("/status", methods=["GET"])
async def get_status():
    """Get proxy status."""
    with prefill_cv:
        p_list = list(prefill_instances.keys())
    with decode_cv:
        d_list = list(decode_instances.keys())
    with ppd_decode_cv:
        pd_list = list(ppd_decode_instances.keys())
    with replica_cv:
        r_list = list(replica_instances.keys())

    with stats_lock:
        current_stats = stats.copy()

    return {
        "config": CONFIG_NAME,
        "instances": {
            "prefill": p_list,
            "decode": d_list,
            "ppd_decode": pd_list,
            "replica": r_list,
        },
        "conversations": len(conversation_state),
        "stats": current_stats,
    }


@app.route("/stats", methods=["GET"])
async def get_stats():
    """Get detailed statistics."""
    with stats_lock:
        return stats.copy()


@app.route("/conversations", methods=["GET"])
async def get_conversations():
    """Get active conversations."""
    with conversation_lock:
        return {
            "count": len(conversation_state),
            "conversations": [
                {
                    "hash": k[:8],
                    "turn": v[0],
                    "age": int(time.time() - v[1]),
                    "assigned": v[2],
                }
                for k, v in list(conversation_state.items())[:50]
            ]
        }


def main():
    global CONFIG_NAME

    parser = argparse.ArgumentParser(description="Comprehensive Proxy Server")
    parser.add_argument("--config", type=str, default="2P_2D",
                        help="Configuration name (e.g., 2P_2D, 1P_3D, 4R)")
    parser.add_argument("--http-port", type=int, default=10001,
                        help="HTTP port for the proxy")
    parser.add_argument("--zmq-port", type=int, default=30001,
                        help="ZMQ port for service discovery")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")

    args = parser.parse_args()
    CONFIG_NAME = args.config

    # Get decode port configuration
    decode_config = DECODE_PORTS_BY_CONFIG.get(CONFIG_NAME, {"D": [], "pD": []})
    d_ports = decode_config.get("D", [])
    ppd_ports = decode_config.get("pD", [])

    print(f"[PROXY] Starting Comprehensive Proxy")
    print(f"[PROXY] Config: {CONFIG_NAME}")
    print(f"[PROXY] Decode ports: D={d_ports}, pD={ppd_ports}")
    print(f"[PROXY] HTTP: {args.host}:{args.http_port}")
    print(f"[PROXY] ZMQ: {args.host}:{args.zmq_port}")

    # Start service discovery
    start_service_discovery(args.host, args.zmq_port)

    # Register static replicas (if any in this config)
    register_static_replicas()

    # Run the proxy
    app.run(host=args.host, port=args.http_port)


if __name__ == "__main__":
    main()
