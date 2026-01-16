#!/usr/bin/env python3
"""
Optimizer Proxy V2 - Dynamic Mode Selection using RuleBasedSelector

Uses RuleBasedSelector to dynamically decide the optimal mode (PD/PPD/Replica)
for each request based on:
  - input_length, output_length
  - turn_number, has_cache, cached_gpu
  - queue_depths, current_qps
  - optimization objective (ttft/tpot/e2e/throughput)

Architecture (V2: 1P + 1D_pure + 1pD + 1R):
  GPU 0: P (kv_producer)       - Prefill for PD/PPD modes
  GPU 1: D_pure (kv_consumer)  - Decode only, for PD mode
  GPU 2: pD (kv_both)          - Decode + append-prefill, for PPD mode
  GPU 3: Replica (standalone)  - Full model, no KV transfer

Usage:
    python optimizer_proxy_v2.py --http-port 10001 --zmq-port 30001 \
        --prefill-port 8100 --decode-pure-port 8200 \
        --decode-ppd-port 8201 --replica-port 8300
"""

import argparse
import asyncio
import hashlib
import os
import sys
import threading
import time
import uuid
from typing import Any, Optional

import aiohttp
import msgpack
import zmq
from quart import Quart, request, Response

# ============================================================================
# Rate Limiting Configuration
# Semaphores to prevent request bursts and ensure fair scheduling
# Set to 0 to disable rate limiting
# ============================================================================
# Maximum concurrent requests to Prefill GPU (shared by PD and PPD)
MAX_CONCURRENT_PREFILL = 0  # Disabled for now
# Maximum concurrent requests per mode
MAX_CONCURRENT_PD = 0  # Disabled for now
MAX_CONCURRENT_PPD = 0  # Disabled for now
MAX_CONCURRENT_REPLICA = 0  # Disabled for now

# Semaphores (initialized in main if limits > 0)
prefill_semaphore: Optional[asyncio.Semaphore] = None
pd_semaphore: Optional[asyncio.Semaphore] = None
ppd_semaphore: Optional[asyncio.Semaphore] = None
replica_semaphore: Optional[asyncio.Semaphore] = None

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from optimizer.models.rule_based_selector import (
    RuleBasedSelector,
    RequestFeatures,
    OptimizationObjective,
    Mode,
    RoutingDecision,
)

# Global state
request_count = 0

# Instance tracking
prefill_instance: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_pure_instance: dict[str, Any] = {}  # D_pure for PD mode (gpu1)
decode_ppd_instance: dict[str, Any] = {}   # pD for PPD mode (gpu2)

prefill_cv = threading.Condition()
decode_pure_cv = threading.Condition()
decode_ppd_cv = threading.Condition()

# Queue depth tracking (requests in flight per GPU)
queue_depths: dict[str, int] = {"gpu1": 0, "gpu2": 0, "gpu3": 0}
queue_depths_lock = threading.Lock()

# Conversation tracking for cache affinity
# Key: conversation_hash, Value: (turn_count, last_access_time, mode, target_gpu)
conversation_state: dict[str, tuple[int, float, str, str]] = {}
conversation_lock = threading.Lock()

# QPS tracking (sliding window)
qps_window: list[float] = []  # Timestamps of recent requests
qps_lock = threading.Lock()
QPS_WINDOW_SIZE = 10  # seconds

# Statistics
stats = {
    "total_requests": 0,
    "ttft_requests": 0,
    "tpot_requests": 0,
    "e2e_requests": 0,
    "throughput_requests": 0,
    "replica_routed": 0,
    "pd_routed": 0,
    "ppd_routed": 0,
}
stats_lock = threading.Lock()

# Configuration
DEFAULT_PING_SECONDS = 5
CONVERSATION_TTL = 3600  # 1 hour

# Port configuration (set via args)
PREFILL_PORT = 8100
DECODE_PURE_PORT = 8200
DECODE_PPD_PORT = 8201
REPLICA_PORT = 8300

# Port to GPU mapping
PORT_TO_GPU = {
    DECODE_PURE_PORT: "gpu1",  # D_pure
    DECODE_PPD_PORT: "gpu2",   # pD
    REPLICA_PORT: "gpu3",      # Replica
}

# Rule-based selector
selector = RuleBasedSelector(verbose=False)


def get_conversation_hash(request_data: dict) -> str:
    """Get hash for conversation tracking"""
    prompt = ""
    if "messages" in request_data:
        for msg in request_data["messages"]:
            if msg.get("role") == "user":
                prompt += msg.get("content", "")[:200]
    elif "prompt" in request_data:
        prompt = str(request_data["prompt"])[:500]
    return hashlib.md5(prompt.encode()).hexdigest()


def get_objective_from_request(request_data: dict) -> OptimizationObjective:
    """
    Extract optimization objective from request.
    Default: 'e2e'
    """
    objective = request_data.get("objective", "e2e").lower()

    if objective in ["ttft", "time_to_first_token"]:
        return OptimizationObjective.TTFT
    elif objective in ["tpot", "time_per_output_token"]:
        return OptimizationObjective.TPOT
    elif objective in ["throughput"]:
        return OptimizationObjective.THROUGHPUT
    elif objective in ["e2e", "end_to_end", "latency"]:
        return OptimizationObjective.E2E
    else:
        return OptimizationObjective.E2E  # Default


def estimate_input_length(request_data: dict) -> int:
    """Estimate input token count from request."""
    if "messages" in request_data:
        # Chat format: approximate 4 chars per token
        total_chars = sum(len(m.get("content", "")) for m in request_data["messages"])
        return max(1, total_chars // 4)
    elif "prompt" in request_data:
        prompt = request_data["prompt"]
        if isinstance(prompt, list):
            # Token IDs
            return len(prompt)
        else:
            # String: approximate 4 chars per token
            return max(1, len(str(prompt)) // 4)
    return 100  # Default


def get_output_length(request_data: dict) -> int:
    """Get expected output length from request."""
    return request_data.get("max_tokens", request_data.get("max_completion_tokens", 256))


def get_current_qps() -> float:
    """Calculate current QPS from sliding window."""
    current_time = time.time()
    with qps_lock:
        # Remove old entries
        cutoff = current_time - QPS_WINDOW_SIZE
        qps_window[:] = [t for t in qps_window if t > cutoff]

        if len(qps_window) < 2:
            return 1.0  # Low QPS default

        # Calculate QPS
        time_span = qps_window[-1] - qps_window[0]
        if time_span > 0:
            return len(qps_window) / time_span
        return 1.0


def record_request():
    """Record request timestamp for QPS tracking."""
    with qps_lock:
        qps_window.append(time.time())


def get_queue_depths() -> dict[str, int]:
    """Get current queue depths."""
    with queue_depths_lock:
        return dict(queue_depths)


def increment_queue(gpu: str):
    """Increment queue depth for a GPU."""
    with queue_depths_lock:
        queue_depths[gpu] = queue_depths.get(gpu, 0) + 1


def decrement_queue(gpu: str):
    """Decrement queue depth for a GPU."""
    with queue_depths_lock:
        queue_depths[gpu] = max(0, queue_depths.get(gpu, 0) - 1)


def get_conversation_info(conv_hash: str) -> tuple[int, Optional[str]]:
    """Get turn number and cached GPU for a conversation."""
    with conversation_lock:
        if conv_hash in conversation_state:
            turn_count, _, mode, cached_gpu = conversation_state[conv_hash]
            return turn_count, cached_gpu
        return 0, None


def update_conversation(conv_hash: str, mode: str, target_gpu: str) -> int:
    """Update conversation state, return new turn number."""
    with conversation_lock:
        current_time = time.time()
        # Clean old entries
        expired = [k for k, v in conversation_state.items()
                   if current_time - v[1] > CONVERSATION_TTL]
        for k in expired:
            del conversation_state[k]

        # Update
        if conv_hash in conversation_state:
            turn_count = conversation_state[conv_hash][0] + 1
        else:
            turn_count = 1

        conversation_state[conv_hash] = (turn_count, current_time, mode, target_gpu)
        return turn_count


def _listen_for_register(poller, router_socket):
    """Listen for instance registration from vLLM servers."""
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            http_addr = data.get("http_address", "")
            zmq_addr = data.get("zmq_address", "")

            stamp = time.time() + DEFAULT_PING_SECONDS

            # Determine instance type by port
            port = int(http_addr.split(":")[-1]) if ":" in http_addr else 0

            if port == PREFILL_PORT:
                with prefill_cv:
                    prefill_instance[http_addr] = (zmq_addr, stamp)
            elif port == DECODE_PURE_PORT:
                with decode_pure_cv:
                    decode_pure_instance[http_addr] = (zmq_addr, stamp)
            elif port == DECODE_PPD_PORT:
                with decode_ppd_cv:
                    decode_ppd_instance[http_addr] = (zmq_addr, stamp)


def start_zmq_server(hostname: str, port: int):
    """Start ZMQ server for instance registration."""
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    thread = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    thread.start()
    print(f"[Proxy] ZMQ server started on tcp://{hostname}:{port}")


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


async def forward_request(url, data, request_id):
    """Forward request with SSE streaming."""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_any():
                    if chunk_bytes:
                        yield chunk_bytes


async def execute_pd_flow(
    original_request: dict,
    prefill_addr: str,
    prefill_zmq: str,
    decode_addr: str,
    decode_zmq: str,
    request_path: str,
) -> tuple[str, str]:
    """
    Execute PD flow: Prefill on P, then decode on D_pure.
    Returns (target_url, request_id) for the decode phase.
    """
    # Prefill with max_tokens=1
    prefill_request = original_request.copy()
    prefill_request["max_tokens"] = 1
    if "max_completion_tokens" in prefill_request:
        prefill_request["max_completion_tokens"] = 1

    request_id = f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{random_uuid()}"

    # Send prefill request (triggers KV transfer to D)
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"X-Request-Id": request_id}
        async with session.post(
            url=f"http://{prefill_addr}{request_path}",
            json=prefill_request,
            headers=headers
        ) as prefill_resp:
            # Consume prefill response
            async for _ in prefill_resp.content.iter_any():
                pass

    target_url = f"http://{decode_addr}{request_path}"
    return target_url, request_id


async def execute_ppd_flow(
    original_request: dict,
    prefill_addr: str,
    prefill_zmq: str,
    decode_addr: str,
    decode_zmq: str,
    request_path: str,
) -> tuple[str, str]:
    """
    Execute PPD flow: Prefill on P, then decode on pD.
    Returns (target_url, request_id) for the decode phase.
    """
    # Prefill with max_tokens=1
    prefill_request = original_request.copy()
    prefill_request["max_tokens"] = 1
    if "max_completion_tokens" in prefill_request:
        prefill_request["max_completion_tokens"] = 1

    request_id = f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{random_uuid()}"

    # Send prefill request (triggers KV transfer to pD)
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"X-Request-Id": request_id}
        async with session.post(
            url=f"http://{prefill_addr}{request_path}",
            json=prefill_request,
            headers=headers
        ) as prefill_resp:
            # Consume prefill response
            async for _ in prefill_resp.content.iter_any():
                pass

    target_url = f"http://{decode_addr}{request_path}"
    return target_url, request_id


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """Main request handler with dynamic mode selection and rate limiting."""
    target_gpu = None
    mode_semaphore = None
    acquired_prefill = False

    try:
        original_request_data = await request.get_json()

        global request_count
        record_request()

        # Extract features for mode selection
        objective = get_objective_from_request(original_request_data)
        input_length = estimate_input_length(original_request_data)
        output_length = get_output_length(original_request_data)

        conv_hash = get_conversation_hash(original_request_data)
        turn_count, cached_gpu = get_conversation_info(conv_hash)
        turn_number = turn_count + 1  # Next turn

        has_cache = cached_gpu is not None and turn_count > 0
        current_qps = get_current_qps()
        current_queue_depths = get_queue_depths()

        # Build request features
        features = RequestFeatures(
            input_length=input_length,
            output_length=output_length,
            turn_number=turn_number,
            has_cache=has_cache,
            cached_gpu=cached_gpu,
            queue_depths=current_queue_depths,
            objective=objective,
            current_qps=current_qps,
        )

        # Get routing decision from selector
        decision: RoutingDecision = selector.select_mode(features)
        mode = decision.mode
        target_gpu = decision.target_gpu

        # ====================================================================
        # Rate Limiting: Acquire appropriate semaphore before processing
        # This prevents burst requests from overwhelming specific GPUs
        # ====================================================================
        if mode == Mode.PD:
            mode_semaphore = pd_semaphore
            # PD mode needs both prefill and pd semaphores
            if prefill_semaphore:
                await prefill_semaphore.acquire()
                acquired_prefill = True
        elif mode == Mode.PPD:
            mode_semaphore = ppd_semaphore
            # PPD mode needs prefill semaphore (unless D-direct)
            if not (has_cache and cached_gpu == "gpu2" and turn_number > 1):
                if prefill_semaphore:
                    await prefill_semaphore.acquire()
                    acquired_prefill = True
        else:
            mode_semaphore = replica_semaphore

        if mode_semaphore:
            await mode_semaphore.acquire()
        # ====================================================================

        # Update stats
        with stats_lock:
            stats["total_requests"] += 1
            stats[f"{objective.value}_requests"] += 1
            stats[f"{mode.value}_routed"] += 1

        # Increment queue depth
        increment_queue(target_gpu)

        # Execute the routing decision
        if mode == Mode.REPLICA:
            # Direct to Replica
            target_url = f"http://localhost:{REPLICA_PORT}{request.path}"
            request_id = f"replica_{random_uuid()}"

            turn_number = update_conversation(conv_hash, "replica", target_gpu)
            print(f"[Proxy] {objective.value.upper()}: turn={turn_number}, conv={conv_hash[:8]} → Replica:{REPLICA_PORT} ({decision.reason})")

        elif mode == Mode.PD:
            # PD mode: P→D_pure
            with prefill_cv:
                if not prefill_instance:
                    decrement_queue(target_gpu)
                    return {"error": "No prefill instance available"}, 503
                prefill_addr = list(prefill_instance.keys())[0]
                prefill_zmq = prefill_instance[prefill_addr][0]

            with decode_pure_cv:
                if not decode_pure_instance:
                    decrement_queue(target_gpu)
                    return {"error": "No pure decode instance available"}, 503
                decode_addr = list(decode_pure_instance.keys())[0]
                decode_zmq = decode_pure_instance[decode_addr][0]

            turn_number = update_conversation(conv_hash, "pd", target_gpu)
            print(f"[Proxy] {objective.value.upper()}: turn={turn_number}, conv={conv_hash[:8]} → PD P:{prefill_addr} D:{decode_addr} ({decision.reason})")

            target_url, request_id = await execute_pd_flow(
                original_request_data,
                prefill_addr, prefill_zmq,
                decode_addr, decode_zmq,
                request.path,
            )

        elif mode == Mode.PPD:
            # Check if we can use D-direct mode (cache reuse on pD)
            if has_cache and cached_gpu == "gpu2" and turn_number > 1:
                # D-direct: send directly to pD for prefix cache reuse
                target_url = f"http://localhost:{DECODE_PPD_PORT}{request.path}"
                request_id = f"ppd_direct_{random_uuid()}"

                turn_number = update_conversation(conv_hash, "ppd", target_gpu)
                print(f"[Proxy] {objective.value.upper()}: turn={turn_number}, conv={conv_hash[:8]} → PPD D-direct:{DECODE_PPD_PORT} ({decision.reason})")
            else:
                # PPD mode: P→pD
                with prefill_cv:
                    if not prefill_instance:
                        decrement_queue(target_gpu)
                        return {"error": "No prefill instance available"}, 503
                    prefill_addr = list(prefill_instance.keys())[0]
                    prefill_zmq = prefill_instance[prefill_addr][0]

                with decode_ppd_cv:
                    if not decode_ppd_instance:
                        decrement_queue(target_gpu)
                        return {"error": "No PPD decode instance available"}, 503
                    decode_addr = list(decode_ppd_instance.keys())[0]
                    decode_zmq = decode_ppd_instance[decode_addr][0]

                turn_number = update_conversation(conv_hash, "ppd", target_gpu)
                print(f"[Proxy] {objective.value.upper()}: turn={turn_number}, conv={conv_hash[:8]} → PPD P:{prefill_addr} pD:{decode_addr} ({decision.reason})")

                target_url, request_id = await execute_ppd_flow(
                    original_request_data,
                    prefill_addr, prefill_zmq,
                    decode_addr, decode_zmq,
                    request.path,
                )

        else:
            # Fallback to Replica
            target_url = f"http://localhost:{REPLICA_PORT}{request.path}"
            request_id = f"fallback_{random_uuid()}"
            turn_number = update_conversation(conv_hash, "replica", "gpu3")

        # Forward the actual request
        request_count += 1
        is_streaming = original_request_data.get("stream", False)

        # Remove our custom 'objective' field before forwarding
        forward_data = {k: v for k, v in original_request_data.items() if k != "objective"}

        # Helper to release semaphores
        def release_semaphores():
            nonlocal acquired_prefill, mode_semaphore
            if acquired_prefill and prefill_semaphore:
                prefill_semaphore.release()
            if mode_semaphore:
                mode_semaphore.release()

        if is_streaming:
            # Capture semaphore state for the generator
            _acquired_prefill = acquired_prefill
            _mode_semaphore = mode_semaphore

            async def streaming_response():
                try:
                    async for chunk in forward_request(target_url, forward_data, request_id):
                        yield chunk
                finally:
                    if target_gpu:
                        decrement_queue(target_gpu)
                    # Release semaphores after streaming completes
                    if _acquired_prefill and prefill_semaphore:
                        prefill_semaphore.release()
                    if _mode_semaphore:
                        _mode_semaphore.release()

            # Mark as released so exception handler doesn't double-release
            acquired_prefill = False
            mode_semaphore = None

            return Response(
                streaming_response(),
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
            async for chunk in forward_request(target_url, forward_data, request_id):
                response_bytes += chunk
            decrement_queue(target_gpu)
            release_semaphores()
            return response_bytes, 200, {"Content-Type": "application/json"}

    except Exception as e:
        if target_gpu:
            decrement_queue(target_gpu)
        # Release semaphores on error
        if acquired_prefill and prefill_semaphore:
            prefill_semaphore.release()
        if mode_semaphore:
            mode_semaphore.release()
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.route("/status", methods=["GET"])
async def status():
    """Return proxy status."""
    with prefill_cv:
        p_count = len(prefill_instance)
    with decode_pure_cv:
        d_pure_count = len(decode_pure_instance)
    with decode_ppd_cv:
        d_ppd_count = len(decode_ppd_instance)

    return {
        "status": "running",
        "mode": "optimizer_v2_dynamic",
        "instances": {
            "prefill (gpu0)": p_count,
            "decode_pure (gpu1)": d_pure_count,
            "decode_ppd (gpu2)": d_ppd_count,
            "replica (gpu3)": 1,
        },
        "queue_depths": get_queue_depths(),
        "current_qps": get_current_qps(),
        "routing": "Dynamic via RuleBasedSelector",
    }


@app.route("/stats", methods=["GET"])
async def get_stats():
    """Return routing statistics."""
    with stats_lock:
        return dict(stats)


@app.route("/stats/reset", methods=["POST"])
async def reset_stats():
    """Reset statistics."""
    with stats_lock:
        for key in stats:
            stats[key] = 0
    return {"status": "reset"}


@app.route("/warmup", methods=["POST"])
async def warmup():
    """
    Send warmup requests to trigger NCCL initialization.
    """
    warmup_data = {
        "model": "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B",
        "prompt": "Hello",
        "max_tokens": 5,
        "stream": False,
    }

    results = {}

    # Warmup Replica path
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(
                f"http://localhost:{REPLICA_PORT}/v1/completions",
                json=warmup_data
            ) as resp:
                results["replica"] = resp.status == 200
    except Exception as e:
        results["replica"] = f"error: {e}"

    # Warmup PD path
    try:
        warmup_data["objective"] = "tpot"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(
                f"http://localhost:{OPTIMIZER_PROXY_PORT}/v1/completions",
                json=warmup_data
            ) as resp:
                results["pd"] = resp.status == 200
    except Exception as e:
        results["pd"] = f"error: {e}"

    # Warmup PPD path
    try:
        warmup_data["objective"] = "e2e"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(
                f"http://localhost:{OPTIMIZER_PROXY_PORT}/v1/completions",
                json=warmup_data
            ) as resp:
                results["ppd"] = resp.status == 200
    except Exception as e:
        results["ppd"] = f"error: {e}"

    return {"warmup": results}


# Global for self-warmup
OPTIMIZER_PROXY_PORT = 10001


def main():
    global PREFILL_PORT, DECODE_PURE_PORT, DECODE_PPD_PORT, REPLICA_PORT, OPTIMIZER_PROXY_PORT
    global PORT_TO_GPU
    global prefill_semaphore, pd_semaphore, ppd_semaphore, replica_semaphore

    # Initialize rate limiting semaphores (only if limits > 0)
    # These prevent request bursts from overwhelming specific GPUs
    if MAX_CONCURRENT_PREFILL > 0:
        prefill_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PREFILL)
    if MAX_CONCURRENT_PD > 0:
        pd_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PD)
    if MAX_CONCURRENT_PPD > 0:
        ppd_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PPD)
    if MAX_CONCURRENT_REPLICA > 0:
        replica_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REPLICA)

    parser = argparse.ArgumentParser(description="Optimizer Proxy V2 with Dynamic Mode Selection")
    parser.add_argument("--http-port", type=int, default=10001, help="HTTP port")
    parser.add_argument("--zmq-port", type=int, default=30001, help="ZMQ port for registration")
    parser.add_argument("--prefill-port", type=int, default=8100, help="Prefill server port (GPU0)")
    parser.add_argument("--decode-pure-port", type=int, default=8200, help="Pure decode server port (GPU1)")
    parser.add_argument("--decode-ppd-port", type=int, default=8201, help="PPD decode server port (GPU2)")
    parser.add_argument("--replica-port", type=int, default=8300, help="Replica server port (GPU3)")
    args = parser.parse_args()

    PREFILL_PORT = args.prefill_port
    DECODE_PURE_PORT = args.decode_pure_port
    DECODE_PPD_PORT = args.decode_ppd_port
    REPLICA_PORT = args.replica_port
    OPTIMIZER_PROXY_PORT = args.http_port

    # Update port to GPU mapping
    PORT_TO_GPU = {
        DECODE_PURE_PORT: "gpu1",
        DECODE_PPD_PORT: "gpu2",
        REPLICA_PORT: "gpu3",
    }

    # Start ZMQ server
    start_zmq_server("0.0.0.0", args.zmq_port)

    print(f"[Proxy] Optimizer V2 Proxy (Dynamic) starting on http://0.0.0.0:{args.http_port}")
    print(f"[Proxy] Architecture:")
    print(f"  GPU0: P (prefill) - port {PREFILL_PORT}")
    print(f"  GPU1: D_pure (PD decode) - port {DECODE_PURE_PORT}")
    print(f"  GPU2: pD (PPD decode) - port {DECODE_PPD_PORT}")
    print(f"  GPU3: Replica - port {REPLICA_PORT}")
    print(f"[Proxy] Routing: Dynamic via RuleBasedSelector")
    print(f"[Proxy] Decision factors: objective, input/output length, turn, cache, QPS, queue depth")
    print(f"[Proxy] Rate Limiting: Prefill={MAX_CONCURRENT_PREFILL}, PD={MAX_CONCURRENT_PD}, PPD={MAX_CONCURRENT_PPD}, Replica={MAX_CONCURRENT_REPLICA}")

    app.run(host="0.0.0.0", port=args.http_port, debug=False)


if __name__ == "__main__":
    main()
