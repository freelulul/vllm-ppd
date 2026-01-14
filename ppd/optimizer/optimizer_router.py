#!/usr/bin/env python3
"""
Unified Optimizer Router

Main entry point for intelligent mode selection.
Combines rule-based and ML-based selectors with:
- Cache affinity handling
- Queue-aware load balancing
- Optimization objective support
"""

import asyncio
import hashlib
import time
import json
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
from quart import Quart, request, Response

# Import selectors
from .models.rule_based_selector import RuleBasedSelector, RequestFeatures as RuleFeatures
from .models.rule_based_selector import OptimizationObjective, Mode, RoutingDecision


@dataclass
class ServerConfig:
    """Configuration for a server endpoint"""
    name: str
    url: str
    gpu_id: str
    mode: str  # "pd", "ppd", "replica"
    is_decode: bool = False  # True for Decode GPU in PD/PPD


@dataclass
class ServerState:
    """Runtime state for a server"""
    config: ServerConfig
    queue_depth: int = 0
    active_requests: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    last_request_time: float = 0.0


@dataclass
class ConversationInfo:
    """Tracking info for a conversation"""
    conversation_id: str
    turn_count: int
    cached_gpu: str
    mode_history: List[str]
    created_at: float
    last_access: float


class SelectorType(Enum):
    RULE_BASED = "rule"
    XGBOOST = "xgboost"
    HYBRID = "hybrid"  # Use rules first, fallback to XGBoost


class OptimizerRouter:
    """
    Main optimizer router that handles all incoming requests.

    Architecture (1P + 1D + 2R):
    - GPU 0: Prefill (P) - used with PD/PPD modes
    - GPU 1: Decode (D) - used with PD/PPD modes
    - GPU 2: Replica 0 (R0)
    - GPU 3: Replica 1 (R1)

    Routing Strategy:
    1. Extract conversation ID
    2. Check cache affinity (Turn 2+)
    3. If Turn 1 or no strong affinity: apply selector logic
    4. Route to selected server
    """

    DEFAULT_SERVERS = {
        "pd_proxy": ServerConfig("pd_proxy", "http://localhost:10001", "gpu0+1", "pd"),
        "ppd_proxy": ServerConfig("ppd_proxy", "http://localhost:10001", "gpu0+1", "ppd"),
        "decode_direct": ServerConfig("decode", "http://localhost:8200", "gpu1", "ppd", is_decode=True),
        "replica0": ServerConfig("replica0", "http://localhost:8300", "gpu2", "replica"),
        "replica1": ServerConfig("replica1", "http://localhost:8400", "gpu3", "replica"),
    }

    def __init__(
        self,
        selector_type: SelectorType = SelectorType.RULE_BASED,
        default_objective: OptimizationObjective = OptimizationObjective.TTFT,
        verbose: bool = False
    ):
        self.selector_type = selector_type
        self.default_objective = default_objective
        self.verbose = verbose

        # Initialize selectors
        self.rule_selector = RuleBasedSelector(verbose=verbose)
        self.xgboost_selector = None

        # Try to load XGBoost model
        if selector_type in [SelectorType.XGBOOST, SelectorType.HYBRID]:
            self._load_xgboost_model()

        # Server states
        self.servers: Dict[str, ServerState] = {}
        for name, config in self.DEFAULT_SERVERS.items():
            self.servers[name] = ServerState(config=config)

        # Conversation tracking
        self.conversations: Dict[str, ConversationInfo] = {}

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "mode_counts": {"pd": 0, "ppd": 0, "replica": 0},
            "objective_counts": {},
        }

    def _load_xgboost_model(self):
        """Try to load pre-trained XGBoost model"""
        try:
            from models.xgboost_selector import XGBoostSelector
            model_path = Path(__file__).parent / "models" / "xgboost_model"
            if model_path.with_suffix('.json').exists():
                self.xgboost_selector = XGBoostSelector(str(model_path))
                print(f"Loaded XGBoost model from {model_path}")
            else:
                print("XGBoost model not found, using rule-based only")
        except ImportError:
            print("XGBoost not available, using rule-based only")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Router] {msg}")

    def get_conversation_id(self, data: dict) -> str:
        """
        Extract stable conversation ID from request.

        Priority:
        1. Explicit session_id
        2. First user message in chat format
        3. First "User:" block in completion format
        4. Hash of first 500 chars
        """
        # Option 1: Explicit session_id
        if "session_id" in data:
            return data["session_id"]

        # Option 2: Chat format - first user message
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
            start = prompt.find("User:") + 5
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

        # Option 4: Fallback - hash of first 500 chars
        return hashlib.md5(prompt[:500].encode()).hexdigest()

    def _get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depths for all GPUs"""
        return {
            "gpu1": self.servers["decode_direct"].queue_depth,
            "gpu2": self.servers["replica0"].queue_depth,
            "gpu3": self.servers["replica1"].queue_depth,
        }

    def _estimate_input_length(self, data: dict) -> int:
        """Estimate input length from request data"""
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, str):
                # Rough estimate: 1 token ≈ 4 chars
                return len(prompt) // 4
            elif isinstance(prompt, list):
                return sum(len(str(p)) // 4 for p in prompt)
        elif "messages" in data:
            total = sum(len(m.get("content", "")) for m in data["messages"])
            return total // 4
        return 100  # default

    def _estimate_output_length(self, data: dict) -> int:
        """Estimate output length from request data"""
        return data.get("max_tokens", 100)

    def route_request(self, data: dict, objective: Optional[OptimizationObjective] = None) -> Tuple[str, RoutingDecision]:
        """
        Main routing logic.

        Args:
            data: Request data
            objective: Optimization objective (uses default if not specified)

        Returns:
            (server_url, routing_decision)
        """
        self.stats["total_requests"] += 1

        # Get conversation info
        conv_id = self.get_conversation_id(data)
        conv_info = self.conversations.get(conv_id)

        # Determine turn number
        if conv_info:
            turn_number = conv_info.turn_count + 1
            has_cache = True
            cached_gpu = conv_info.cached_gpu
        else:
            turn_number = 1
            has_cache = False
            cached_gpu = None

        # Use provided objective or default
        if objective is None:
            objective = self.default_objective

        # Update stats
        obj_key = objective.value
        self.stats["objective_counts"][obj_key] = self.stats["objective_counts"].get(obj_key, 0) + 1

        self._log(f"Request: conv={conv_id[:8]}, turn={turn_number}, obj={objective.value}")
        self._log(f"  has_cache={has_cache}, cached_gpu={cached_gpu}")

        # Build features for selector
        features = RuleFeatures(
            input_length=self._estimate_input_length(data),
            output_length=self._estimate_output_length(data),
            turn_number=turn_number,
            has_cache=has_cache,
            cached_gpu=cached_gpu,
            queue_depths=self._get_queue_depths(),
            objective=objective
        )

        # Select mode using configured selector
        if self.selector_type == SelectorType.XGBOOST and self.xgboost_selector:
            from models.xgboost_selector import RequestFeatures as XGBFeatures
            xgb_features = XGBFeatures(
                input_length=features.input_length,
                output_length=features.output_length,
                turn_number=features.turn_number,
                has_cache=features.has_cache,
                cached_gpu=features.cached_gpu,
                queue_depths=features.queue_depths,
                objective=objective
            )
            decision = self.xgboost_selector.select_mode(xgb_features)
        else:
            decision = self.rule_selector.select_mode(features)

        # Determine actual server URL based on decision
        mode = decision.mode.value
        target_gpu = decision.target_gpu

        if mode == "pd":
            server_url = self.servers["pd_proxy"].config.url
            actual_gpu = "gpu1"
        elif mode == "ppd":
            if turn_number >= 2 and has_cache and cached_gpu == "gpu1":
                # Direct to decode for Turn 2+ with PPD
                server_url = self.servers["decode_direct"].config.url
            else:
                server_url = self.servers["ppd_proxy"].config.url
            actual_gpu = "gpu1"
        else:  # replica
            if target_gpu == "gpu2":
                server_url = self.servers["replica0"].config.url
                actual_gpu = "gpu2"
            else:
                server_url = self.servers["replica1"].config.url
                actual_gpu = "gpu3"

        # Update conversation tracking
        current_time = time.time()
        if conv_info:
            conv_info.turn_count = turn_number
            conv_info.last_access = current_time
            conv_info.mode_history.append(mode)
            if not has_cache or cached_gpu != actual_gpu:
                # Cache location changed
                conv_info.cached_gpu = actual_gpu
                self.stats["cache_misses"] += 1
            else:
                self.stats["cache_hits"] += 1
        else:
            self.conversations[conv_id] = ConversationInfo(
                conversation_id=conv_id,
                turn_count=1,
                cached_gpu=actual_gpu,
                mode_history=[mode],
                created_at=current_time,
                last_access=current_time
            )

        # Update mode counts
        self.stats["mode_counts"][mode] += 1

        self._log(f"  → Decision: {mode} @ {server_url}")

        return server_url, decision

    def get_stats(self) -> Dict:
        """Get router statistics"""
        total = self.stats["total_requests"]
        cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]

        return {
            "total_requests": total,
            "cache_hit_rate": self.stats["cache_hits"] / cache_total if cache_total > 0 else 0,
            "mode_distribution": {
                k: v / total if total > 0 else 0
                for k, v in self.stats["mode_counts"].items()
            },
            "objective_distribution": self.stats["objective_counts"],
            "active_conversations": len(self.conversations),
            "selector_type": self.selector_type.value,
        }


# ============================================================================
# HTTP Server Implementation
# ============================================================================

app = Quart(__name__)
router: Optional[OptimizerRouter] = None
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=3600)


async def forward_request(url: str, data: dict):
    """Forward request to selected server"""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(f"{url}/v1/completions", json=data) as response:
            if response.status == 200:
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """Main request handler"""
    data = await request.get_json()

    # Extract optimization objective from request (optional)
    obj_str = data.pop("optimization_objective", None)
    if obj_str:
        try:
            objective = OptimizationObjective(obj_str)
        except ValueError:
            objective = None
    else:
        objective = None

    # Route request
    server_url, decision = router.route_request(data, objective)

    # Forward to selected server
    is_streaming = data.get("stream", False)
    if is_streaming:
        return Response(
            forward_request(server_url, data),
            mimetype="text/event-stream"
        )
    else:
        response_bytes = b""
        async for chunk in forward_request(server_url, data):
            response_bytes += chunk
        return response_bytes, 200, {"Content-Type": "application/json"}


@app.route("/stats", methods=["GET"])
async def get_stats():
    """Get router statistics"""
    return router.get_stats()


@app.route("/config", methods=["GET"])
async def get_config():
    """Get router configuration"""
    return {
        "selector_type": router.selector_type.value,
        "default_objective": router.default_objective.value,
        "servers": {
            name: {
                "url": state.config.url,
                "gpu": state.config.gpu_id,
                "mode": state.config.mode
            }
            for name, state in router.servers.items()
        }
    }


@app.route("/conversations", methods=["GET"])
async def get_conversations():
    """Get active conversations"""
    return {
        "count": len(router.conversations),
        "conversations": [
            {
                "id": conv.conversation_id[:8],
                "turns": conv.turn_count,
                "cached_gpu": conv.cached_gpu,
                "modes": conv.mode_history[-5:]  # Last 5 modes
            }
            for conv in list(router.conversations.values())[:20]  # Limit to 20
        ]
    }


def main():
    global router

    import argparse
    parser = argparse.ArgumentParser(description="Optimizer Router Server")
    parser.add_argument("--port", type=int, default=10000,
                        help="HTTP port (default: 10000)")
    parser.add_argument("--selector", choices=["rule", "xgboost", "hybrid"],
                        default="rule", help="Selector type")
    parser.add_argument("--objective", choices=["ttft", "tpot", "throughput", "e2e"],
                        default="ttft", help="Default optimization objective")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    # Initialize router
    router = OptimizerRouter(
        selector_type=SelectorType(args.selector),
        default_objective=OptimizationObjective(args.objective),
        verbose=args.verbose
    )

    print("=" * 60)
    print("Optimizer Router Server")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Selector: {args.selector}")
    print(f"Default Objective: {args.objective}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
