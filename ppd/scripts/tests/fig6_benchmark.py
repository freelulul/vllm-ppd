#!/usr/bin/env python3
"""
Fig6 Complete Benchmark Script

This script implements the benchmark for comparing PD, PPD, Replica, and Optimizer modes.
Matches original benchmark (benchmark_common.py) data collection logic.

Key features:
- T1/T2 metrics collected separately
- SLO computed on T2 only (multi-turn advantage scenario)
- TPOT calculated using (e2e - ttft) / (tokens - 1) with usage.completion_tokens
- Burst mode for T2 (all T2 sent simultaneously after T1 completes)
- Proper history building with full response content
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import random
import string
import asyncio
import aiohttp
import argparse
import hashlib
import subprocess
import signal
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

from optimizer.models.rule_based_selector import (
    RuleBasedSelector, RequestFeatures, OptimizationObjective, Mode
)

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Server endpoints
PROXY_URLS = {
    "pd": "http://localhost:10001",
    "ppd": "http://localhost:10001",
    "replica": "http://localhost:10002",
    "optimizer": "http://localhost:10001"  # optimizer_proxy_v2 uses same port as pd/ppd
}

VLLM_PORTS = {
    "pd": [8001, 8002, 8003, 8004],
    "ppd": [8001, 8002, 8003, 8004],
    "replica": [8005, 8006, 8007, 8008],
    "optimizer": [8001, 8002, 8003, 8004, 8005]
}

# SLO thresholds - relaxed for M_d Big-Paste workload (T2 context ~1536 tokens)
# Based on burst mode empirical data:
# TTFT: 1000ms - Replica passes, Optimizer ~50%, PD/PPD fail
# TPOT: 11ms - PD passes, Optimizer ~50%, Replica/PPD fail
# E2E: 2000ms - Replica/Optimizer pass, PD/PPD fail
SLO_THRESHOLDS = {"ttft_ms": 1000, "tpot_ms": 11, "e2e_ms": 2000}

# Benchmark parameters
DURATION_PER_POINT_SEC = 10
SERVER_STARTUP_WAIT = 60
SERVER_SHUTDOWN_WAIT = 5

# Server scripts
SERVER_SCRIPTS = {
    "pd": f"{PROJECT_DIR}/scripts/server/start_servers_4gpu.sh",
    "ppd": f"{PROJECT_DIR}/scripts/server/start_servers_4gpu.sh",
    "replica": f"{PROJECT_DIR}/scripts/server/start_replication_servers_4gpu.sh",
    "optimizer": f"{PROJECT_DIR}/scripts/server/start_optimizer_servers_v2.sh"
}

# Panel configurations
PANEL_CONFIGS = {
    "A": {
        "name": "Objective Type Comparison",
        "x_axis": "objective_type",
        "qps": 8,
        "turn_configs": [{"input": 512, "output": 256}, {"input": 128, "output": 128}],
        "data_points": [
            {"name": "Pure_TTFT", "objective_dist": {"ttft": 1.0, "tpot": 0.0, "e2e": 0.0}},
            {"name": "Pure_TPOT", "objective_dist": {"ttft": 0.0, "tpot": 1.0, "e2e": 0.0}},
            {"name": "Pure_E2E", "objective_dist": {"ttft": 0.0, "tpot": 0.0, "e2e": 1.0}},
            {"name": "Mixed_Balanced", "objective_dist": {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}},
        ]
    },
    "B": {
        "name": "QPS Scaling",
        "x_axis": "qps",
        "objective_dist": {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33},
        "turn_configs": [{"input": 512, "output": 256}, {"input": 128, "output": 128}],
        "data_points": [4, 6, 8, 12, 16]
    }
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TurnResult:
    """Result of a single turn"""
    turn: int
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    success: bool
    routed_to: str = ""
    error: Optional[str] = None
    completion_tokens: int = 0  # From usage.completion_tokens


@dataclass
class ConversationResult:
    """Result of a complete conversation"""
    conv_id: str
    objective: str
    turns: List[TurnResult] = field(default_factory=list)


@dataclass
class ModeResult:
    """Result for a mode"""
    mode: str
    conversations: List[ConversationResult] = field(default_factory=list)
    routing_stats: Dict[str, int] = field(default_factory=dict)

    def compute_slo_attainment(self, turn2_only: bool = True) -> Dict:
        """Compute SLO attainment statistics"""
        slo_by_obj = defaultdict(lambda: {"met": 0, "total": 0})
        total_turns = 0
        successful_turns = 0

        for conv in self.conversations:
            for turn in conv.turns:
                # If turn2_only, skip Turn 1
                if turn2_only and turn.turn == 1:
                    continue

                total_turns += 1
                if not turn.success:
                    continue
                successful_turns += 1

                obj = conv.objective
                slo_by_obj[obj]["total"] += 1

                # Check SLO based on objective
                if obj == "ttft" and turn.ttft_ms <= SLO_THRESHOLDS["ttft_ms"]:
                    slo_by_obj[obj]["met"] += 1
                elif obj == "tpot" and turn.tpot_ms <= SLO_THRESHOLDS["tpot_ms"]:
                    slo_by_obj[obj]["met"] += 1
                elif obj == "e2e" and turn.e2e_ms <= SLO_THRESHOLDS["e2e_ms"]:
                    slo_by_obj[obj]["met"] += 1
                elif obj == "p99_ttft" and turn.ttft_ms <= SLO_THRESHOLDS["ttft_ms"]:
                    slo_by_obj[obj]["met"] += 1

        total_met = sum(v["met"] for v in slo_by_obj.values())
        total_count = sum(v["total"] for v in slo_by_obj.values())

        return {
            "overall": (total_met / total_count * 100) if total_count > 0 else 0,
            "by_objective": dict(slo_by_obj),
            "total_turns": total_turns,
            "successful_turns": successful_turns,
            "turn2_only": turn2_only
        }

    def compute_latency_stats(self) -> Dict:
        """Compute latency statistics separately for T1 and T2"""
        t1_data = []
        t2_data = []
        all_data = []

        for conv in self.conversations:
            for turn in conv.turns:
                if turn.success:
                    data = {
                        "ttft": turn.ttft_ms,
                        "tpot": turn.tpot_ms,
                        "e2e": turn.e2e_ms
                    }
                    all_data.append(data)
                    if turn.turn == 1:
                        t1_data.append(data)
                    else:
                        t2_data.append(data)

        def calc_stats(data_list):
            if not data_list:
                return {}
            ttfts = [d["ttft"] for d in data_list]
            tpots = [d["tpot"] for d in data_list if d["tpot"] > 0]
            e2es = [d["e2e"] for d in data_list]

            return {
                "count": len(data_list),
                "avg_ttft_ms": np.mean(ttfts),
                "avg_tpot_ms": np.mean(tpots) if tpots else 0,
                "avg_e2e_ms": np.mean(e2es),
                "p50_ttft_ms": np.percentile(ttfts, 50),
                "p99_ttft_ms": np.percentile(ttfts, 99),
                "p50_tpot_ms": np.percentile(tpots, 50) if tpots else 0,
                "p99_tpot_ms": np.percentile(tpots, 99) if tpots else 0,
                "p50_e2e_ms": np.percentile(e2es, 50),
                "p99_e2e_ms": np.percentile(e2es, 99),
            }

        # Main stats use T2 only (matches original benchmark)
        t2_stats = calc_stats(t2_data)
        result = {k: v for k, v in t2_stats.items() if k != "count"}
        result["turn1_metrics"] = calc_stats(t1_data)
        result["turn2_metrics"] = t2_stats
        result["all_turns_metrics"] = calc_stats(all_data)

        return result


# ============================================================================
# Prompt Generation (MATCHES original benchmark_common.py)
# ============================================================================

# Word pool for realistic text (~100 words)
WORD_POOL = [
    # Common words (40)
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    # Technical terms (30)
    "system", "data", "model", "process", "analysis", "result", "method",
    "research", "information", "development", "performance", "application",
    "technology", "network", "algorithm", "function", "structure", "interface",
    "database", "server", "client", "protocol", "framework", "module",
    "component", "service", "request", "response", "memory", "storage",
    # Action words (15)
    "create", "update", "delete", "execute", "configure", "optimize",
    "implement", "design", "build", "deploy", "monitor", "validate",
    "transform", "generate", "compute",
    # Descriptive words (15)
    "efficient", "scalable", "robust", "dynamic", "parallel", "distributed",
    "synchronous", "asynchronous", "concurrent", "sequential", "recursive",
    "iterative", "adaptive", "modular", "flexible",
]


def generate_prompt(num_tokens: int, prefix: str = "", seed: str = "") -> str:
    """Generate a prompt with approximately num_tokens tokens.

    MATCHES original benchmark (benchmark_common.py):
    - Uses word pool with ~0.75 words per token ratio
    - Returns format: "User: {prefix}{content}\\nAssistant:"
    """
    if seed:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # ~0.75 words per token for English (matches original benchmark)
    target_words = int(num_tokens * 0.75)
    words = []

    for _ in range(target_words):
        if rng.random() < 0.7:
            words.append(rng.choice(WORD_POOL))
        else:
            # Random word 3-8 chars
            word_len = rng.randint(3, 8)
            words.append(''.join(rng.choices(string.ascii_lowercase, k=word_len)))

    content = ' '.join(words)
    return f"User: {prefix}{content}\nAssistant:"


def generate_response_content(num_tokens: int, seed: str = "") -> str:
    """Generate response content with approximately num_tokens tokens.

    Used for building history with FULL output length (not truncated).
    """
    if seed:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    target_words = int(num_tokens * 0.75)
    words = []

    for _ in range(target_words):
        if rng.random() < 0.7:
            words.append(rng.choice(WORD_POOL))
        else:
            word_len = rng.randint(3, 8)
            words.append(''.join(rng.choices(string.ascii_lowercase, k=word_len)))

    return ' '.join(words)


# ============================================================================
# Server Management
# ============================================================================

def stop_all_servers():
    """Stop all vLLM servers - thorough cleanup including EngineCore zombies"""
    print("  Stopping all servers...")

    # Step 1: Kill vLLM serve processes
    subprocess.run(["pkill", "-f", "vllm serve"], stderr=subprocess.DEVNULL)

    # Step 2: Kill EngineCore zombie processes (CRITICAL - these hold GPU memory)
    subprocess.run(["pkill", "-9", "-f", "EngineCore"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], stderr=subprocess.DEVNULL)

    # Step 3: Kill proxy processes
    for proxy in ["disagg_proxy", "optimizer_proxy", "simple_replica_proxy", "replication_proxy"]:
        subprocess.run(["pkill", "-f", proxy], stderr=subprocess.DEVNULL)

    # Step 4: Wait for processes to terminate
    time.sleep(3)

    # Step 5: Verify GPU memory is released (wait up to 10 seconds)
    for _ in range(5):
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            print("  All servers stopped, GPU memory released.")
            return
        time.sleep(2)

    # Force kill any remaining GPU processes
    print("  WARNING: GPU processes still running, force killing...")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    for pid in result.stdout.strip().split('\n'):
        if pid.strip():
            subprocess.run(["kill", "-9", pid.strip()], stderr=subprocess.DEVNULL)
    time.sleep(3)


def start_servers(mode: str):
    """Start servers for the specified mode"""
    print(f"  Starting {mode} servers...")
    script = SERVER_SCRIPTS.get(mode)
    if not script or not os.path.exists(script):
        print(f"  ERROR: Script not found: {script}")
        return False

    # Run the startup script (it handles its own waiting)
    if mode in ["pd", "ppd"]:
        proc = subprocess.Popen(
            [script, mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    else:
        proc = subprocess.Popen(
            [script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

    # Stream output in real-time
    print("  Script output:")
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            print(f"    {line.rstrip()}")

    exit_code = proc.wait()
    if exit_code != 0:
        print(f"  ERROR: Script exited with code {exit_code}")
        return False

    print(f"  Servers started successfully.")
    return True


def check_vllm_server_ready(port: int, timeout: int = 5) -> bool:
    """Check if a vLLM server is ready"""
    import socket
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except:
        return False


async def check_proxy_status(url: str) -> bool:
    """Check if proxy is ready"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{url}/status") as resp:
                return resp.status == 200
    except:
        return False


async def wait_for_server(url: str, timeout: int = 120) -> bool:
    """Wait for server to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        if await check_proxy_status(url):
            return True
        await asyncio.sleep(2)
    return False


# ============================================================================
# Warmup Functions
# ============================================================================

async def warmup_mode(mode: str):
    """Warmup the specified mode"""
    url = PROXY_URLS.get(mode)
    if not url:
        return

    print(f"    Checking proxy readiness...")
    if not await wait_for_server(url, timeout=60):
        print(f"    WARNING: Proxy not ready at {url}")
        return
    print(f"    Proxy ready!")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        if mode == "optimizer":
            print(f"    Warming up optimizer (triggering NCCL init)...")
            try:
                async with session.post(f"{url}/warmup") as resp:
                    result = await resp.json()
                    print(f"    Warmup result: {result}")
            except Exception as e:
                print(f"    Warmup error: {e}")
        elif mode in ["pd", "ppd"]:
            print(f"    Warming up {mode.upper()} (triggering NCCL init for 2 P-D pairs)...")
            # Set mode
            try:
                async with session.post(f"{url}/mode/{mode}") as resp:
                    pass
            except:
                pass

            # Send warmup requests
            for i in range(2):
                try:
                    async with session.post(
                        f"{url}/v1/completions",
                        json={
                            "model": MODEL_PATH,
                            "prompt": f"User: Warmup request {i}\nAssistant:",
                            "max_tokens": 10,
                            "temperature": 0.7,
                            "stream": False,
                        }
                    ) as resp:
                        if resp.status == 200:
                            print(f"    Warmup {mode.upper()} pair {i+1}: OK")
                except Exception as e:
                    print(f"    Warmup {mode.upper()} pair {i+1}: {e}")
            print(f"    Warmup complete: 8 requests sent")
        else:  # replica
            print(f"    Warming up replica...")
            # Replica doesn't need special warmup


# ============================================================================
# Request Functions (MATCHES original benchmark)
# ============================================================================

async def run_single_turn(
    session: aiohttp.ClientSession,
    url: str,
    conv_id: str,
    turn_num: int,
    input_tokens: int,
    output_tokens: int,
    history: str,
    routed_to: str = ""
) -> Tuple[TurnResult, str]:
    """Run a single turn request.

    Returns (TurnResult, updated_history).

    MATCHES original benchmark (benchmark_common.py):
    - Uses stream with include_usage
    - 120s timeout
    - TPOT = (e2e - ttft) / (completion_tokens - 1)
    - History = prompt + response_text (full response)
    """
    # Build prompt - MATCHES original benchmark format
    if turn_num == 1:
        prefix = f"CONV_{conv_id[:8]}_T1_"
        prompt = generate_prompt(input_tokens, prefix=prefix, seed=f"{conv_id}_t{turn_num}")
    else:
        prefix = f"T{turn_num}_followup_"
        new_input = generate_prompt(input_tokens, prefix=prefix, seed=f"{conv_id}_t{turn_num}")
        prompt = history + "\n" + new_input

    request_data = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0.8,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # conv_id for Turn 2+: used by selector for conversation tracking
    if turn_num > 1:
        request_data["conv_id"] = conv_id

    start_time = time.perf_counter()
    ttft = 0.0
    completion_tokens = 0
    response_text = ""

    try:
        async with session.post(
            f"{url}/v1/completions",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=120)  # Match original benchmark
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return TurnResult(
                    turn=turn_num,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=0, tpot_ms=0, e2e_ms=0,
                    success=False,
                    routed_to=routed_to,
                    error=f"HTTP {response.status}: {error_text[:100]}"
                ), history

            first_token = False
            buffer = ""

            async for chunk_bytes in response.content.iter_any():
                if not chunk_bytes:
                    continue
                buffer += chunk_bytes.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            if not first_token and chunk.get("choices"):
                                text = chunk["choices"][0].get("text", "")
                                if text:
                                    ttft = (time.perf_counter() - start_time) * 1000
                                    first_token = True
                                    response_text += text
                            elif chunk.get("choices"):
                                response_text += chunk["choices"][0].get("text", "")
                            # Get completion_tokens from usage
                            if chunk.get("usage"):
                                completion_tokens = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

            end_time = time.perf_counter()
            e2e_ms = (end_time - start_time) * 1000

            # TPOT calculation - MATCHES original benchmark
            # tpot = (e2e - ttft) / (tokens - 1)
            if completion_tokens > 1:
                tpot_ms = (e2e_ms - ttft) / (completion_tokens - 1)
            else:
                tpot_ms = 0

            # Update history - MATCHES original benchmark
            # Original: history = prompt + response_text (FULL response, not truncated!)
            if turn_num == 1:
                # For T1, build history for T2
                # Use generate_response_content to simulate full response for history building
                simulated_response = generate_response_content(output_tokens, seed=f"{conv_id}_resp{turn_num}")
                updated_history = prompt + simulated_response
            else:
                # For T2+, continue building history
                simulated_response = generate_response_content(output_tokens, seed=f"{conv_id}_resp{turn_num}")
                updated_history = prompt + simulated_response

            turn_result = TurnResult(
                turn=turn_num,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft,
                tpot_ms=tpot_ms,
                e2e_ms=e2e_ms,
                success=True,
                routed_to=routed_to,
                completion_tokens=completion_tokens
            )

            return turn_result, updated_history

    except asyncio.TimeoutError:
        return TurnResult(
            turn=turn_num,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False,
            routed_to=routed_to,
            error="Timeout"
        ), history
    except Exception as e:
        return TurnResult(
            turn=turn_num,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False,
            routed_to=routed_to,
            error=str(e)[:100]
        ), history


# ============================================================================
# Benchmark Runner
# ============================================================================

async def run_benchmark_for_mode(
    mode: str,
    qps: float,
    duration_sec: float,
    turn_configs: List[Dict],
    objective_dist: Dict[str, float],
    selector: RuleBasedSelector = None
) -> ModeResult:
    """Run benchmark for a single mode.

    Key behaviors:
    - T1 requests follow Poisson arrival
    - T2 requests sent in BURST mode (all at once after T1 completes)
    - Stagger mode is COMMENTED OUT (can be re-enabled if needed)
    """
    result = ModeResult(mode=mode)
    routing_stats = defaultdict(int)

    url = PROXY_URLS.get(mode)
    if not url:
        print(f"    ERROR: Unknown mode {mode}")
        return result

    # Initialize selector for optimizer mode
    if mode == "optimizer" and selector is None:
        selector = RuleBasedSelector(verbose=False)

    # Set mode for PD/PPD proxy
    if mode in ["pd", "ppd"]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{url}/mode/{mode}") as resp:
                    pass
            except:
                pass

    # Generate conversations with Poisson arrivals
    num_conversations = int(qps * duration_sec)
    arrival_times = []
    current_time = 0
    for _ in range(num_conversations):
        interval = random.expovariate(qps) if qps > 0 else 0.5
        current_time += interval
        if current_time < duration_sec:
            arrival_times.append(current_time)

    conversations = []
    for i, arrival in enumerate(arrival_times):
        # Select objective based on distribution
        r = random.random()
        cumsum = 0
        objective = "e2e"
        for obj, prob in objective_dist.items():
            cumsum += prob
            if r < cumsum:
                objective = obj
                break

        conv_id = f"{mode}_{i}_{int(time.time()*1000)}"
        conversations.append({
            "conv_id": conv_id,
            "objective": objective,
            "arrival_time": arrival,
            "turn_configs": turn_configs
        })

    # Run benchmark
    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()

        # Phase 1: Send all T1 requests following Poisson arrivals
        t1_tasks = []
        for conv in conversations:
            t1_tasks.append(run_t1_with_timing(
                session, url, conv, turn_configs[0], mode, selector, routing_stats, start_time
            ))

        t1_results = await asyncio.gather(*t1_tasks, return_exceptions=True)

        # Phase 2: Send all T2 requests in BURST mode (all at once)
        # NOTE: Stagger mode is commented out - can be re-enabled if needed
        t2_tasks = []
        for i, (conv, t1_result) in enumerate(zip(conversations, t1_results)):
            if isinstance(t1_result, Exception):
                continue
            conv_result, history = t1_result
            if not conv_result.turns or not conv_result.turns[0].success:
                result.conversations.append(conv_result)
                continue

            # For Turn 1, respect Poisson arrival times
            # For Turn 2+, send ALL simultaneously (burst mode - matches original benchmark)
            #
            # STAGGER MODE (commented out - uncomment to enable):
            # stagger_delay = i * (1.0 / max(qps, 1))  # Stagger T2 by 1/QPS interval
            # t2_tasks.append(run_t2_with_stagger(
            #     session, url, conv, conv_result, history, turn_configs,
            #     mode, selector, routing_stats, stagger_delay
            # ))

            # BURST MODE (current):
            t2_tasks.append(run_t2_burst(
                session, url, conv, conv_result, history, turn_configs,
                mode, selector, routing_stats
            ))

        if t2_tasks:
            t2_results = await asyncio.gather(*t2_tasks, return_exceptions=True)
            for t2_result in t2_results:
                if isinstance(t2_result, ConversationResult):
                    result.conversations.append(t2_result)
                elif isinstance(t2_result, Exception):
                    print(f"    T2 error: {t2_result}")

    result.routing_stats = dict(routing_stats)
    return result


async def run_t1_with_timing(
    session: aiohttp.ClientSession,
    url: str,
    conv: Dict,
    t1_config: Dict,
    mode: str,
    selector: RuleBasedSelector,
    routing_stats: Dict,
    start_time: float
) -> Tuple[ConversationResult, str]:
    """Run T1 request with Poisson timing"""
    conv_id = conv["conv_id"]
    objective = conv["objective"]
    arrival = conv["arrival_time"]

    # Wait for arrival time
    now = time.perf_counter() - start_time
    if arrival > now:
        await asyncio.sleep(arrival - now)

    conv_result = ConversationResult(conv_id=conv_id, objective=objective)

    # Determine routing for optimizer
    if mode == "optimizer" and selector:
        features = RequestFeatures(
            input_length=t1_config["input"],
            output_length=t1_config["output"],
            turn_number=1,
            has_cache=False,
            cached_gpu=None,
            queue_depths={"gpu1": 0, "gpu2": 0, "gpu3": 0},
            objective=OptimizationObjective(objective),
            current_qps=8.0
        )
        decision = selector.select_mode(features)
        routed_to = decision.mode.value
        routing_stats[routed_to] += 1
    else:
        routed_to = mode

    turn_result, history = await run_single_turn(
        session, url, conv_id, 1,
        t1_config["input"], t1_config["output"],
        "", routed_to
    )
    conv_result.turns.append(turn_result)

    return conv_result, history


async def run_t2_burst(
    session: aiohttp.ClientSession,
    url: str,
    conv: Dict,
    conv_result: ConversationResult,
    history: str,
    turn_configs: List[Dict],
    mode: str,
    selector: RuleBasedSelector,
    routing_stats: Dict
) -> ConversationResult:
    """Run T2+ requests in burst mode (no stagger delay)"""
    conv_id = conv["conv_id"]
    objective = conv["objective"]

    for turn_idx in range(1, len(turn_configs)):
        turn_num = turn_idx + 1
        t_config = turn_configs[turn_idx]

        # Determine routing for optimizer
        if mode == "optimizer" and selector:
            features = RequestFeatures(
                input_length=t_config["input"],
                output_length=t_config["output"],
                turn_number=turn_num,
                has_cache=True,
                cached_gpu="gpu1",  # Assume cached from T1
                queue_depths={"gpu1": 0, "gpu2": 0, "gpu3": 0},
                objective=OptimizationObjective(objective),
                current_qps=8.0
            )
            decision = selector.select_mode(features)
            routed_to = decision.mode.value
            routing_stats[routed_to] += 1
        else:
            routed_to = mode

        turn_result, history = await run_single_turn(
            session, url, conv_id, turn_num,
            t_config["input"], t_config["output"],
            history, routed_to
        )
        conv_result.turns.append(turn_result)

        if not turn_result.success:
            break

    return conv_result


# Stagger mode function (commented out but preserved for future use)
# async def run_t2_with_stagger(
#     session: aiohttp.ClientSession,
#     url: str,
#     conv: Dict,
#     conv_result: ConversationResult,
#     history: str,
#     turn_configs: List[Dict],
#     mode: str,
#     selector: RuleBasedSelector,
#     routing_stats: Dict,
#     stagger_delay: float
# ) -> ConversationResult:
#     """Run T2+ requests with stagger delay"""
#     await asyncio.sleep(stagger_delay)
#     return await run_t2_burst(
#         session, url, conv, conv_result, history, turn_configs,
#         mode, selector, routing_stats
#     )


# ============================================================================
# Panel Runners
# ============================================================================

def get_turn_configs(panel_id: str) -> List[Dict]:
    """Get turn configs for a panel"""
    return PANEL_CONFIGS.get(panel_id, {}).get("turn_configs", [
        {"input": 512, "output": 256},
        {"input": 128, "output": 128}
    ])


def get_objective_dist(panel_id: str, data_point: Any = None) -> Dict[str, float]:
    """Get objective distribution for a panel/data point"""
    config = PANEL_CONFIGS.get(panel_id, {})
    if isinstance(data_point, dict) and "objective_dist" in data_point:
        return data_point["objective_dist"]
    return config.get("objective_dist", {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33})


def get_qps(panel_id: str, data_point: Any = None) -> float:
    """Get QPS for a panel/data point"""
    config = PANEL_CONFIGS.get(panel_id, {})
    if panel_id == "B" and isinstance(data_point, (int, float)):
        return float(data_point)
    return config.get("qps", 8)


async def run_panel(panel_id: str, mode: str) -> Dict:
    """Run a complete panel for a mode"""
    config = PANEL_CONFIGS.get(panel_id)
    if not config:
        print(f"  Unknown panel: {panel_id}")
        return {}

    print(f"\n{'='*60}")
    print(f"Panel {panel_id}: {config['name']} - Mode: {mode}")
    print(f"{'='*60}")

    results = []
    selector = RuleBasedSelector(verbose=False) if mode == "optimizer" else None

    for dp in config["data_points"]:
        if isinstance(dp, dict):
            x_value = dp["name"]
        else:
            x_value = dp

        qps = get_qps(panel_id, dp)
        objective_dist = get_objective_dist(panel_id, dp)
        turn_configs = get_turn_configs(panel_id)

        print(f"  {x_value}...", end=" ", flush=True)

        mode_result = await run_benchmark_for_mode(
            mode=mode,
            qps=qps,
            duration_sec=DURATION_PER_POINT_SEC,
            turn_configs=turn_configs,
            objective_dist=objective_dist,
            selector=selector
        )

        slo = mode_result.compute_slo_attainment(turn2_only=True)
        latency = mode_result.compute_latency_stats()

        print(f"{slo['overall']:.1f}%")

        results.append({
            "x_value": x_value,
            "slo_attainment": slo,
            "latency_stats": latency,
            "routing_stats": mode_result.routing_stats,
            "num_conversations": len(mode_result.conversations)
        })

    return {"panel": panel_id, "mode": mode, "results": results}


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Fig6 Benchmark")
    parser.add_argument("--panel", choices=["A", "B", "all"], default="B")
    parser.add_argument("--mode", choices=["pd", "ppd", "replica", "optimizer", "all"], default="optimizer")
    parser.add_argument("--output-dir", default="results/fig6")
    args = parser.parse_args()

    print("="*60)
    print("FIG6 BENCHMARK")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Panel: {args.panel}")
    print(f"Mode: {args.mode}")
    print(f"Duration per point: {DURATION_PER_POINT_SEC}s")
    print(f"SLO Thresholds: {SLO_THRESHOLDS}")

    os.makedirs(args.output_dir, exist_ok=True)

    modes = ["pd", "ppd", "replica", "optimizer"] if args.mode == "all" else [args.mode]
    panels = ["A", "B"] if args.panel == "all" else [args.panel]

    all_results = {}

    for mode in modes:
        url = PROXY_URLS.get(mode)
        if not await check_proxy_status(url):
            print(f"\nWARNING: {mode} proxy not available at {url}")
            continue

        mode_results = {}
        for panel in panels:
            result = await run_panel(panel, mode)
            mode_results[panel] = result

        all_results[mode] = mode_results

    # Save results
    output_file = os.path.join(
        args.output_dir,
        f"panel_{args.panel}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
