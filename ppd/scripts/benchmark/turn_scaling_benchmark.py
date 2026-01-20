#!/usr/bin/env python3
"""
Turn Scaling Benchmark - G1 Supplementary Experiment

Tests how PPD performance scales with number of conversation turns.
Goal: Prove that PPD trends remain consistent across multi-turn scenarios.

Turn Numbers: 2 (baseline), 4, 8, 16
Configurations: 4R, 2P_2D, 2P_2pD, 1R_1P_2pD
Workload: small_mid_bal (representative balanced workload)
QPS: 4 (medium load)
"""

import os
import sys

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

import json
import time
import random
import string
import asyncio
import aiohttp
import argparse
import subprocess
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from src.config import MODEL_PATH

# ============================================================================
# Configuration
# ============================================================================

PROXY_URL = "http://localhost:10001"

# Experiment configurations
TURN_NUMBERS = [2, 4, 8, 16]

CONFIGS_TO_TEST = ["4R", "2P_2D", "2P_2pD", "1R_1P_2pD"]

# Workload: balanced (T1: 128->128, T2: 128->128)
WORKLOAD_CONFIG = {
    "name": "small_mid_bal",
    "t1_input": 128,
    "t1_output": 128,
    "t2_input": 128,
    "t2_output": 128,
}

DEFAULT_QPS = 4
DURATION_PER_POINT_SEC = 30  # Longer duration for multi-turn
REQUEST_TIMEOUT_SEC = 180    # Longer timeout for multi-turn
WARMUP_REQUESTS = 5

# Reliability parameters (consistent with comprehensive_benchmark.py)
TEST_POINT_TIMEOUT_SEC = 600  # 10 minutes max per test point
HEALTH_CHECK_TIMEOUT_SEC = 10
SERVER_STARTUP_WAIT_SEC = 120
SERVER_RESTART_WAIT_SEC = 120
MAX_RETRIES = 2
MAX_CONSECUTIVE_FAILURES = 3

# Word pool for prompt generation
WORD_POOL = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "system", "data", "model", "process", "analysis", "result", "method",
    "research", "information", "development", "performance", "application",
    "technology", "network", "algorithm", "function", "structure", "interface",
    "create", "update", "delete", "execute", "configure", "optimize",
    "implement", "design", "build", "deploy", "monitor", "validate",
]


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
    completion_tokens: int = 0
    error: Optional[str] = None


@dataclass
class PerTurnStats:
    """Statistics for a single turn number"""
    turn: int
    count: int = 0
    avg_ttft_ms: float = 0
    avg_tpot_ms: float = 0
    avg_e2e_ms: float = 0
    p50_ttft_ms: float = 0
    p99_ttft_ms: float = 0
    p50_tpot_ms: float = 0
    p99_tpot_ms: float = 0

    @classmethod
    def from_turns(cls, turn_num: int, turns: List[TurnResult]) -> "PerTurnStats":
        successful = [t for t in turns if t.success and t.turn == turn_num]
        if not successful:
            return cls(turn=turn_num)

        ttfts = [t.ttft_ms for t in successful]
        tpots = [t.tpot_ms for t in successful if t.tpot_ms > 0]
        e2es = [t.e2e_ms for t in successful]

        return cls(
            turn=turn_num,
            count=len(successful),
            avg_ttft_ms=float(np.mean(ttfts)),
            avg_tpot_ms=float(np.mean(tpots)) if tpots else 0,
            avg_e2e_ms=float(np.mean(e2es)),
            p50_ttft_ms=float(np.percentile(ttfts, 50)),
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
            p50_tpot_ms=float(np.percentile(tpots, 50)) if tpots else 0,
            p99_tpot_ms=float(np.percentile(tpots, 99)) if tpots else 0,
        )


@dataclass
class TurnScalingResult:
    """Result of a turn scaling benchmark"""
    config: str
    num_turns: int
    workload: str
    qps: float
    per_turn_metrics: List[Dict]  # List of PerTurnStats as dicts
    summary: Dict[str, float]     # Averaged metrics across all turns
    t1_metrics: Dict[str, float]  # Turn 1 specific metrics
    t2plus_metrics: Dict[str, float]  # Turn 2+ averaged metrics
    total_conversations: int
    successful_conversations: int
    success_rate: float
    duration_sec: float
    timestamp: str
    error: Optional[str] = None


# ============================================================================
# Prompt Generation
# ============================================================================

def generate_prompt(num_tokens: int, prefix: str = "", seed: str = "") -> str:
    """Generate a realistic prompt with approximately num_tokens tokens."""
    rng = random.Random(seed) if seed else random.Random()
    target_words = int(num_tokens * 0.75)

    topics = [
        "coding", "machine learning", "data science", "web development",
        "cloud computing", "database design", "API development", "testing",
    ]

    topic = rng.choice(topics)
    templates = [
        f"Explain the concept of {topic} in detail.",
        f"What are the best practices for {topic}?",
        f"Can you provide a guide to {topic}?",
        f"Help me understand {topic} better.",
    ]

    intro = rng.choice(templates)
    words = []
    remaining_words = target_words - len(intro.split())

    for _ in range(remaining_words):
        if rng.random() < 0.5:
            words.append(rng.choice(WORD_POOL))
        else:
            word_len = rng.randint(3, 8)
            words.append(''.join(rng.choices(string.ascii_lowercase, k=word_len)))

    content_parts = [intro]
    chunk_size = rng.randint(8, 12)
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk:
            content_parts.append(chunk + ".")

    content = ' '.join(content_parts)
    return f"User: {prefix}{content}\nAssistant:"


# ============================================================================
# Server Management
# ============================================================================

def run_cleanup() -> bool:
    """Run comprehensive cleanup"""
    cleanup_script = Path(PROJECT_DIR) / "scripts" / "server" / "cleanup_all.sh"
    if cleanup_script.exists():
        try:
            result = subprocess.run(
                ["bash", str(cleanup_script)],
                capture_output=True, text=True, timeout=60
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("  WARNING: Cleanup timed out")
            return False
    return False


def start_config(config: str) -> bool:
    """Start a specific configuration"""
    script = Path(PROJECT_DIR) / "scripts" / "server" / f"start_{config}.sh"
    if not script.exists():
        print(f"  ERROR: Script not found: {script}")
        return False

    try:
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        start_time = time.time()
        while time.time() - start_time < 600:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"    {line.rstrip()}")

        if proc.poll() is None:
            proc.kill()
            print("  ERROR: Server startup timed out")
            return False

        return proc.wait() == 0
    except Exception as e:
        print(f"  ERROR: Failed to start server: {e}")
        return False


async def check_server_health() -> Tuple[bool, Optional[str]]:
    """Check if proxy and backend servers are healthy."""
    try:
        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_SEC)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{PROXY_URL}/status") as resp:
                if resp.status != 200:
                    return False, f"Proxy returned status {resp.status}"
                data = await resp.json()
                instances = data.get("instances", {})
                total_instances = sum(len(v) for v in instances.values())
                if total_instances == 0:
                    return False, "No backend instances registered"
                return True, None
    except asyncio.TimeoutError:
        return False, "Health check timed out"
    except Exception as e:
        return False, f"Health check failed: {str(e)[:50]}"


async def warmup_servers() -> bool:
    """Send warmup requests."""
    print("  Warming up servers...")
    success_count = 0

    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i in range(WARMUP_REQUESTS):
                try:
                    async with session.post(
                        f"{PROXY_URL}/v1/completions",
                        json={
                            "model": MODEL_PATH,
                            "prompt": f"User: Warmup {i}\nAssistant:",
                            "max_tokens": 10,
                            "temperature": 0.7,
                        }
                    ) as resp:
                        await resp.read()
                        if resp.status == 200:
                            success_count += 1
                except Exception as e:
                    print(f"    Warmup {i}: {e}")

        if success_count >= WARMUP_REQUESTS // 2:
            print(f"  Warmup complete ({success_count}/{WARMUP_REQUESTS})")
            return True
        else:
            print(f"  Warmup failed ({success_count}/{WARMUP_REQUESTS})")
            return False
    except Exception as e:
        print(f"  Warmup error: {e}")
        return False


async def restart_server(config: str) -> bool:
    """Restart server after failure (consistent with comprehensive_benchmark.py)."""
    print(f"\n  Attempting server restart for {config}...")

    # Cleanup
    run_cleanup()
    await asyncio.sleep(5)

    # Start fresh
    if not start_config(config):
        print("  ERROR: Server restart failed during startup")
        return False

    # Wait for full startup
    print(f"  Waiting {SERVER_RESTART_WAIT_SEC}s for server to stabilize...")
    await asyncio.sleep(SERVER_RESTART_WAIT_SEC)

    # Health check
    healthy, error = await check_server_health()
    if not healthy:
        print(f"  ERROR: Server unhealthy after restart: {error}")
        return False

    # Warmup
    if not await warmup_servers():
        print("  WARNING: Warmup failed after restart, but server is healthy")

    print("  Server restart complete")
    return True


# ============================================================================
# Request Functions
# ============================================================================

async def run_single_turn(
    session: aiohttp.ClientSession,
    conv_id: str,
    turn_num: int,
    input_tokens: int,
    output_tokens: int,
    history: str,
) -> Tuple[TurnResult, str]:
    """Run a single turn request."""
    # Build prompt
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

    if turn_num > 1:
        request_data["conv_id"] = conv_id

    start_time = time.perf_counter()
    ttft = 0.0
    completion_tokens = 0
    response_text = ""

    try:
        async with session.post(
            f"{PROXY_URL}/v1/completions",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return TurnResult(
                    turn=turn_num,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=0, tpot_ms=0, e2e_ms=0,
                    success=False,
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
                            if chunk.get("usage"):
                                completion_tokens = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

            end_time = time.perf_counter()
            e2e_ms = (end_time - start_time) * 1000

            if completion_tokens > 1:
                tpot_ms = (e2e_ms - ttft) / (completion_tokens - 1)
            else:
                tpot_ms = 0

            updated_history = prompt + response_text

            return TurnResult(
                turn=turn_num,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft,
                tpot_ms=tpot_ms,
                e2e_ms=e2e_ms,
                success=True,
                completion_tokens=completion_tokens
            ), updated_history

    except asyncio.TimeoutError:
        return TurnResult(
            turn=turn_num,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False,
            error="Timeout"
        ), history
    except Exception as e:
        return TurnResult(
            turn=turn_num,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            success=False,
            error=str(e)[:100]
        ), history


async def run_multi_turn_conversation(
    session: aiohttp.ClientSession,
    conv_idx: int,
    num_turns: int,
    workload: Dict,
) -> List[TurnResult]:
    """Run a complete multi-turn conversation."""
    conv_id = f"turn_scaling_{conv_idx}_{num_turns}t_{int(time.time())}"
    history = ""
    results = []

    for turn_num in range(1, num_turns + 1):
        if turn_num == 1:
            input_tokens = workload["t1_input"]
            output_tokens = workload["t1_output"]
        else:
            input_tokens = workload["t2_input"]
            output_tokens = workload["t2_output"]

        result, history = await run_single_turn(
            session, conv_id, turn_num, input_tokens, output_tokens, history
        )
        results.append(result)

        if not result.success:
            # Stop conversation on failure
            break

    return results


async def run_benchmark(
    config: str,
    num_turns: int,
    qps: float,
    duration_sec: float,
) -> TurnScalingResult:
    """Run a complete turn scaling benchmark."""
    print(f"\n  Running: {num_turns} turns @ QPS={qps} for {duration_sec}s")

    workload = WORKLOAD_CONFIG
    all_turn_results: List[TurnResult] = []
    total_conversations = 0
    successful_conversations = 0

    start_time = time.time()
    connector = aiohttp.TCPConnector(limit=100, force_close=True)
    session = aiohttp.ClientSession(connector=connector)

    try:
        interval = 1.0 / qps if qps > 0 else 1.0
        conv_idx = 0

        while time.time() - start_time < duration_sec:
            conv_start = time.time()

            results = await run_multi_turn_conversation(
                session, conv_idx, num_turns, workload
            )
            all_turn_results.extend(results)
            total_conversations += 1

            if all(r.success for r in results):
                successful_conversations += 1

            conv_idx += 1

            # Maintain QPS
            elapsed = time.time() - conv_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

    finally:
        await session.close()

    actual_duration = time.time() - start_time

    # Compute per-turn statistics
    per_turn_metrics = []
    for turn_num in range(1, num_turns + 1):
        stats = PerTurnStats.from_turns(turn_num, all_turn_results)
        per_turn_metrics.append(asdict(stats))

    # Compute summary (averaged across all turns)
    successful_results = [r for r in all_turn_results if r.success]
    if successful_results:
        summary = {
            "avg_ttft_ms": float(np.mean([r.ttft_ms for r in successful_results])),
            "avg_tpot_ms": float(np.mean([r.tpot_ms for r in successful_results if r.tpot_ms > 0])),
            "avg_e2e_ms": float(np.mean([r.e2e_ms for r in successful_results])),
        }
    else:
        summary = {"avg_ttft_ms": 0, "avg_tpot_ms": 0, "avg_e2e_ms": 0}

    # T1 metrics
    t1_results = [r for r in all_turn_results if r.success and r.turn == 1]
    if t1_results:
        t1_metrics = {
            "avg_ttft_ms": float(np.mean([r.ttft_ms for r in t1_results])),
            "avg_tpot_ms": float(np.mean([r.tpot_ms for r in t1_results if r.tpot_ms > 0])),
            "avg_e2e_ms": float(np.mean([r.e2e_ms for r in t1_results])),
        }
    else:
        t1_metrics = {"avg_ttft_ms": 0, "avg_tpot_ms": 0, "avg_e2e_ms": 0}

    # T2+ metrics
    t2plus_results = [r for r in all_turn_results if r.success and r.turn > 1]
    if t2plus_results:
        t2plus_metrics = {
            "avg_ttft_ms": float(np.mean([r.ttft_ms for r in t2plus_results])),
            "avg_tpot_ms": float(np.mean([r.tpot_ms for r in t2plus_results if r.tpot_ms > 0])),
            "avg_e2e_ms": float(np.mean([r.e2e_ms for r in t2plus_results])),
        }
    else:
        t2plus_metrics = {"avg_ttft_ms": 0, "avg_tpot_ms": 0, "avg_e2e_ms": 0}

    success_rate = successful_conversations / total_conversations * 100 if total_conversations > 0 else 0

    print(f"    Conversations: {successful_conversations}/{total_conversations} ({success_rate:.1f}%)")
    print(f"    T1 TTFT: {t1_metrics['avg_ttft_ms']:.1f}ms, T2+ TTFT: {t2plus_metrics['avg_ttft_ms']:.1f}ms")

    return TurnScalingResult(
        config=config,
        num_turns=num_turns,
        workload=workload["name"],
        qps=qps,
        per_turn_metrics=per_turn_metrics,
        summary=summary,
        t1_metrics=t1_metrics,
        t2plus_metrics=t2plus_metrics,
        total_conversations=total_conversations,
        successful_conversations=successful_conversations,
        success_rate=success_rate,
        duration_sec=actual_duration,
        timestamp=datetime.now().isoformat(),
    )


# ============================================================================
# Timeout-Protected Benchmark Runner
# ============================================================================

async def run_benchmark_with_timeout(
    config: str,
    num_turns: int,
    qps: float,
    duration_sec: float,
) -> Tuple[TurnScalingResult, bool]:
    """
    Run benchmark with timeout protection (consistent with comprehensive_benchmark.py).

    Returns:
        Tuple of (result, server_healthy)
    """
    try:
        # Wrap in timeout
        result = await asyncio.wait_for(
            run_benchmark(config, num_turns, qps, duration_sec),
            timeout=TEST_POINT_TIMEOUT_SEC
        )

        # Check server health after test
        healthy, error = await check_server_health()
        if not healthy:
            print(f"\n    WARNING: Server unhealthy after test: {error}")

        return result, healthy

    except asyncio.TimeoutError:
        # Test point timed out - likely server is stuck
        print(f"\n    TIMEOUT: Test point exceeded {TEST_POINT_TIMEOUT_SEC}s limit")

        # Create error result
        error_result = TurnScalingResult(
            config=config,
            num_turns=num_turns,
            workload=WORKLOAD_CONFIG["name"],
            qps=qps,
            per_turn_metrics=[],
            summary={"avg_ttft_ms": 0, "avg_tpot_ms": 0, "avg_e2e_ms": 0},
            t1_metrics={"avg_ttft_ms": 0, "avg_tpot_ms": 0, "avg_e2e_ms": 0},
            t2plus_metrics={"avg_ttft_ms": 0, "avg_tpot_ms": 0, "avg_e2e_ms": 0},
            total_conversations=0,
            successful_conversations=0,
            success_rate=0,
            duration_sec=TEST_POINT_TIMEOUT_SEC,
            timestamp=datetime.now().isoformat(),
            error="TestPointTimeout",
        )

        return error_result, False

    except Exception as e:
        print(f"\n    ERROR: Unexpected error during benchmark: {e}")

        error_result = TurnScalingResult(
            config=config,
            num_turns=num_turns,
            workload=WORKLOAD_CONFIG["name"],
            qps=qps,
            per_turn_metrics=[],
            summary={"avg_ttft_ms": 0, "avg_tpot_ms": 0, "avg_e2e_ms": 0},
            t1_metrics={"avg_ttft_ms": 0, "avg_tpot_ms": 0, "avg_e2e_ms": 0},
            t2plus_metrics={"avg_ttft_ms": 0, "avg_tpot_ms": 0, "avg_e2e_ms": 0},
            total_conversations=0,
            successful_conversations=0,
            success_rate=0,
            duration_sec=0,
            timestamp=datetime.now().isoformat(),
            error=str(e)[:100],
        )

        return error_result, False


# ============================================================================
# Main
# ============================================================================

async def main_async(args):
    """Async main function with OOM resilience (consistent with comprehensive_benchmark.py)."""
    results_dir = Path(PROJECT_DIR) / "results" / "turn_scaling"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for config in args.configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {config}")
        print(f"{'='*70}")

        consecutive_failures = 0
        server_restarts = 0

        if not args.skip_startup:
            # Start server
            print(f"\n  Starting {config}...")
            run_cleanup()
            time.sleep(5)

            if not start_config(config):
                print(f"  ERROR: Failed to start {config}")
                continue

            time.sleep(10)

        # Health check
        healthy, error = await check_server_health()
        if not healthy:
            print(f"  ERROR: Server not healthy: {error}")
            if not args.skip_startup:
                run_cleanup()
            continue

        # Warmup
        if not await warmup_servers():
            print("  WARNING: Warmup failed, continuing anyway...")

        # Run benchmarks for each turn number with failure handling
        turn_idx = 0
        turns_to_test = list(args.turns)

        while turn_idx < len(turns_to_test):
            num_turns = turns_to_test[turn_idx]

            result, server_healthy = await run_benchmark_with_timeout(
                config=config,
                num_turns=num_turns,
                qps=args.qps,
                duration_sec=args.duration,
            )

            # Check for failure
            if result.error or not server_healthy:
                consecutive_failures += 1
                print(f"\n    Failure count: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}")

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\n  Too many consecutive failures, restarting server...")
                    server_restarts += 1

                    if not args.skip_startup and await restart_server(config):
                        print("  Server restart successful, retrying...")
                        consecutive_failures = 0
                        continue  # Retry same turn
                    else:
                        print("  Server restart failed, skipping remaining turns for this config")
                        break
            else:
                consecutive_failures = 0

            all_results.append(asdict(result))

            # Save incrementally
            output_file = results_dir / f"{config}_{num_turns}turns.json"
            with open(output_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"    Saved: {output_file}")

            # Check server health before next test point
            if turn_idx < len(turns_to_test) - 1:
                healthy, _ = await check_server_health()
                if not healthy and not args.skip_startup:
                    print("\n  [Health] Server unhealthy, attempting restart...")
                    server_restarts += 1
                    if await restart_server(config):
                        print("  [Health] Server restart successful")
                        consecutive_failures = 0
                    else:
                        print("  [Health] Server restart failed, stopping config")
                        break

            turn_idx += 1

        if not args.skip_startup:
            print(f"\n  Stopping {config}...")
            print(f"  Server restarts for this config: {server_restarts}")
            run_cleanup()
            time.sleep(5)

    # Save combined results
    combined_file = results_dir / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"All results saved to: {combined_file}")
    print(f"Total test points: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description="Turn Scaling Benchmark")
    parser.add_argument("--configs", nargs="+", default=CONFIGS_TO_TEST,
                       help="Configurations to test")
    parser.add_argument("--turns", nargs="+", type=int, default=TURN_NUMBERS,
                       help="Turn numbers to test")
    parser.add_argument("--qps", type=float, default=DEFAULT_QPS,
                       help="Requests per second")
    parser.add_argument("--duration", type=float, default=DURATION_PER_POINT_SEC,
                       help="Duration per test point (seconds)")
    parser.add_argument("--skip-startup", action="store_true",
                       help="Skip server startup (assume already running)")

    args = parser.parse_args()

    print("="*70)
    print("TURN SCALING BENCHMARK (G1)")
    print("="*70)
    print(f"Configurations: {args.configs}")
    print(f"Turn numbers: {args.turns}")
    print(f"QPS: {args.qps}")
    print(f"Duration: {args.duration}s per point")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
