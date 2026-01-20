#!/usr/bin/env python3
"""
Model Scaling Benchmark - G2 Supplementary Experiment

Tests how PPD performance scales with different model sizes.
Goal: Prove that PPD trends remain consistent across different model sizes.

Model Sizes:
- 8B: Llama-3.1-8B (current, ~16GB FP16)
- ~14B: e.g., Qwen2.5-14B (~28GB FP16)
- ~27B: e.g., Gemma-2-27B (~54GB FP16)

Configurations: 4R, 2P_2D, 2P_2pD
Workload: small_mid_bal
QPS: 4

Note: This script requires manual specification of model paths since
different models need different paths. The server start scripts need
to be modified to accept model path as parameter.
"""

import os
import sys

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
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

PROXY_URL = "http://localhost:10001"

# Model configurations
# Users should update these paths based on their available models
MODEL_CONFIGS = {
    "8B": {
        "name": "Llama-3.1-8B",
        "path": "/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.90,
    },
    "14B": {
        "name": "Qwen2.5-14B",
        "path": "/net/projects2/ds3lab/zongzel/qwen-2.5-14B",  # Update this path
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.92,
    },
    "27B": {
        "name": "Gemma-2-27B",
        "path": "/net/projects2/ds3lab/zongzel/gemma-2-27B",  # Update this path
        "max_model_len": 4096,  # May need to reduce for larger models
        "gpu_memory_utilization": 0.95,
    },
}

CONFIGS_TO_TEST = ["4R", "2P_2D", "2P_2pD", "1R_1P_1D_1pD"]

# Workload: balanced
WORKLOAD_CONFIG = {
    "name": "small_mid_bal",
    "t1_input": 128,
    "t1_output": 128,
    "t2_input": 128,
    "t2_output": 128,
}

DEFAULT_QPS = 4
NUM_TURNS = 2
DURATION_PER_POINT_SEC = 30
REQUEST_TIMEOUT_SEC = 180
WARMUP_REQUESTS = 5

# Reliability parameters (consistent with comprehensive_benchmark.py)
TEST_POINT_TIMEOUT_SEC = 600  # 10 minutes max per test point
HEALTH_CHECK_TIMEOUT_SEC = 10
SERVER_STARTUP_WAIT_SEC = 180  # Longer for larger models
SERVER_RESTART_WAIT_SEC = 180
MAX_CONSECUTIVE_FAILURES = 3

WORD_POOL = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "system", "data", "model", "process", "analysis", "result", "method",
    "research", "information", "development", "performance", "application",
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TurnResult:
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
class LatencyStats:
    count: int = 0
    avg_ttft_ms: float = 0
    avg_tpot_ms: float = 0
    avg_e2e_ms: float = 0
    p50_ttft_ms: float = 0
    p99_ttft_ms: float = 0

    @classmethod
    def from_turns(cls, turns: List[TurnResult]) -> "LatencyStats":
        successful = [t for t in turns if t.success]
        if not successful:
            return cls()

        ttfts = [t.ttft_ms for t in successful]
        tpots = [t.tpot_ms for t in successful if t.tpot_ms > 0]
        e2es = [t.e2e_ms for t in successful]

        return cls(
            count=len(successful),
            avg_ttft_ms=float(np.mean(ttfts)),
            avg_tpot_ms=float(np.mean(tpots)) if tpots else 0,
            avg_e2e_ms=float(np.mean(e2es)),
            p50_ttft_ms=float(np.percentile(ttfts, 50)),
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
        )


@dataclass
class ModelScalingResult:
    config: str
    model_name: str
    model_size: str
    model_path: str
    workload: str
    qps: float
    turn1: Dict
    turn2: Dict
    average: Dict
    throughput: Dict[str, float]
    total_requests: int
    successful_requests: int
    success_rate: float
    duration_sec: float
    timestamp: str
    error: Optional[str] = None


# ============================================================================
# Prompt Generation
# ============================================================================

def generate_prompt(num_tokens: int, prefix: str = "", seed: str = "") -> str:
    rng = random.Random(seed) if seed else random.Random()
    target_words = int(num_tokens * 0.75)

    topics = ["coding", "machine learning", "data science", "web development"]
    topic = rng.choice(topics)
    intro = f"Explain the concept of {topic} in detail."

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
    cleanup_script = Path(PROJECT_DIR) / "scripts" / "server" / "cleanup_all.sh"
    if cleanup_script.exists():
        try:
            result = subprocess.run(
                ["bash", str(cleanup_script)],
                capture_output=True, text=True, timeout=60
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    return False


def start_config_with_model(config: str, model_config: Dict) -> bool:
    """Start a configuration with a specific model.

    This function modifies the environment to use a different model.
    For production use, you may want to create model-specific start scripts.
    """
    # For now, we use the standard start script but set MODEL_PATH env var
    script = Path(PROJECT_DIR) / "scripts" / "server" / f"start_{config}.sh"
    if not script.exists():
        print(f"  ERROR: Script not found: {script}")
        return False

    env = os.environ.copy()
    env["MODEL_PATH"] = model_config["path"]
    env["MAX_MODEL_LEN"] = str(model_config["max_model_len"])
    env["GPU_MEMORY_UTILIZATION"] = str(model_config["gpu_memory_utilization"])
    env["MAX_WAIT"] = "1200"

    try:
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )

        start_time = time.time()
        while time.time() - start_time < 1500:  # 25 min for larger models (must be > MAX_WAIT)
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
    except Exception as e:
        return False, f"Health check failed: {str(e)[:50]}"


async def warmup_servers(model_path: str) -> bool:
    print("  Warming up servers...")
    success_count = 0

    try:
        timeout = aiohttp.ClientTimeout(total=120)  # Longer for larger models
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i in range(WARMUP_REQUESTS):
                try:
                    async with session.post(
                        f"{PROXY_URL}/v1/completions",
                        json={
                            "model": model_path,
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


async def restart_server(config: str, model_config: Dict) -> bool:
    """Restart server after failure (consistent with comprehensive_benchmark.py)."""
    print(f"\n  Attempting server restart for {config} with {model_config['name']}...")

    # Cleanup
    run_cleanup()
    await asyncio.sleep(5)

    # Start fresh
    if not start_config_with_model(config, model_config):
        print("  ERROR: Server restart failed during startup")
        return False

    # Wait for full startup (longer for larger models)
    print(f"  Waiting {SERVER_RESTART_WAIT_SEC}s for server to stabilize...")
    await asyncio.sleep(SERVER_RESTART_WAIT_SEC)

    # Health check
    healthy, error = await check_server_health()
    if not healthy:
        print(f"  ERROR: Server unhealthy after restart: {error}")
        return False

    # Warmup
    if not await warmup_servers(model_config["path"]):
        print("  WARNING: Warmup failed after restart, but server is healthy")

    print("  Server restart complete")
    return True


# ============================================================================
# Request Functions
# ============================================================================

async def run_single_turn(
    session: aiohttp.ClientSession,
    model_path: str,
    conv_id: str,
    turn_num: int,
    input_tokens: int,
    output_tokens: int,
    history: str,
) -> Tuple[TurnResult, str]:
    if turn_num == 1:
        prefix = f"CONV_{conv_id[:8]}_T1_"
        prompt = generate_prompt(input_tokens, prefix=prefix, seed=f"{conv_id}_t{turn_num}")
    else:
        prefix = f"T{turn_num}_followup_"
        new_input = generate_prompt(input_tokens, prefix=prefix, seed=f"{conv_id}_t{turn_num}")
        prompt = history + "\n" + new_input

    request_data = {
        "model": model_path,
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
                    error=f"HTTP {response.status}"
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


async def run_benchmark(
    config: str,
    model_config: Dict,
    model_size: str,
    qps: float,
    duration_sec: float,
) -> ModelScalingResult:
    print(f"\n  Running: {model_config['name']} @ QPS={qps} for {duration_sec}s")

    workload = WORKLOAD_CONFIG
    model_path = model_config["path"]
    t1_results: List[TurnResult] = []
    t2_results: List[TurnResult] = []
    total_requests = 0
    successful_requests = 0

    start_time = time.time()
    connector = aiohttp.TCPConnector(limit=100, force_close=True)
    session = aiohttp.ClientSession(connector=connector)

    try:
        interval = 1.0 / qps if qps > 0 else 1.0
        req_idx = 0

        while time.time() - start_time < duration_sec:
            req_start = time.time()
            conv_id = f"model_scaling_{req_idx}_{int(time.time())}"
            history = ""

            # Turn 1
            result, history = await run_single_turn(
                session, model_path, conv_id, 1,
                workload["t1_input"], workload["t1_output"], history
            )
            t1_results.append(result)
            total_requests += 1
            if result.success:
                successful_requests += 1

            # Turn 2
            if result.success:
                result, history = await run_single_turn(
                    session, model_path, conv_id, 2,
                    workload["t2_input"], workload["t2_output"], history
                )
                t2_results.append(result)
                total_requests += 1
                if result.success:
                    successful_requests += 1

            req_idx += 1

            elapsed = time.time() - req_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

    finally:
        await session.close()

    actual_duration = time.time() - start_time

    t1_stats = LatencyStats.from_turns(t1_results)
    t2_stats = LatencyStats.from_turns(t2_results)

    all_results = t1_results + t2_results
    all_successful = [r for r in all_results if r.success]
    avg_stats = LatencyStats.from_turns(all_results)

    total_tokens = sum(r.completion_tokens for r in all_successful)
    throughput = {
        "requests_per_sec": len(all_successful) / actual_duration if actual_duration > 0 else 0,
        "tokens_per_sec": total_tokens / actual_duration if actual_duration > 0 else 0,
    }

    success_rate = successful_requests / total_requests * 100 if total_requests > 0 else 0

    print(f"    Success: {successful_requests}/{total_requests} ({success_rate:.1f}%)")
    print(f"    T1 TTFT: {t1_stats.avg_ttft_ms:.1f}ms, T2 TTFT: {t2_stats.avg_ttft_ms:.1f}ms")

    return ModelScalingResult(
        config=config,
        model_name=model_config["name"],
        model_size=model_size,
        model_path=model_path,
        workload=workload["name"],
        qps=qps,
        turn1=asdict(t1_stats),
        turn2=asdict(t2_stats),
        average=asdict(avg_stats),
        throughput=throughput,
        total_requests=total_requests,
        successful_requests=successful_requests,
        success_rate=success_rate,
        duration_sec=actual_duration,
        timestamp=datetime.now().isoformat(),
    )


# ============================================================================
# Timeout-Protected Benchmark Runner
# ============================================================================

async def run_benchmark_with_timeout(
    config: str,
    model_config: Dict,
    model_size: str,
    qps: float,
    duration_sec: float,
) -> Tuple[ModelScalingResult, bool]:
    """
    Run benchmark with timeout protection (consistent with comprehensive_benchmark.py).

    Returns:
        Tuple of (result, server_healthy)
    """
    try:
        # Wrap in timeout
        result = await asyncio.wait_for(
            run_benchmark(config, model_config, model_size, qps, duration_sec),
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
        error_result = ModelScalingResult(
            config=config,
            model_name=model_config["name"],
            model_size=model_size,
            model_path=model_config["path"],
            workload=WORKLOAD_CONFIG["name"],
            qps=qps,
            turn1={},
            turn2={},
            average={},
            throughput={"requests_per_sec": 0, "tokens_per_sec": 0},
            total_requests=0,
            successful_requests=0,
            success_rate=0,
            duration_sec=TEST_POINT_TIMEOUT_SEC,
            timestamp=datetime.now().isoformat(),
            error="TestPointTimeout",
        )

        return error_result, False

    except Exception as e:
        print(f"\n    ERROR: Unexpected error during benchmark: {e}")

        error_result = ModelScalingResult(
            config=config,
            model_name=model_config["name"],
            model_size=model_size,
            model_path=model_config["path"],
            workload=WORKLOAD_CONFIG["name"],
            qps=qps,
            turn1={},
            turn2={},
            average={},
            throughput={"requests_per_sec": 0, "tokens_per_sec": 0},
            total_requests=0,
            successful_requests=0,
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
    results_dir = Path(PROJECT_DIR) / "results" / "model_scaling"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_size in args.models:
        if model_size not in MODEL_CONFIGS:
            print(f"  ERROR: Unknown model size: {model_size}")
            continue

        model_config = MODEL_CONFIGS[model_size]
        print(f"\n{'='*70}")
        print(f"Model: {model_config['name']} ({model_size})")
        print(f"Path: {model_config['path']}")
        print(f"{'='*70}")

        # Check if model path exists
        if not Path(model_config["path"]).exists():
            print(f"  WARNING: Model path not found: {model_config['path']}")
            print("  Please update MODEL_CONFIGS with correct paths")
            continue

        for config in args.configs:

            print(f"\n  Configuration: {config}")

            consecutive_failures = 0
            server_restarts = 0

            if not args.skip_startup:
                print(f"  Starting {config} with {model_config['name']}...")
                run_cleanup()
                time.sleep(5)

                if not start_config_with_model(config, model_config):
                    print(f"  ERROR: Failed to start {config}")
                    continue

                # Dynamic wait based on model size (startup script already waits for ports)
                # Just a short stabilization period after ports are ready
                stabilize_sec = {"8B": 10, "14B": 30, "27B": 60}.get(model_size, 30)
                print(f"  Waiting {stabilize_sec}s for model to stabilize...")
                time.sleep(stabilize_sec)

            # Health check
            healthy, error = await check_server_health()
            if not healthy:
                print(f"  ERROR: Server not healthy: {error}")
                if not args.skip_startup:
                    run_cleanup()
                continue

            # Warmup
            if not await warmup_servers(model_config["path"]):
                print("  WARNING: Warmup failed, continuing...")

            # Run benchmark with retry logic
            retry_count = 0
            max_retries = 2

            while retry_count <= max_retries:
                result, server_healthy = await run_benchmark_with_timeout(
                    config=config,
                    model_config=model_config,
                    model_size=model_size,
                    qps=args.qps,
                    duration_sec=args.duration,
                )

                # Check for failure
                if result.error or not server_healthy:
                    consecutive_failures += 1
                    retry_count += 1
                    print(f"\n    Failure count: {consecutive_failures}, Retry: {retry_count}/{max_retries}")

                    if retry_count <= max_retries and not args.skip_startup:
                        print(f"\n  Attempting server restart...")
                        server_restarts += 1

                        if await restart_server(config, model_config):
                            print("  Server restart successful, retrying...")
                            consecutive_failures = 0
                            continue  # Retry
                        else:
                            print("  Server restart failed, recording error result")
                            break
                    else:
                        print("  Max retries reached or skip-startup mode, recording error result")
                        break
                else:
                    consecutive_failures = 0
                    break  # Success

            all_results.append(asdict(result))

            # Save incrementally
            output_file = results_dir / f"{model_size}_{config}.json"
            with open(output_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"    Saved: {output_file}")

            if not args.skip_startup:
                print(f"  Stopping {config}...")
                print(f"  Server restarts for this test: {server_restarts}")
                run_cleanup()
                time.sleep(10)

    # Save combined results
    combined_file = results_dir / "all_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"All results saved to: {combined_file}")
    print(f"Total test points: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description="Model Scaling Benchmark")
    parser.add_argument("--models", nargs="+", default=["8B"],
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model sizes to test")
    parser.add_argument("--configs", nargs="+", default=CONFIGS_TO_TEST,
                       help="Configurations to test")
    parser.add_argument("--qps", type=float, default=DEFAULT_QPS,
                       help="Requests per second")
    parser.add_argument("--duration", type=float, default=DURATION_PER_POINT_SEC,
                       help="Duration per test point (seconds)")
    parser.add_argument("--skip-startup", action="store_true",
                       help="Skip server startup")
    parser.add_argument("--list-models", action="store_true",
                       help="List available model configurations")

    args = parser.parse_args()

    if args.list_models:
        print("Available model configurations:")
        for size, config in MODEL_CONFIGS.items():
            exists = Path(config["path"]).exists()
            status = "OK" if exists else "NOT FOUND"
            print(f"  {size}: {config['name']}")
            print(f"      Path: {config['path']} [{status}]")
            print(f"      Max len: {config['max_model_len']}, GPU util: {config['gpu_memory_utilization']}")
        return

    print("="*70)
    print("MODEL SCALING BENCHMARK (G2)")
    print("="*70)
    print(f"Models: {args.models}")
    print(f"Configurations: {args.configs}")
    print(f"QPS: {args.qps}")
    print(f"Duration: {args.duration}s per point")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
