#!/usr/bin/env python3
"""
Interference Benchmark - Prefill-Decode干扰测试

用于论文引入部分的micro-benchmark，证明：
1. Full-prefill与decode混合时干扰严重
2. Append-prefill与decode混合时干扰很小
3. 因此PPD模式中pD上的append-prefill+decode共存是可行的

实验设计（参考DistServe Figure 2）：
- Line 1: decode-only baseline
- Line 2: decode + one full-prefill (input=1024, context=0)
- Line 3: decode + one append-prefill (input=128, context=2048)

X轴: Batch size (通过并发decode请求数控制)
Y轴: Average TPOT (ms) 和 Batch Execution Time (ms)
"""

import os
import sys
import asyncio
import aiohttp
import time
import json
import random
import string
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from src.config import MODEL_PATH, REQUEST_TIMEOUT_SEC

# =============================================================================
# Configuration
# =============================================================================

# Server URL (single replica server)
SERVER_URL = "http://localhost:8300"

# Decode request configuration (simulates ongoing decode work)
DECODE_CONFIG = {
    "context": 2048,      # Existing prefix cache size
    "new_input": 32,      # Small new input (continuation)
    "output": 64,         # Number of tokens to generate
}

# Full-prefill configuration (simulates Turn 1 - no cache)
FULL_PREFILL_CONFIG = {
    "context": 0,         # No cache
    "input": 1024,        # Long input
    "output": 64,         # Generate some tokens to measure TPOT
}

# Append-prefill configuration (simulates PPD Turn 2+ - with cache)
APPEND_PREFILL_CONFIG = {
    "context": 2048,      # Full cache from previous turn
    "input": 128,         # Short new input
    "output": 64,         # Generate some tokens to measure TPOT
}

# Test parameters
BATCH_SIZES = [8, 16, 32, 64, 128]
REPEAT_TIMES = 10
WARMUP_REQUESTS = 5

# Sensitivity test parameters
SENSITIVITY_BATCH_SIZE = 64
APPEND_INPUT_SIZES = [32, 64, 128, 256, 512, 1024]

# Output directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "interference")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RequestResult:
    """Single request result"""
    request_type: str  # "decode", "full_prefill", "append_prefill"
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    output_tokens: int
    success: bool
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Batch execution result"""
    scenario: str  # "decode_only", "decode_with_full_prefill", "decode_with_append_prefill"
    batch_size: int
    decode_count: int
    prefill_count: int
    prefill_type: Optional[str]  # "full" or "append"

    # Timing metrics
    batch_execution_time_ms: float  # Total time for all requests to complete

    # Decode requests metrics
    decode_avg_tpot_ms: float
    decode_avg_ttft_ms: float
    decode_avg_e2e_ms: float

    # Prefill request metrics (if any)
    prefill_ttft_ms: Optional[float] = None
    prefill_tpot_ms: Optional[float] = None
    prefill_e2e_ms: Optional[float] = None

    # Success tracking
    decode_success_count: int = 0
    prefill_success: bool = True


@dataclass
class ExperimentResult:
    """Full experiment result"""
    experiment_type: str  # "core" or "sensitivity"
    timestamp: str
    results: List[Dict]

    # Metadata
    decode_config: Dict
    full_prefill_config: Dict
    append_prefill_config: Dict
    batch_sizes: List[int]
    repeat_times: int


# =============================================================================
# Utility Functions
# =============================================================================

def generate_random_text(num_tokens: int, seed: str = None) -> str:
    """Generate random text with approximately num_tokens tokens."""
    if seed:
        random.seed(hash(seed) % (2**32))

    # Approximate 1 token = 4 characters
    chars_needed = num_tokens * 4
    words = []
    chars_generated = 0

    while chars_generated < chars_needed:
        word_len = random.randint(3, 10)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)
        chars_generated += word_len + 1

    return ' '.join(words)


def generate_prompt_with_context(context_tokens: int, new_input_tokens: int,
                                  unique_id: str) -> Tuple[str, str]:
    """
    Generate a prompt that simulates having context (prefix cache).
    Returns (full_prompt, context_prefix) where context_prefix is the cacheable part.
    """
    # Generate context prefix (will be cached)
    context_prefix = f"CONTEXT_{unique_id}_START " + generate_random_text(
        context_tokens - 10, seed=f"context_{unique_id}"
    ) + " CONTEXT_END "

    # Generate new input
    new_input = f"NEW_INPUT_{unique_id} " + generate_random_text(
        new_input_tokens - 5, seed=f"input_{unique_id}_{time.time()}"
    )

    return context_prefix + new_input, context_prefix


# =============================================================================
# Request Execution
# =============================================================================

async def send_request(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int,
    request_type: str,
) -> RequestResult:
    """Send a single request and measure timing."""

    request_data = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start_time = time.perf_counter()
    ttft = 0.0
    first_token = False
    output_tokens = 0

    try:
        async with session.post(
            f"{SERVER_URL}/v1/completions",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return RequestResult(
                    request_type=request_type,
                    ttft_ms=0, tpot_ms=0, e2e_ms=0,
                    output_tokens=0, success=False,
                    error=f"HTTP {response.status}: {error_text[:100]}"
                )

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

                            # Detect first token
                            if not first_token and chunk.get("choices"):
                                text = chunk["choices"][0].get("text", "")
                                if text:
                                    ttft = (time.perf_counter() - start_time) * 1000
                                    first_token = True

                            # Get final token count
                            if chunk.get("usage"):
                                output_tokens = chunk["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

            e2e_ms = (time.perf_counter() - start_time) * 1000

            # Calculate TPOT
            if output_tokens > 1:
                tpot_ms = (e2e_ms - ttft) / (output_tokens - 1)
            else:
                tpot_ms = 0

            return RequestResult(
                request_type=request_type,
                ttft_ms=ttft,
                tpot_ms=tpot_ms,
                e2e_ms=e2e_ms,
                output_tokens=output_tokens,
                success=True
            )

    except asyncio.TimeoutError:
        return RequestResult(
            request_type=request_type,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            output_tokens=0, success=False,
            error="Timeout"
        )
    except Exception as e:
        return RequestResult(
            request_type=request_type,
            ttft_ms=0, tpot_ms=0, e2e_ms=0,
            output_tokens=0, success=False,
            error=str(e)[:100]
        )


async def warmup_cache(session: aiohttp.ClientSession, context_prefix: str):
    """Send a warmup request to establish prefix cache."""
    warmup_prompt = context_prefix + " Please respond briefly."

    request_data = {
        "model": MODEL_PATH,
        "prompt": warmup_prompt,
        "max_tokens": 10,
        "temperature": 0.8,
        "stream": False,
    }

    try:
        async with session.post(
            f"{SERVER_URL}/v1/completions",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            await response.text()
    except Exception as e:
        print(f"  Warmup failed: {e}")


# =============================================================================
# Batch Execution
# =============================================================================

async def run_batch(
    session: aiohttp.ClientSession,
    scenario: str,
    decode_count: int,
    context_prefix: str,  # Shared prefix for cache
    prefill_config: Optional[Dict] = None,  # None for decode-only
) -> BatchResult:
    """
    Run a batch of requests concurrently.

    For decode requests: use context_prefix + small new input
    For prefill request: use prefill_config (full or append)
    """
    batch_id = f"batch_{int(time.time()*1000)}"
    tasks = []

    # Create decode requests
    for i in range(decode_count):
        new_input = generate_random_text(
            DECODE_CONFIG["new_input"],
            seed=f"{batch_id}_decode_{i}"
        )
        prompt = context_prefix + f" DECODE_{i}: " + new_input

        tasks.append(send_request(
            session, prompt, DECODE_CONFIG["output"], "decode"
        ))

    # Create prefill request if specified
    prefill_type = None
    if prefill_config:
        if prefill_config.get("context", 0) == 0:
            # Full prefill - no cache
            prefill_type = "full"
            prefill_prompt = generate_random_text(
                prefill_config["input"],
                seed=f"{batch_id}_full_prefill"
            )
        else:
            # Append prefill - with cache
            prefill_type = "append"
            new_input = generate_random_text(
                prefill_config["input"],
                seed=f"{batch_id}_append_prefill"
            )
            prefill_prompt = context_prefix + f" APPEND_PREFILL: " + new_input

        tasks.append(send_request(
            session, prefill_prompt, prefill_config["output"], f"{prefill_type}_prefill"
        ))

    # Run all requests concurrently
    batch_start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    batch_end = time.perf_counter()

    batch_execution_time_ms = (batch_end - batch_start) * 1000

    # Separate decode and prefill results
    decode_results = []
    prefill_result = None

    for r in results:
        if isinstance(r, Exception):
            continue
        if r.request_type == "decode":
            decode_results.append(r)
        else:
            prefill_result = r

    # Calculate decode metrics
    successful_decodes = [r for r in decode_results if r.success]
    decode_success_count = len(successful_decodes)

    if successful_decodes:
        decode_avg_tpot = np.mean([r.tpot_ms for r in successful_decodes])
        decode_avg_ttft = np.mean([r.ttft_ms for r in successful_decodes])
        decode_avg_e2e = np.mean([r.e2e_ms for r in successful_decodes])
    else:
        decode_avg_tpot = decode_avg_ttft = decode_avg_e2e = 0

    # Build result
    result = BatchResult(
        scenario=scenario,
        batch_size=decode_count + (1 if prefill_config else 0),
        decode_count=decode_count,
        prefill_count=1 if prefill_config else 0,
        prefill_type=prefill_type,
        batch_execution_time_ms=batch_execution_time_ms,
        decode_avg_tpot_ms=decode_avg_tpot,
        decode_avg_ttft_ms=decode_avg_ttft,
        decode_avg_e2e_ms=decode_avg_e2e,
        decode_success_count=decode_success_count,
    )

    if prefill_result:
        result.prefill_ttft_ms = prefill_result.ttft_ms
        result.prefill_tpot_ms = prefill_result.tpot_ms
        result.prefill_e2e_ms = prefill_result.e2e_ms
        result.prefill_success = prefill_result.success

    return result


# =============================================================================
# Core Experiment
# =============================================================================

async def run_core_experiment() -> ExperimentResult:
    """
    Run the core three-line comparison experiment.

    For each batch size:
    1. decode-only baseline
    2. decode + one full-prefill
    3. decode + one append-prefill
    """
    print("=" * 70)
    print("Core Experiment: Prefill-Decode Interference Comparison")
    print("=" * 70)

    all_results = []

    connector = aiohttp.TCPConnector(limit=200, limit_per_host=200)
    async with aiohttp.ClientSession(connector=connector) as session:

        # First, establish prefix cache
        print("\nEstablishing prefix cache...")
        context_prefix = generate_random_text(
            DECODE_CONFIG["context"],
            seed="shared_context_prefix"
        )
        context_prefix = "SHARED_CONTEXT_START " + context_prefix + " CONTEXT_END "

        for _ in range(WARMUP_REQUESTS):
            await warmup_cache(session, context_prefix)
        print("  Cache established.")

        # Run experiments for each batch size
        for batch_size in BATCH_SIZES:
            print(f"\n--- Batch Size: {batch_size} ---")

            for repeat in range(REPEAT_TIMES):
                print(f"  Repeat {repeat + 1}/{REPEAT_TIMES}...", end=" ", flush=True)

                # Scenario 1: Decode-only
                result1 = await run_batch(
                    session, "decode_only", batch_size, context_prefix, None
                )
                all_results.append(asdict(result1))

                await asyncio.sleep(0.5)  # Brief pause between scenarios

                # Scenario 2: Decode + Full-prefill
                result2 = await run_batch(
                    session, "decode_with_full_prefill", batch_size,
                    context_prefix, FULL_PREFILL_CONFIG
                )
                all_results.append(asdict(result2))

                await asyncio.sleep(0.5)

                # Scenario 3: Decode + Append-prefill
                result3 = await run_batch(
                    session, "decode_with_append_prefill", batch_size,
                    context_prefix, APPEND_PREFILL_CONFIG
                )
                all_results.append(asdict(result3))

                print(f"TPOT: base={result1.decode_avg_tpot_ms:.1f}, "
                      f"full={result2.decode_avg_tpot_ms:.1f}, "
                      f"append={result3.decode_avg_tpot_ms:.1f}")

                await asyncio.sleep(0.5)

    return ExperimentResult(
        experiment_type="core",
        timestamp=datetime.now().isoformat(),
        results=all_results,
        decode_config=DECODE_CONFIG,
        full_prefill_config=FULL_PREFILL_CONFIG,
        append_prefill_config=APPEND_PREFILL_CONFIG,
        batch_sizes=BATCH_SIZES,
        repeat_times=REPEAT_TIMES,
    )


# =============================================================================
# Sensitivity Experiment
# =============================================================================

async def run_sensitivity_experiment() -> ExperimentResult:
    """
    Run the append-prefill size sensitivity experiment.

    Fixed decode batch size, varying append-prefill input length.
    """
    print("\n" + "=" * 70)
    print("Sensitivity Experiment: Append-prefill Size Impact")
    print("=" * 70)

    all_results = []

    connector = aiohttp.TCPConnector(limit=200, limit_per_host=200)
    async with aiohttp.ClientSession(connector=connector) as session:

        # Establish prefix cache
        print("\nEstablishing prefix cache...")
        context_prefix = generate_random_text(
            DECODE_CONFIG["context"],
            seed="sensitivity_context_prefix"
        )
        context_prefix = "SENSITIVITY_CONTEXT_START " + context_prefix + " CONTEXT_END "

        for _ in range(WARMUP_REQUESTS):
            await warmup_cache(session, context_prefix)
        print("  Cache established.")

        # Get decode-only baseline first
        print(f"\nGetting decode-only baseline (batch={SENSITIVITY_BATCH_SIZE})...")
        baseline_tpots = []
        for repeat in range(REPEAT_TIMES):
            result = await run_batch(
                session, "decode_only_baseline",
                SENSITIVITY_BATCH_SIZE, context_prefix, None
            )
            baseline_tpots.append(result.decode_avg_tpot_ms)
            all_results.append(asdict(result))
            await asyncio.sleep(0.3)

        baseline_avg_tpot = np.mean(baseline_tpots)
        print(f"  Baseline TPOT: {baseline_avg_tpot:.2f} ms")

        # Test different append-prefill sizes
        for input_size in APPEND_INPUT_SIZES:
            print(f"\n--- Append-prefill Input Size: {input_size} ---")

            append_config = {
                "context": DECODE_CONFIG["context"],
                "input": input_size,
                "output": 64,
            }

            for repeat in range(REPEAT_TIMES):
                print(f"  Repeat {repeat + 1}/{REPEAT_TIMES}...", end=" ", flush=True)

                result = await run_batch(
                    session, f"decode_with_append_{input_size}",
                    SENSITIVITY_BATCH_SIZE, context_prefix, append_config
                )

                # Calculate slowdown
                slowdown = ((result.decode_avg_tpot_ms - baseline_avg_tpot)
                           / baseline_avg_tpot * 100) if baseline_avg_tpot > 0 else 0

                result_dict = asdict(result)
                result_dict["append_input_size"] = input_size
                result_dict["decode_slowdown_percent"] = slowdown
                result_dict["baseline_tpot_ms"] = baseline_avg_tpot
                all_results.append(result_dict)

                print(f"TPOT={result.decode_avg_tpot_ms:.1f}ms, slowdown={slowdown:.1f}%")

                await asyncio.sleep(0.3)

    return ExperimentResult(
        experiment_type="sensitivity",
        timestamp=datetime.now().isoformat(),
        results=all_results,
        decode_config=DECODE_CONFIG,
        full_prefill_config=FULL_PREFILL_CONFIG,
        append_prefill_config=APPEND_PREFILL_CONFIG,
        batch_sizes=[SENSITIVITY_BATCH_SIZE],
        repeat_times=REPEAT_TIMES,
    )


# =============================================================================
# Main
# =============================================================================

def save_results(result: ExperimentResult, filename: str):
    """Save experiment results to JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\nResults saved to: {filepath}")


async def check_server_health() -> bool:
    """Check if the server is healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{SERVER_URL}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return response.status == 200
    except:
        return False


async def main():
    parser = argparse.ArgumentParser(description="Interference Benchmark")
    parser.add_argument("--core", action="store_true", help="Run core experiment")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity experiment")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--server-url", type=str, default=SERVER_URL,
                       help="Server URL (default: http://localhost:8300)")
    args = parser.parse_args()

    global SERVER_URL
    SERVER_URL = args.server_url

    if not any([args.core, args.sensitivity, args.all]):
        args.all = True

    print("=" * 70)
    print("Interference Benchmark - Prefill-Decode Interference Test")
    print("=" * 70)
    print(f"Server URL: {SERVER_URL}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Check server health
    print("\nChecking server health...")
    if not await check_server_health():
        print("ERROR: Server is not healthy. Please start the server first.")
        print("  Run: ./scripts/server/start_single_replica.sh")
        sys.exit(1)
    print("  Server is healthy.")

    # Run experiments
    if args.core or args.all:
        core_result = await run_core_experiment()
        save_results(core_result, "core_experiment.json")

    if args.sensitivity or args.all:
        sensitivity_result = await run_sensitivity_experiment()
        save_results(sensitivity_result, "sensitivity_experiment.json")

    print("\n" + "=" * 70)
    print("All experiments complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
