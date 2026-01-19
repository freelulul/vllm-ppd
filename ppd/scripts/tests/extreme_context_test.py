#!/usr/bin/env python3
"""
Extreme Context Test - Finding Memory Capacity Limits

Tests fixed config at fixed QPS with progressively larger context sizes
to find the true memory capacity upper limit (beyond workload bottlenecks).

Test Matrix:
- Config (1): 2P_2pD (representative baseline with good balance)
- QPS (1): 20 (known stable point for 2048-token context)
- Context sizes (5): 4096, 6144, 8192, 12288, 16384 tokens

Total: 1 × 1 × 5 = 5 test points (~30-60 minutes)

Memory Estimation (Llama-3.1-8B):
- Single request KV cache = context_size × 32 layers × 2 (K+V) × 128 × 8 heads × 2 bytes
- 2048 tokens:  ~1 GB per request
- 4096 tokens:  ~2 GB per request
- 8192 tokens:  ~4 GB per request
- 16384 tokens: ~8 GB per request

Expected OOM:
- At QPS=20, throughput ~4 req/s, avg latency ~50s
- Concurrent requests = 4 × 50 = 200 requests
- 4096 tokens: 200 × 2GB = 400GB (100GB per GPU) → Near pool limit
- 8192 tokens: 200 × 4GB = 800GB → Severe OOM expected
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from scripts.benchmark.comprehensive_benchmark import (
    run_benchmark_point,
    start_config,
    run_cleanup,
    check_server_health,
    warmup_servers,
    BenchmarkResult,
)

# Import OOM detection from high_qps_boundary_test
from scripts.tests.high_qps_boundary_test import is_oom_failure, categorize_failure

# ============================================================================
# Configuration
# ============================================================================

# Representative configuration with good balance
CONFIG = "2P_2pD"

# Fixed QPS (known stable for 2048-token context)
QPS = 20

# Extreme context sizes (in tokens)
CONTEXT_SIZES = [4096, 6144, 8192, 12288, 16384]

# Output directory
OUTPUT_DIR = Path(PROJECT_DIR) / "results" / "extreme_context"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Dynamic Workload Generation
# ============================================================================

def generate_extreme_workload(context_size: int) -> Tuple[Dict, Dict, str]:
    """
    Generate T1/T2 configs for a given context size.

    Strategy:
    - T1: Use full context (split evenly between input/output)
    - T2: Keep small to isolate T1 memory impact (64 input, 32 output)

    Returns:
        (t1_config, t2_config, workload_name)
    """
    t1_input = context_size // 2
    t1_output = context_size // 2

    t1_config = {"input": t1_input, "output": t1_output}
    t2_config = {"input": 64, "output": 32}  # Keep T2 minimal

    workload_name = f"extreme_{context_size}"

    return t1_config, t2_config, workload_name


def estimate_memory_per_request(context_size: int) -> float:
    """
    Estimate KV cache memory per request (in GB).

    Formula for Llama-3.1-8B:
    memory = context_size × 32 layers × 2 (K+V) × 128 dim_per_head × 8 heads × 2 bytes

    Returns:
        Memory in GB
    """
    num_layers = 32
    num_kv_heads = 8
    head_dim = 128
    bytes_per_element = 2  # FP16

    total_bytes = context_size * num_layers * 2 * head_dim * num_kv_heads * bytes_per_element
    total_gb = total_bytes / (1024 ** 3)

    return total_gb


# ============================================================================
# Main Test Logic
# ============================================================================

async def test_context_size(context_size: int) -> Dict:
    """
    Test a single context size.

    Returns result dict.
    """
    print(f"\n{'='*80}")
    print(f"Testing Context Size: {context_size} tokens")
    print(f"{'='*80}")

    # Generate workload
    t1_config, t2_config, workload_name = generate_extreme_workload(context_size)

    print(f"Workload: {workload_name}")
    print(f"  T1: {t1_config['input']} input → {t1_config['output']} output = {context_size} tokens")
    print(f"  T2: {t2_config['input']} input → {t2_config['output']} output = {t2_config['input'] + t2_config['output']} tokens")
    print(f"  Estimated memory per request: {estimate_memory_per_request(context_size):.2f} GB")
    print()

    try:
        # Run test point
        result, server_healthy = await run_benchmark_point(
            CONFIG, workload_name, QPS, t1_config, t2_config
        )

        # Categorize result
        category = categorize_failure(result, server_healthy)
        is_oom, oom_reason = is_oom_failure(result, server_healthy)

        # Calculate memory metrics
        estimated_mem_per_req = estimate_memory_per_request(context_size)
        throughput_rps = result.throughput["requests_per_sec"]
        avg_latency_s = result.turn1.avg_e2e_ms / 1000 if result.turn1.avg_e2e_ms > 0 else 50
        estimated_concurrency = throughput_rps * avg_latency_s
        estimated_total_mem = estimated_mem_per_req * estimated_concurrency

        # Save result
        result_dict = {
            "context_size": context_size,
            "success_rate": result.success_rate,
            "category": category,
            "is_oom": is_oom,
            "oom_reason": oom_reason,
            "server_healthy": server_healthy,
            "total_requests": result.total_requests,
            "failed_requests": result.failed_requests,
            "t1_avg_ttft": result.turn1.avg_ttft_ms,
            "t1_p99_ttft": result.turn1.p99_ttft_ms,
            "t1_avg_e2e": result.turn1.avg_e2e_ms,
            "t2_avg_ttft": result.turn2.avg_ttft_ms,
            "throughput_rps": throughput_rps,
            "estimated_memory": {
                "mem_per_request_gb": estimated_mem_per_req,
                "estimated_concurrency": estimated_concurrency,
                "estimated_total_mem_gb": estimated_total_mem,
            },
            "error": result.error,
        }

        # Print summary
        print(f"✓ Success Rate: {result.success_rate:.1f}% ({result.total_requests - result.failed_requests}/{result.total_requests})")
        print(f"  Category: {category.upper()}")
        print(f"  T1: Avg TTFT={result.turn1.avg_ttft_ms:.0f}ms, P99={result.turn1.p99_ttft_ms:.0f}ms, E2E={result.turn1.avg_e2e_ms:.0f}ms")
        print(f"  T2: Avg TTFT={result.turn2.avg_ttft_ms:.0f}ms, E2E={result.turn2.avg_e2e_ms:.0f}ms")
        print(f"  Throughput: {throughput_rps:.2f} req/s")
        print(f"  Estimated concurrency: {estimated_concurrency:.1f} requests")
        print(f"  Estimated total memory: {estimated_total_mem:.1f} GB ({estimated_total_mem/4:.1f} GB per GPU)")

        if is_oom:
            print(f"\n⚠️  OOM DETECTED: {oom_reason}")
            print(f"  Memory limit reached at context size: {context_size} tokens")
        elif not server_healthy:
            print(f"\n✗ Server became unhealthy")

        return result_dict

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

        return {
            "context_size": context_size,
            "success_rate": 0,
            "category": "error",
            "is_oom": False,
            "oom_reason": str(e),
            "error": str(e),
        }


async def extreme_context_test():
    """Main test orchestration."""

    print("="*80)
    print("Extreme Context Test - Finding Memory Capacity Limits")
    print("="*80)
    print(f"Configuration: {CONFIG}")
    print(f"Fixed QPS: {QPS}")
    print(f"Context sizes: {CONTEXT_SIZES}")
    print(f"Total test points: {len(CONTEXT_SIZES)}")
    print("="*80)

    # Start server
    print(f"\nStarting {CONFIG} servers...")
    run_cleanup()
    await asyncio.sleep(8)

    if not start_config(CONFIG):
        print(f"ERROR: Failed to start {CONFIG}")
        return

    print("Waiting for server startup...")
    await asyncio.sleep(15)

    # Health check
    healthy, error = await check_server_health()
    if not healthy:
        print(f"ERROR: Server not healthy: {error}")
        run_cleanup()
        return

    # Warmup
    print("\nWarming up servers...")
    if not await warmup_servers():
        print("WARNING: Warmup had issues")

    await asyncio.sleep(5)

    # Test each context size
    results = []

    for context_size in CONTEXT_SIZES:
        result = await test_context_size(context_size)
        results.append(result)

        # Check if we should continue
        if result["category"] == "oom" or not result.get("server_healthy", True):
            print(f"\n⚠️  Stopping test - OOM or server crash detected")
            print(f"  Maximum viable context size: {CONTEXT_SIZES[CONTEXT_SIZES.index(context_size)-1] if CONTEXT_SIZES.index(context_size) > 0 else 'Unknown'}")
            break

        # Wait for stabilization (longer for larger contexts)
        wait_time = 30 if context_size >= 8192 else 20
        print(f"\nWaiting {wait_time}s for stabilization...")
        await asyncio.sleep(wait_time)

    # Cleanup
    print(f"\nCleaning up {CONFIG}...")
    run_cleanup()
    await asyncio.sleep(10)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"extreme_context_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "config": CONFIG,
                "qps": QPS,
                "context_sizes": CONTEXT_SIZES,
            },
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    # Generate summary
    print_summary(results)

    return results


def print_summary(results: List[Dict]):
    """Print summary analysis of results."""

    print(f"\n{'='*80}")
    print("Summary: Extreme Context Test Results")
    print(f"{'='*80}")
    print(f"Configuration: {CONFIG}")
    print(f"Fixed QPS: {QPS}")
    print(f"{'-'*80}")
    print(f"{'Context':<10} {'Success%':<10} {'Category':<12} {'Throughput':<12} {'Est Mem/GPU':<15} {'OOM?'}")
    print(f"{'-'*80}")

    max_viable_context = None
    oom_context = None

    for r in results:
        context = r["context_size"]
        success = r.get("success_rate", 0)
        category = r.get("category", "unknown")
        throughput = r.get("throughput_rps", 0)
        is_oom = r.get("is_oom", False)

        est_mem = r.get("estimated_memory", {})
        mem_per_gpu = est_mem.get("estimated_total_mem_gb", 0) / 4

        status_symbol = "✓" if category == "success" else ("⚠" if category == "degradation" else "✗")
        oom_marker = "🔥 OOM" if is_oom else ""

        print(f"{context:<10} {success:<10.1f} {category:<12} {throughput:<12.2f} {mem_per_gpu:<15.1f} {oom_marker}")

        if category == "success":
            max_viable_context = context

        if is_oom and oom_context is None:
            oom_context = context

    print(f"{'-'*80}")
    print(f"\n📊 Analysis:")
    print(f"  Maximum viable context size: {max_viable_context if max_viable_context else 'None (all failed)'} tokens")

    if oom_context:
        print(f"  OOM boundary: {oom_context} tokens")
        if max_viable_context:
            print(f"  Memory headroom: {max_viable_context} → {oom_context} tokens ({(oom_context/max_viable_context - 1)*100:.1f}% increase)")
    else:
        print(f"  No OOM detected - all tested contexts viable or prefill-limited")

    print(f"\n💡 Recommendation:")
    if max_viable_context:
        conservative_limit = int(max_viable_context * 0.75)
        print(f"  Conservative production limit: {conservative_limit} tokens (75% of max viable)")
        print(f"  Do NOT exceed: {max_viable_context} tokens")
    else:
        print(f"  ⚠️  Configuration {CONFIG} at QPS={QPS} cannot handle tested context sizes")
        print(f"  Consider: Reducing QPS or increasing GPU resources")

    print(f"{'='*80}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    asyncio.run(extreme_context_test())
