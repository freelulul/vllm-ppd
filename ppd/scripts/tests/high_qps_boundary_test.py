#!/usr/bin/env python3
"""
High QPS Boundary Test - Finding True OOM Limits

Tests multi-prefill configurations at extreme QPS to bypass prefill bottleneck
and find the true memory OOM boundary (not just timeout/queue buildup).

Test Matrix:
- Configs (4): 3P_1D, 3P_1pD, 2P_2D, 2P_2pD (multi-P to avoid prefill saturation)
- Workloads (2): large_very_long_gen, large_huge_paste (max memory pressure)
- QPS (6): 30, 40, 50, 60, 80, 100 (far beyond normal range)

Total: 4 × 2 × 6 = 48 test points (~2 hours)

OOM Detection Criteria:
1. Server crash (health check fails)
2. Error message contains "out of memory" or "cuda"
3. GPU memory >75GB and requests fail
4. Success rate <50%
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
    T1_CONFIGS,
    T2_CONFIGS,
    start_config,
    run_cleanup,
    check_server_health,
    warmup_servers,
    BenchmarkResult,
)

# ============================================================================
# Configuration
# ============================================================================

# Multi-prefill configs to bypass prefill bottleneck
CONFIGS = ["3P_1D", "3P_1pD", "2P_2D", "2P_2pD"]

# Heavy workloads to maximize memory pressure
WORKLOADS = [
    "large_very_long_gen",   # 1024→1024 (T1) + 64→1024 (T2) = max decode pressure
    "large_huge_paste",      # 1024→1024 (T1) + 1024→32 (T2) = max prefill pressure
]

# Extreme QPS points
HIGH_QPS = [30, 40, 50, 60, 80, 100]

# Output directory
OUTPUT_DIR = Path(PROJECT_DIR) / "results" / "oom_boundary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# OOM Detection
# ============================================================================

def is_oom_failure(result: BenchmarkResult, server_healthy: bool) -> Tuple[bool, str]:
    """
    Determine if failure is due to OOM (vs timeout/degradation).

    Returns:
        (is_oom, reason)
    """
    # Server crash = likely OOM
    if not server_healthy:
        return True, "Server crash (likely OOM)"

    # Check error message
    if result.error:
        error_lower = result.error.lower()
        if "out of memory" in error_lower or "oom" in error_lower:
            return True, f"OOM error: {result.error[:100]}"
        if "cuda" in error_lower and "error" in error_lower:
            return True, f"CUDA error (likely OOM): {result.error[:100]}"

    # Severe degradation (but not timeout)
    if result.success_rate < 50 and result.success_rate > 0:
        # Check if it's timeout vs OOM
        if "timeout" in (result.error or "").lower():
            return False, "Timeout/queue buildup"
        else:
            return True, f"Severe degradation ({result.success_rate:.1f}% success)"

    return False, "Not OOM"


def categorize_failure(result: BenchmarkResult, server_healthy: bool) -> str:
    """
    Categorize failure type for analysis.

    Returns:
        "success", "timeout", "oom", "degradation"
    """
    if result.success_rate >= 95:
        return "success"

    is_oom, reason = is_oom_failure(result, server_healthy)
    if is_oom:
        return "oom"

    if result.success_rate < 50:
        return "timeout"  # Severe failure = timeout/queue buildup

    return "degradation"  # 50-95% success


# ============================================================================
# Main Test Logic
# ============================================================================

async def test_config_workload(config: str, workload: str) -> List[Dict]:
    """
    Test a single config-workload pair across all HIGH_QPS points.
    Stops when OOM is detected.

    Returns list of result dicts.
    """
    print(f"\n{'='*80}")
    print(f"Testing {config} - {workload}")
    print(f"{'='*80}")

    # Parse workload
    t1_name, t2_name = workload.split("_", 1)
    t1_config = T1_CONFIGS[t1_name]
    t2_config = T2_CONFIGS[t2_name]

    results = []

    for qps in HIGH_QPS:
        print(f"\n{'-'*80}")
        print(f"QPS = {qps}")
        print(f"{'-'*80}")

        try:
            # Run test point
            result, server_healthy = await run_benchmark_point(
                config, workload, qps, t1_config, t2_config
            )

            # Categorize result
            category = categorize_failure(result, server_healthy)
            is_oom, oom_reason = is_oom_failure(result, server_healthy)

            # Save result
            result_dict = {
                "qps": qps,
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
                "throughput_rps": result.throughput["requests_per_sec"],
                "error": result.error,
            }
            results.append(result_dict)

            # Print summary
            print(f"✓ Success Rate: {result.success_rate:.1f}% ({result.total_requests - result.failed_requests}/{result.total_requests})")
            print(f"  Category: {category.upper()}")
            print(f"  T1: Avg TTFT={result.turn1.avg_ttft_ms:.0f}ms, P99={result.turn1.p99_ttft_ms:.0f}ms, E2E={result.turn1.avg_e2e_ms:.0f}ms")
            print(f"  T2: Avg TTFT={result.turn2.avg_ttft_ms:.0f}ms, E2E={result.turn2.avg_e2e_ms:.0f}ms")
            print(f"  Throughput: {result.throughput['requests_per_sec']:.2f} req/s")

            if is_oom:
                print(f"\n⚠️  OOM DETECTED: {oom_reason}")
                print(f"  OOM boundary found between QPS={HIGH_QPS[HIGH_QPS.index(qps)-1] if HIGH_QPS.index(qps) > 0 else 0} and QPS={qps}")
                break  # Stop testing higher QPS for this workload

            if not server_healthy:
                print(f"\n✗ Server became unhealthy - stopping test")
                break

            # Wait for stabilization (longer for high QPS)
            wait_time = 20 if qps >= 50 else 15
            print(f"\nWaiting {wait_time}s for stabilization...")
            await asyncio.sleep(wait_time)

        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                "qps": qps,
                "success_rate": 0,
                "category": "error",
                "is_oom": False,
                "oom_reason": str(e),
                "error": str(e),
            })
            break

    return results


async def high_qps_boundary_test():
    """Main test orchestration."""

    print("="*80)
    print("High QPS Boundary Test - Finding True OOM Limits")
    print("="*80)
    print(f"Configs: {CONFIGS}")
    print(f"Workloads: {WORKLOADS}")
    print(f"High QPS: {HIGH_QPS}")
    print(f"Total test points: {len(CONFIGS)} × {len(WORKLOADS)} × {len(HIGH_QPS)} = {len(CONFIGS) * len(WORKLOADS) * len(HIGH_QPS)}")
    print("="*80)

    all_results = {}

    for config in CONFIGS:
        print(f"\n\n{'#'*80}")
        print(f"# Testing Configuration: {config}")
        print(f"{'#'*80}")

        # Start server
        print(f"\nStarting {config} servers...")
        run_cleanup()
        await asyncio.sleep(8)

        if not start_config(config):
            print(f"ERROR: Failed to start {config}")
            continue

        print("Waiting for server startup...")
        await asyncio.sleep(15)

        # Health check
        healthy, error = await check_server_health()
        if not healthy:
            print(f"ERROR: Server not healthy: {error}")
            run_cleanup()
            continue

        # Warmup
        print("\nWarming up servers...")
        if not await warmup_servers():
            print("WARNING: Warmup had issues")

        await asyncio.sleep(5)

        # Test each workload
        config_results = {}
        for workload in WORKLOADS:
            workload_results = await test_config_workload(config, workload)
            config_results[workload] = workload_results

        all_results[config] = config_results

        # Cleanup after config
        print(f"\nCleaning up {config}...")
        run_cleanup()
        await asyncio.sleep(10)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"high_qps_boundary_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "configs": CONFIGS,
                "workloads": WORKLOADS,
                "qps_points": HIGH_QPS,
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    # Generate summary
    print_summary(all_results)

    return all_results


def print_summary(all_results: Dict):
    """Print summary analysis of all results."""

    print(f"\n{'='*80}")
    print("Summary: High QPS Boundary Test Results")
    print(f"{'='*80}")

    for config, config_results in all_results.items():
        print(f"\n{'-'*80}")
        print(f"Configuration: {config}")
        print(f"{'-'*80}")

        for workload, workload_results in config_results.items():
            print(f"\n  Workload: {workload}")
            print(f"  {'QPS':<6} {'Success%':<10} {'Category':<12} {'Throughput':<12} {'OOM?'}")
            print(f"  {'-'*70}")

            oom_boundary = None
            max_stable_qps = None

            for r in workload_results:
                qps = r["qps"]
                success = r.get("success_rate", 0)
                category = r.get("category", "unknown")
                throughput = r.get("throughput_rps", 0)
                is_oom = r.get("is_oom", False)

                status_symbol = "✓" if category == "success" else ("⚠" if category == "degradation" else "✗")
                oom_marker = "🔥 OOM" if is_oom else ""

                print(f"  {qps:<6} {success:<10.1f} {category:<12} {throughput:<12.2f} {oom_marker}")

                if category == "success":
                    max_stable_qps = qps

                if is_oom and oom_boundary is None:
                    oom_boundary = qps

            # Summary for this workload
            print(f"\n  → Max stable QPS: {max_stable_qps if max_stable_qps else 'None (all failed)'}")
            if oom_boundary:
                print(f"  → OOM boundary: QPS = {oom_boundary}")
            else:
                print(f"  → No OOM detected (prefill-limited or all stable)")

    print(f"\n{'='*80}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    asyncio.run(high_qps_boundary_test())
