#!/usr/bin/env python3
"""
OOM Boundary Test - Find the maximum load the system can handle.

Strategy:
1. Test most resource-intensive workload: large_huge_paste (1024→1024 T1 + 1024→32 T2)
2. Test on most constrained config: 1P_3D (single prefill bottleneck)
3. Progressively increase QPS from 1 to 30 to find OOM point
"""

import os
import sys
import asyncio
import subprocess

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from scripts.benchmark.comprehensive_benchmark import (
    run_benchmark_point,
    T1_CONFIGS, T2_CONFIGS,
    start_config,
    run_cleanup,
    check_server_health,
    warmup_servers
)

async def test_oom_boundary():
    """Find OOM boundary by testing progressively higher QPS."""
    config = "1P_3D"
    workload = "large_huge_paste"

    print("="*80)
    print("OOM Boundary Test")
    print("="*80)
    print(f"Configuration: {config}")
    print(f"Workload: {workload}")
    print(f"T1: 1024 input → 1024 output (2048 context)")
    print(f"T2: 1024 input → 32 output (extreme prefill-heavy)")
    print("="*80)

    # Start server
    print("\nStarting servers...")
    run_cleanup()
    await asyncio.sleep(5)

    if not start_config(config):
        print("ERROR: Failed to start server")
        return

    await asyncio.sleep(10)  # Wait for server ready

    # Check health
    healthy, error = await check_server_health()
    if not healthy:
        print(f"ERROR: Server not healthy: {error}")
        return

    # Warmup
    print("\nWarming up...")
    if not await warmup_servers():
        print("WARNING: Warmup had issues")

    # Get workload config
    t1_name, t2_name = workload.split("_", 1)
    t1_config = T1_CONFIGS[t1_name]
    t2_config = T2_CONFIGS[t2_name]

    # Test QPS points
    qps_points = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

    results = []

    for qps in qps_points:
        print(f"\n{'='*80}")
        print(f"Testing QPS = {qps}")
        print(f"{'='*80}")

        try:
            result, server_healthy = await run_benchmark_point(
                config, workload, qps, t1_config, t2_config
            )

            results.append({
                "qps": qps,
                "success": result.success_rate > 0,
                "success_rate": result.success_rate,
                "t1_avg_e2e": result.turn1.avg_e2e_ms,
                "t2_avg_e2e": result.turn2.avg_e2e_ms,
                "t1_p99_ttft": result.turn1.p99_ttft_ms,
                "t2_p99_ttft": result.turn2.p99_ttft_ms,
                "throughput_rps": result.throughput["requests_per_sec"],
                "throughput_tps": result.throughput["tokens_per_sec"],
                "error": result.error,
                "server_healthy": server_healthy,
            })

            print(f"✓ Success Rate: {result.success_rate:.1f}%")
            print(f"  T1 E2E: {result.turn1.avg_e2e_ms:.1f}ms (P99 TTFT: {result.turn1.p99_ttft_ms:.1f}ms)")
            print(f"  T2 E2E: {result.turn2.avg_e2e_ms:.1f}ms (P99 TTFT: {result.turn2.p99_ttft_ms:.1f}ms)")
            print(f"  Throughput: {result.throughput['requests_per_sec']:.2f} req/s, {result.throughput['tokens_per_sec']:.1f} tok/s")

            if not server_healthy:
                print(f"✗ Server became unhealthy, stopping test")
                break

            if result.success_rate < 50:
                print(f"✗ Success rate too low (<50%), likely hitting OOM")
                break

            # Give server time to recover between tests
            print(f"Waiting 10s before next test...")
            await asyncio.sleep(10)

        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append({
                "qps": qps,
                "success": False,
                "error": str(e),
            })
            break

    # Summary
    print(f"\n{'='*80}")
    print("OOM Boundary Test Results")
    print(f"{'='*80}")
    print(f"{'QPS':<6} {'Success%':<10} {'T1 E2E':<12} {'T2 E2E':<12} {'RPS':<8} {'TPS':<10} {'Status'}")
    print("-"*80)

    for r in results:
        if r["success"]:
            status = "✓ OK" if r["success_rate"] > 90 else "⚠ Degraded"
            print(f"{r['qps']:<6} {r['success_rate']:<10.1f} {r['t1_avg_e2e']:<12.1f} {r['t2_avg_e2e']:<12.1f} "
                  f"{r['throughput_rps']:<8.2f} {r['throughput_tps']:<10.1f} {status}")
        else:
            print(f"{r['qps']:<6} {'FAILED':<10} {'-':<12} {'-':<12} {'-':<8} {'-':<10} ✗ {r.get('error', 'Unknown')[:20]}")

    print("\n" + "="*80)

    # Find boundary
    successful_qps = [r["qps"] for r in results if r["success"] and r.get("success_rate", 0) > 90]
    if successful_qps:
        max_qps = max(successful_qps)
        print(f"Maximum stable QPS: {max_qps}")

        # Find first failure
        all_qps = [r["qps"] for r in results]
        if max_qps < max(all_qps):
            failure_idx = all_qps.index(max_qps) + 1
            if failure_idx < len(results):
                failure_result = results[failure_idx]
                print(f"First failure at QPS {failure_result['qps']}: {failure_result.get('error', 'Low success rate')}")
    else:
        print("No successful QPS point found!")

    print("="*80)

    # Cleanup
    run_cleanup()

if __name__ == "__main__":
    asyncio.run(test_oom_boundary())
