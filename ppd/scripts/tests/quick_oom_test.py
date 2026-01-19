#!/usr/bin/env python3
"""
Quick OOM Boundary Validation Test

Tests 1P_3D config (most constrained) with large_huge_paste (most intensive)
at strategic QPS points to find OOM boundary.
"""

import os
import sys
import asyncio

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

async def quick_oom_test():
    """Quick test at strategic QPS points to find OOM boundary."""
    config = "1P_3D"
    workload = "large_huge_paste"

    print("="*80)
    print("Quick OOM Boundary Validation Test")
    print("="*80)
    print(f"Configuration: {config} (most constrained - single prefill bottleneck)")
    print(f"Workload: {workload} (most intensive)")
    print(f"  T1: 1024 input → 1024 output (2048 context)")
    print(f"  T2: 1024 input → 32 output (extreme prefill-heavy)")
    print("="*80)

    # Start server
    print("\nStarting servers...")
    run_cleanup()
    await asyncio.sleep(8)

    if not start_config(config):
        print("ERROR: Failed to start server")
        return

    print("Waiting for server startup...")
    await asyncio.sleep(15)

    # Check health
    healthy, error = await check_server_health()
    if not healthy:
        print(f"ERROR: Server not healthy: {error}")
        return

    # Warmup
    print("\nWarming up...")
    if not await warmup_servers():
        print("WARNING: Warmup had issues")

    await asyncio.sleep(5)

    # Get workload config
    t1_name, t2_name = workload.split("_", 1)
    t1_config = T1_CONFIGS[t1_name]
    t2_config = T2_CONFIGS[t2_name]

    # Strategic QPS points: start low, increase to find boundary
    qps_points = [2, 4, 6, 8, 10]

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
                "t1_avg_ttft": result.turn1.avg_ttft_ms,
                "throughput_rps": result.throughput["requests_per_sec"],
                "total_reqs": result.total_requests,
                "failed_reqs": result.failed_requests,
                "error": result.error,
                "server_healthy": server_healthy,
            })

            print(f"✓ Success Rate: {result.success_rate:.1f}% ({result.total_requests - result.failed_requests}/{result.total_requests})")
            print(f"  T1: Avg TTFT={result.turn1.avg_ttft_ms:.0f}ms, P99 TTFT={result.turn1.p99_ttft_ms:.0f}ms, Avg E2E={result.turn1.avg_e2e_ms:.0f}ms")
            print(f"  T2: Avg TTFT={result.turn2.avg_ttft_ms:.0f}ms, P99 TTFT={result.turn2.p99_ttft_ms:.0f}ms, Avg E2E={result.turn2.avg_e2e_ms:.0f}ms")
            print(f"  Throughput: {result.throughput['requests_per_sec']:.2f} req/s")
            if result.error:
                print(f"  ⚠ Errors: {result.error}")

            if not server_healthy:
                print(f"✗ Server became unhealthy - likely OOM or crash")
                print(f"  OOM BOUNDARY FOUND: Between QPS={qps_points[qps_points.index(qps)-1] if qps_points.index(qps) > 0 else 0} and QPS={qps}")
                break

            if result.success_rate < 50:
                print(f"✗ Success rate < 50% - severe overload")
                print(f"  FAILURE BOUNDARY: QPS={qps}")
                break

            # Longer wait for heavily loaded system to stabilize
            wait_time = 15 if qps >= 6 else 10
            print(f"Waiting {wait_time}s for system to stabilize...")
            await asyncio.sleep(wait_time)

        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "qps": qps,
                "success": False,
                "error": str(e),
            })
            break

    # Summary
    print(f"\n{'='*80}")
    print("Quick OOM Validation Results")
    print(f"{'='*80}")
    print(f"Configuration: {config}")
    print(f"Workload: {workload}")
    print("-"*80)
    print(f"{'QPS':<5} {'Succ%':<7} {'T1 TTFT':<12} {'T1 P99':<12} {'RPS':<8} {'Total/Fail':<12} {'Status'}")
    print("-"*80)

    for r in results:
        if r["success"]:
            status = "✓ OK" if r["success_rate"] >= 95 else ("⚠ Degraded" if r["success_rate"] >= 50 else "✗ Failed")
            t1_ttft = f"{r['t1_avg_ttft']:.0f}ms" if r['t1_avg_ttft'] < 1000 else f"{r['t1_avg_ttft']/1000:.1f}s"
            t1_p99 = f"{r['t1_p99_ttft']:.0f}ms" if r['t1_p99_ttft'] < 1000 else f"{r['t1_p99_ttft']/1000:.1f}s"
            print(f"{r['qps']:<5} {r['success_rate']:<7.1f} {t1_ttft:<12} {t1_p99:<12} {r['throughput_rps']:<8.2f} "
                  f"{r['total_reqs']}/{r['failed_reqs']:<12} {status}")
        else:
            print(f"{r['qps']:<5} {'FAIL':<7} {'-':<12} {'-':<12} {'-':<8} {'-':<12} ✗ {r.get('error', 'Unknown')[:30]}")

    print("-"*80)

    # Analysis
    successful = [r for r in results if r["success"] and r.get("success_rate", 0) >= 95]
    degraded = [r for r in results if r["success"] and 50 <= r.get("success_rate", 0) < 95]
    failed = [r for r in results if not r["success"] or r.get("success_rate", 0) < 50]

    print("\n📊 Analysis:")
    if successful:
        max_stable = max(r["qps"] for r in successful)
        print(f"✓ Maximum stable QPS (≥95% success): {max_stable}")

        # Show performance degradation
        if len(successful) > 1:
            baseline = successful[0]
            peak = successful[-1]
            ttft_increase = (peak["t1_avg_ttft"] - baseline["t1_avg_ttft"]) / baseline["t1_avg_ttft"] * 100
            print(f"  TTFT increase from QPS={baseline['qps']} to QPS={peak['qps']}: {ttft_increase:.0f}%")
            print(f"    Baseline (QPS={baseline['qps']}): {baseline['t1_avg_ttft']:.0f}ms")
            print(f"    Peak (QPS={peak['qps']}): {peak['t1_avg_ttft']:.0f}ms")

    if degraded:
        print(f"⚠ Degraded performance at QPS: {', '.join(str(r['qps']) for r in degraded)}")
        for r in degraded:
            print(f"  QPS={r['qps']}: {r['success_rate']:.1f}% success, {r['failed_reqs']}/{r['total_reqs']} failed")

    if failed:
        first_failure = failed[0]
        print(f"✗ First failure at QPS={first_failure['qps']}")
        if first_failure.get("error"):
            print(f"  Error: {first_failure['error']}")

        # Determine failure type
        if "OOM" in str(first_failure.get("error", "")):
            print(f"  Failure type: OOM (Out of Memory)")
        elif "Timeout" in str(first_failure.get("error", "")):
            print(f"  Failure type: Request timeouts (queue buildup)")
        elif not first_failure.get("server_healthy", True):
            print(f"  Failure type: Server crash (likely OOM)")
        else:
            print(f"  Failure type: Unknown")

    print("\n💡 Recommendations:")
    if successful:
        max_stable = max(r["qps"] for r in successful)
        conservative = max_stable * 0.6
        moderate = max_stable * 0.75
        print(f"  Conservative production limit: {conservative:.1f} QPS (60% of stable max)")
        print(f"  Moderate production limit: {moderate:.1f} QPS (75% of stable max)")
        print(f"  Do NOT exceed: {max_stable} QPS")
    else:
        print(f"  ⚠ No stable QPS found! Config {config} cannot handle {workload}.")
        print(f"  Consider: Increasing GPU capacity or reducing workload intensity")

    print("="*80)

    # Cleanup
    run_cleanup()

if __name__ == "__main__":
    asyncio.run(quick_oom_test())
