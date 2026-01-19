#!/usr/bin/env python3
"""Quick validation of high_qps and extreme_context test scripts."""

import os
import sys
import asyncio

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
)
from scripts.tests.high_qps_boundary_test import is_oom_failure, categorize_failure
from scripts.tests.extreme_context_test import generate_extreme_workload, estimate_memory_per_request

async def validate_high_qps():
    """Validate high_qps_boundary_test.py - single test point."""
    print("\n" + "="*80)
    print("Validating high_qps_boundary_test.py")
    print("="*80)
    print("Testing: 2P_2pD, large_very_long_gen, QPS=30")

    config = "2P_2pD"
    workload = "large_very_long_gen"
    qps = 30

    # Start server
    print("\nStarting servers...")
    run_cleanup()
    await asyncio.sleep(8)

    if not start_config(config):
        print(f"ERROR: Failed to start {config}")
        return False

    await asyncio.sleep(15)

    # Health check
    healthy, error = await check_server_health()
    if not healthy:
        print(f"ERROR: Server not healthy: {error}")
        run_cleanup()
        return False

    # Warmup
    print("Warming up...")
    await warmup_servers()
    await asyncio.sleep(5)

    # Run single test point
    t1_config = T1_CONFIGS["large"]
    t2_config = T2_CONFIGS["very_long_gen"]

    print(f"\nRunning test point: QPS={qps}")
    result, server_healthy = await run_benchmark_point(
        config, workload, qps, t1_config, t2_config
    )

    # Test OOM detection functions
    category = categorize_failure(result, server_healthy)
    is_oom, oom_reason = is_oom_failure(result, server_healthy)

    print(f"\n✓ Test completed:")
    print(f"  Success rate: {result.success_rate:.1f}%")
    print(f"  Category: {category}")
    print(f"  Is OOM: {is_oom}")
    print(f"  T1 Avg TTFT: {result.turn1.avg_ttft_ms:.0f}ms")
    print(f"  Throughput: {result.throughput['requests_per_sec']:.2f} req/s")

    # Cleanup
    run_cleanup()
    await asyncio.sleep(5)

    print("\n✓ high_qps_boundary_test.py validation PASSED")
    return True

async def validate_extreme_context():
    """Validate extreme_context_test.py - single test point."""
    print("\n" + "="*80)
    print("Validating extreme_context_test.py")
    print("="*80)
    print("Testing: 2P_2pD, context=4096, QPS=20")

    config = "2P_2pD"
    qps = 20
    context_size = 4096

    # Test helper functions
    t1_config, t2_config, workload_name = generate_extreme_workload(context_size)
    mem_estimate = estimate_memory_per_request(context_size)

    print(f"\nWorkload: {workload_name}")
    print(f"  T1: {t1_config}")
    print(f"  T2: {t2_config}")
    print(f"  Memory estimate: {mem_estimate:.2f} GB/req")

    # Start server
    print("\nStarting servers...")
    run_cleanup()
    await asyncio.sleep(8)

    if not start_config(config):
        print(f"ERROR: Failed to start {config}")
        return False

    await asyncio.sleep(15)

    # Health check
    healthy, error = await check_server_health()
    if not healthy:
        print(f"ERROR: Server not healthy: {error}")
        run_cleanup()
        return False

    # Warmup
    print("Warming up...")
    await warmup_servers()
    await asyncio.sleep(5)

    # Run single test point
    print(f"\nRunning test point: context={context_size}, QPS={qps}")
    result, server_healthy = await run_benchmark_point(
        config, workload_name, qps, t1_config, t2_config
    )

    print(f"\n✓ Test completed:")
    print(f"  Success rate: {result.success_rate:.1f}%")
    print(f"  T1 Avg TTFT: {result.turn1.avg_ttft_ms:.0f}ms")
    print(f"  T1 Avg E2E: {result.turn1.avg_e2e_ms:.0f}ms")
    print(f"  Throughput: {result.throughput['requests_per_sec']:.2f} req/s")

    # Cleanup
    run_cleanup()
    await asyncio.sleep(5)

    print("\n✓ extreme_context_test.py validation PASSED")
    return True

async def main():
    print("\n" + "#"*80)
    print("# Quick Validation of New Test Scripts")
    print("#"*80)

    # Validate high_qps_boundary_test.py
    success1 = await validate_high_qps()

    # Validate extreme_context_test.py
    success2 = await validate_extreme_context()

    print("\n" + "="*80)
    if success1 and success2:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print("✗ SOME VALIDATIONS FAILED")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
