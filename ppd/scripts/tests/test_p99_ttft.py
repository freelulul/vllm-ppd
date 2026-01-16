#!/usr/bin/env python3
"""
P99_TTFT Objective Test - Burst Mode

Test with 100% P99_TTFT optimization objective to analyze:
1. How each mode performs for P99 TTFT optimization
2. How the optimizer routes P99_TTFT requests
3. Whether routing decisions improve SLO attainment

Configuration:
- QPS: 8
- Duration: 10s
- Turn 1: 512→512
- Turn 2: 512→64 (Big-Paste pattern)
- Objective: 100% P99_TTFT
- SLO: TTFT≤1000ms, TPOT≤11ms, E2E≤2000ms
- Mode: Burst (T2 sent simultaneously)
"""

import os
import sys
sys.path.insert(0, "/net/projects2/ds3lab/zongzel/vllm/ppd")

import json
import asyncio
from datetime import datetime

from scripts.tests.fig6_benchmark import (
    run_benchmark_for_mode, stop_all_servers, start_servers, warmup_mode,
    SLO_THRESHOLDS, ModeResult
)


async def run_p99_ttft_test():
    """Run P99_TTFT objective test"""

    print("="*80)
    print("P99_TTFT OBJECTIVE TEST (Burst Mode)")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    print("  - QPS: 8")
    print("  - Duration: 10s")
    print("  - Turn 1: 512→512 (standard multi-turn)")
    print("  - Turn 2: 512→64 (Big-Paste: long input, short output)")
    print("  - Objective: 100% P99_TTFT (stricter TTFT SLO)")
    print(f"  - SLO: TTFT≤{SLO_THRESHOLDS['ttft_ms']}ms, TPOT≤{SLO_THRESHOLDS['tpot_ms']}ms, E2E≤{SLO_THRESHOLDS['e2e_ms']}ms")
    print("  - Mode: Burst (all T2 sent simultaneously after T1)")
    print()
    print("Expected behavior based on selector rules:")
    print("  - P99_TTFT objective routes to:")
    print("    - Tiny T2 output (≤64): Replica")
    print("    - Small context (<600): Replica")
    print("    - Default: PPD (65.7% win rate in original benchmark)")
    print()

    modes = ["pd", "ppd", "replica", "optimizer"]
    qps = 8
    duration = 10

    # M_d workload: T1: 512→512, T2: 512→64 (Big-Paste)
    turn_configs = [
        {"input": 512, "output": 512},  # T1
        {"input": 512, "output": 64},   # T2 Big-Paste
    ]

    # 100% P99_TTFT objective
    objective_dist = {"p99_ttft": 1.0}

    results = {
        "test_name": "P99_TTFT Objective Test (Burst Mode)",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "qps": qps,
            "duration_sec": duration,
            "turn_configs": turn_configs,
            "objective_dist": objective_dist,
            "slo_thresholds": SLO_THRESHOLDS,
            "mode": "burst"
        },
        "modes": {}
    }

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing mode: {mode.upper()}")
        print(f"{'='*60}")

        # Stop all servers and start fresh
        print("Stopping all servers...")
        stop_all_servers()

        print(f"Starting {mode} servers...")
        start_servers(mode)

        print("Waiting 100s for servers to initialize...")
        await asyncio.sleep(100)

        # Warmup
        print(f"Warming up {mode}...")
        await warmup_mode(mode)

        print(f"Running benchmark for {duration}s at QPS={qps}...")
        mode_result = await run_benchmark_for_mode(
            mode=mode,
            qps=qps,
            duration_sec=duration,
            turn_configs=turn_configs,
            objective_dist=objective_dist,
        )

        # Compute stats with T2-only SLO
        slo = mode_result.compute_slo_attainment(turn2_only=True)
        latency = mode_result.compute_latency_stats()

        # Print results
        print(f"\n  Results for {mode.upper()}:")
        print(f"  Conversations: {len(mode_result.conversations)}")
        print(f"  Total turns: {sum(len(c.turns) for c in mode_result.conversations)}")
        print()
        print(f"  SLO Attainment (T2 only): {slo['overall']:.1f}%")
        print(f"    By objective: {slo.get('by_objective', {})}")
        print()

        # T2 metrics (main)
        t2 = latency.get('turn2_metrics', {})
        print(f"  T2 Metrics (main):")
        print(f"    T2 Avg TTFT: {t2.get('avg_ttft_ms', 0):.1f}ms")
        print(f"    T2 P50 TTFT: {t2.get('p50_ttft_ms', 0):.1f}ms")
        print(f"    T2 P99 TTFT: {t2.get('p99_ttft_ms', 0):.1f}ms")
        print(f"    T2 Avg TPOT: {t2.get('avg_tpot_ms', 0):.2f}ms")
        print(f"    T2 Avg E2E: {t2.get('avg_e2e_ms', 0):.1f}ms")
        print(f"    T2 Count: {t2.get('count', 0)}")

        # T1 metrics (reference)
        t1 = latency.get('turn1_metrics', {})
        print(f"\n  T1 Metrics (reference):")
        print(f"    T1 Avg TTFT: {t1.get('avg_ttft_ms', 0):.1f}ms")
        print(f"    T1 P99 TTFT: {t1.get('p99_ttft_ms', 0):.1f}ms")
        print(f"    T1 Avg TPOT: {t1.get('avg_tpot_ms', 0):.2f}ms")
        print(f"    T1 Count: {t1.get('count', 0)}")

        # Routing stats for optimizer
        if mode == "optimizer" and mode_result.routing_stats:
            print(f"\n  Routing Stats:")
            for route, count in sorted(mode_result.routing_stats.items()):
                print(f"    {route}: {count}")

        # Save results
        results["modes"][mode] = {
            "slo_attainment": slo,
            "latency_stats": latency,
            "routing_stats": mode_result.routing_stats,
            "num_conversations": len(mode_result.conversations),
            "num_turns": sum(len(c.turns) for c in mode_result.conversations)
        }

    # Stop servers
    stop_all_servers()

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON (T2 Metrics Only)")
    print("="*80)
    print()
    print(f"{'Mode':<12} {'SLO %':<10} {'Avg TTFT':<12} {'P99 TTFT':<12} {'Avg TPOT':<12} {'Avg E2E':<12}")
    print("-"*70)

    best_mode = None
    best_slo = -1

    for mode in modes:
        data = results["modes"][mode]
        slo_pct = data["slo_attainment"]["overall"]
        t2 = data["latency_stats"].get("turn2_metrics", {})

        winner = " ✓" if slo_pct > best_slo else ""
        if slo_pct > best_slo:
            best_slo = slo_pct
            best_mode = mode

        print(f"{mode:<12} {slo_pct:>7.1f}%  "
              f"{t2.get('avg_ttft_ms', 0):>10.1f}  "
              f"{t2.get('p99_ttft_ms', 0):>10.1f}  "
              f"{t2.get('avg_tpot_ms', 0):>10.2f}  "
              f"{t2.get('avg_e2e_ms', 0):>10.1f}{winner}")

    print()
    print(f"Best mode: {best_mode.upper()} with {best_slo:.1f}% SLO attainment (T2 only)")

    # Routing analysis for optimizer
    print("\n" + "="*80)
    print("OPTIMIZER ROUTING ANALYSIS")
    print("="*80)
    opt_data = results["modes"].get("optimizer", {})
    routing = opt_data.get("routing_stats", {})
    total_routes = sum(routing.values()) if routing else 0
    if routing:
        print(f"\nTotal requests routed: {total_routes}")
        print("Routing distribution:")
        for route, count in sorted(routing.items()):
            pct = (count / total_routes * 100) if total_routes > 0 else 0
            print(f"  {route}: {count} ({pct:.1f}%)")

        print("\nExpected routing for P99_TTFT objective:")
        print("  - Tiny T2 output (≤64): → Replica (benchmark: wins 14/27 for type 'a')")
        print("  - Our T2: 512→64, so output=64 ≤ 64 threshold")
        print("  - Expected: All/most requests should route to Replica")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/net/projects2/ds3lab/zongzel/vllm/ppd/results/p99_ttft_test_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    asyncio.run(run_p99_ttft_test())
