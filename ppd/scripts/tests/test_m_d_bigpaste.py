#!/usr/bin/env python3
"""
M_d Big-Paste Benchmark Test

Re-run the M_d Big-Paste workload test with updated metrics collection:
- T1 and T2 metrics collected separately (matches original benchmark)
- SLO computed on T2 only (PPD's advantage is in multi-turn scenarios)
- TPOT calculated using (e2e - ttft) / (tokens - 1) formula

Configuration matches the first test in mixed_workload_test_20260115.md:
- QPS: 8
- Duration: 10s
- Turn 1: 512→512
- Turn 2: 512→64 (Big-Paste pattern)
- Objective: Mixed (33% P99_TTFT, 34% TPOT, 33% E2E)
- SLO: TTFT≤1000ms, TPOT≤11ms, E2E≤2000ms
"""

import os
import sys
sys.path.insert(0, "/net/projects2/ds3lab/zongzel/vllm/ppd")

import json
import asyncio
from datetime import datetime

from scripts.tests.fig6_benchmark import (
    run_benchmark_for_mode, stop_all_servers, start_servers, warmup_mode,
    SLO_THRESHOLDS
)


async def run_m_d_bigpaste_test():
    """Run M_d Big-Paste workload test"""

    print("="*80)
    print("M_d BIG-PASTE WORKLOAD TEST (T1/T2 Separated Metrics)")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    print("  - QPS: 8")
    print("  - Duration: 10s")
    print("  - Turn 1: 512→512 (standard multi-turn)")
    print("  - Turn 2: 512→64 (Big-Paste: long input, short output)")
    print("  - Objective: Mixed (33% P99_TTFT, 34% TPOT, 33% E2E)")
    print(f"  - SLO: TTFT≤{SLO_THRESHOLDS['ttft_ms']}ms, TPOT≤{SLO_THRESHOLDS['tpot_ms']}ms, E2E≤{SLO_THRESHOLDS['e2e_ms']}ms")
    print()
    print("Key changes from previous test:")
    print("  - Metrics now collected separately for T1 and T2")
    print("  - SLO computed on T2 only (matches original benchmark)")
    print("  - TPOT uses (e2e-ttft)/(tokens-1) formula")
    print()

    modes = ["pd", "ppd", "replica", "optimizer"]
    qps = 8
    duration = 10

    # M_d workload: T1: 512→512, T2: 512→64
    turn_configs = [
        {"input": 512, "output": 512},  # T1
        {"input": 512, "output": 64},   # T2 Big-Paste
    ]

    objective_dist = {"p99_ttft": 0.33, "tpot": 0.34, "e2e": 0.33}

    results = {
        "test_name": "M_d Big-Paste Workload (T1/T2 Separated)",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "qps": qps,
            "duration_sec": duration,
            "turn_configs": turn_configs,
            "objective_dist": objective_dist,
            "slo_thresholds": SLO_THRESHOLDS,
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
        if not start_servers(mode):
            print(f"  ERROR: Failed to start {mode} servers, skipping...")
            results["modes"][mode] = {"error": "Failed to start servers"}
            continue

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
        print(f"  T2 Metrics (main - matches original benchmark):")
        print(f"    T2 Avg TTFT: {t2.get('avg_ttft_ms', 0):.1f}ms")
        print(f"    T2 P99 TTFT: {t2.get('p99_ttft_ms', 0):.1f}ms")
        print(f"    T2 Avg TPOT: {t2.get('avg_tpot_ms', 0):.2f}ms")
        print(f"    T2 P99 TPOT: {t2.get('p99_tpot_ms', 0):.2f}ms")
        print(f"    T2 Avg E2E: {t2.get('avg_e2e_ms', 0):.1f}ms")
        print(f"    T2 Count: {t2.get('count', 0)}")

        # T1 metrics (reference)
        t1 = latency.get('turn1_metrics', {})
        print(f"\n  T1 Metrics (reference):")
        print(f"    T1 Avg TTFT: {t1.get('avg_ttft_ms', 0):.1f}ms")
        print(f"    T1 Avg TPOT: {t1.get('avg_tpot_ms', 0):.2f}ms")
        print(f"    T1 Avg E2E: {t1.get('avg_e2e_ms', 0):.1f}ms")
        print(f"    T1 Count: {t1.get('count', 0)}")

        # Routing stats for optimizer
        if mode == "optimizer" and mode_result.routing_stats:
            print(f"\n  Routing Stats:")
            for route, count in mode_result.routing_stats.items():
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
    print(f"{'Mode':<12} {'SLO %':<10} {'T2 TTFT':<12} {'T2 TPOT':<12} {'T2 E2E':<12} {'T1 TTFT (ref)':<14}")
    print("-"*72)

    best_mode = None
    best_slo = -1

    for mode in modes:
        data = results["modes"][mode]
        slo_pct = data["slo_attainment"]["overall"]
        t2 = data["latency_stats"].get("turn2_metrics", {})
        t1 = data["latency_stats"].get("turn1_metrics", {})

        winner = " Winner" if slo_pct > best_slo else ""
        if slo_pct > best_slo:
            best_slo = slo_pct
            best_mode = mode

        print(f"{mode:<12} {slo_pct:>7.1f}%  "
              f"{t2.get('avg_ttft_ms', 0):>10.1f}  "
              f"{t2.get('avg_tpot_ms', 0):>10.2f}  "
              f"{t2.get('avg_e2e_ms', 0):>10.1f}  "
              f"{t1.get('avg_ttft_ms', 0):>12.1f}{winner}")

    print()
    print(f"Best mode: {best_mode.upper()} with {best_slo:.1f}% SLO attainment (T2 only)")

    # Original benchmark comparison
    print("\n" + "="*80)
    print("COMPARISON WITH ORIGINAL BENCHMARK (M_d @ QPS=4-6)")
    print("="*80)
    print("From merged_results.txt (T2 metrics only):")
    print()
    print("  M_d @ QPS=4:")
    print("    PD:      T2_TTFT=893.7ms,  T2_TPOT=10.50ms")
    print("    PPD:     T2_TTFT=432.5ms,  T2_TPOT=12.22ms")
    print("    Replica: T2_TTFT=396.8ms,  T2_TPOT=10.97ms")
    print()
    print("  M_d @ QPS=6:")
    print("    PD:      T2_TTFT=1058.0ms, T2_TPOT=10.55ms")
    print("    PPD:     T2_TTFT=567.3ms,  T2_TPOT=14.47ms")
    print("    Replica: T2_TTFT=562.1ms,  T2_TPOT=12.57ms")
    print()
    print("  Key insights from original benchmark:")
    print("    - PD has best T2_TPOT at QPS=4-6 (prefill isolation advantage)")
    print("    - PPD has better T2_TTFT than PD (KV cache reuse)")
    print("    - Replica competitive on both metrics")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/net/projects2/ds3lab/zongzel/vllm/ppd/results/m_d_bigpaste_test_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    asyncio.run(run_m_d_bigpaste_test())
