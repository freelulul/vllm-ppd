#!/usr/bin/env python3
"""
Test Panel B at QPS=16 specifically to see if Optimizer can beat Replica at high load.
"""

import os
import sys
sys.path.insert(0, "/net/projects2/ds3lab/zongzel/vllm/ppd")

import json
import asyncio
from datetime import datetime

from scripts.tests.fig6_benchmark import (
    PANEL_CONFIGS, run_panel, stop_all_servers
)

async def test_panel_B_qps16():
    """Test Panel B at QPS=16 only"""

    print("="*80)
    print("PANEL B QPS=16 VALIDATION TEST")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    print("  - QPS: 8")
    print("  - Duration: 10s")
    print("  - Objective: Mixed (33% TTFT, 34% TPOT, 33% E2E)")
    print("  - Turns: 2 (T1: 512→512, T2: 512→64) - M_d Big-Paste workload")
    print("  - Expected: PD best for TPOT, PPD good for T2_TTFT, Replica best for T1_TTFT")
    print()
    print("Hypothesis: At high QPS, Optimizer should perform better than Replica")
    print("            due to intelligent routing reducing contention")
    print()

    modes = ["pd", "ppd", "replica", "optimizer"]
    duration = 10  # 10s per data point

    # Temporarily modify PANEL_CONFIGS to only test QPS=16
    panel_b_cfg = PANEL_CONFIGS["B"]
    original_values = panel_b_cfg["x_values"]
    panel_b_cfg["x_values"] = [8]  # M_d workload test

    results = {}

    try:
        result_b = await run_panel("B", modes, duration)
        results["B"] = result_b

        # Print detailed results
        print("\n" + "="*80)
        print("PANEL B QPS=16 DETAILED RESULTS")
        print("="*80)

        for dp in result_b.get("data_points", []):
            qps = dp["x_value"]
            print(f"\nQPS = {qps}")
            print("-"*70)

            # Print header
            print(f"{'Mode':<12} {'SLO %':<10} {'TTFT':<25} {'TPOT':<25} {'E2E':<25}")
            print(f"{'':12} {'':10} {'met/total':<12} {'met/total':<12} {'met/total':<12}")
            print("-"*70)

            for mode in modes:
                if mode in dp["modes"]:
                    data = dp["modes"][mode]
                    slo = data["slo_attainment"]["overall"]

                    # Get by_objective counts
                    by_obj = data["slo_attainment"].get("by_objective", {})
                    ttft_met = by_obj.get("ttft", {}).get("met", 0)
                    ttft_total = by_obj.get("ttft", {}).get("total", 0)
                    tpot_met = by_obj.get("tpot", {}).get("met", 0)
                    tpot_total = by_obj.get("tpot", {}).get("total", 0)
                    e2e_met = by_obj.get("e2e", {}).get("met", 0)
                    e2e_total = by_obj.get("e2e", {}).get("total", 0)

                    ttft_str = f"{ttft_met}/{ttft_total}" if ttft_total > 0 else "N/A"
                    tpot_str = f"{tpot_met}/{tpot_total}" if tpot_total > 0 else "N/A"
                    e2e_str = f"{e2e_met}/{e2e_total}" if e2e_total > 0 else "N/A"

                    print(f"{mode:<12} {slo:>8.1f}%  {ttft_str:<12} {tpot_str:<12} {e2e_str:<12}")

            print()
            print("Latency Statistics (ms):")
            print(f"{'Mode':<12} {'TTFT avg':<12} {'TTFT p50':<12} {'TTFT p99':<12} {'TPOT avg':<10} {'E2E avg':<12} {'E2E p50':<12}")
            print("-"*90)

            for mode in modes:
                if mode in dp["modes"]:
                    stats = dp["modes"][mode]["latency_stats"]
                    print(f"{mode:<12} "
                          f"{stats['avg_ttft_ms']:>10.1f}  "
                          f"{stats['p50_ttft_ms']:>10.1f}  "
                          f"{stats['p99_ttft_ms']:>10.1f}  "
                          f"{stats['avg_tpot_ms']:>8.2f}  "
                          f"{stats['avg_e2e_ms']:>10.1f}  "
                          f"{stats['p50_e2e_ms']:>10.1f}")

            print()
            print("Optimizer Routing:")
            if "optimizer" in dp["modes"]:
                routing = dp["modes"]["optimizer"].get("routing_stats", {})
                total = sum(routing.values())
                for route, count in sorted(routing.items(), key=lambda x: -x[1]):
                    pct = count / total * 100 if total > 0 else 0
                    print(f"  {route}: {count} ({pct:.1f}%)")

            print()
            print(f"WINNER: {dp.get('best_mode', 'N/A')} with {dp.get('best_slo', 0):.1f}% SLO attainment")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["B"] = {"error": str(e)}

    # Restore original config
    panel_b_cfg["x_values"] = original_values

    # Final cleanup
    stop_all_servers()

    # Save results
    output_file = f"/net/projects2/ds3lab/zongzel/vllm/ppd/results/panel_B_qps16_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    return results

if __name__ == "__main__":
    asyncio.run(test_panel_B_qps16())