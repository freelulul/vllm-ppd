#!/usr/bin/env python3
"""
Quick validation test - one point from each panel with short duration.
Tests all 4 modes at each point to verify everything works.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import asyncio
from datetime import datetime

# Import from main benchmark
from scripts.tests.fig6_benchmark import (
    PANEL_CONFIGS, run_panel, stop_all_servers
)

# Quick test config: one point per panel, 10s duration
# Panel A: Use Pure_TPOT to verify pure objective scenarios (not Mixed where optimizer should win)
QUICK_TEST_POINTS = {
    "A": "Pure_TPOT",        # Pure objective to verify mode-specific performance
    "B": 8,                  # Higher QPS to see load effects
    "C": 512,                # Medium input length
    "D": 128,                # Medium output length
    "E": 4,                  # 4 turns
    "F": 1,                  # 1:1 I/O ratio
}

async def quick_validation():
    """Run quick validation on each panel"""

    print("="*70)
    print("QUICK PANEL VALIDATION")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing one point from each panel with 10s duration")
    print()

    modes = ["pd", "ppd", "replica", "optimizer"]
    duration = 10  # Short duration for quick test

    results = {}

    for panel, test_point in QUICK_TEST_POINTS.items():
        cfg = PANEL_CONFIGS[panel]

        # Temporarily modify panel config to only test one point
        original_values = cfg["x_values"]
        cfg["x_values"] = [test_point]

        print(f"\n{'='*70}")
        print(f"Testing Panel {panel}: {cfg['name']}")
        print(f"Test point: {cfg['x_axis']}={test_point}")
        print(f"{'='*70}")

        try:
            result = await run_panel(panel, modes, duration)
            results[panel] = result

            # Print quick summary
            if result["data_points"]:
                dp = result["data_points"][0]
                print(f"\n  Results for {cfg['x_axis']}={test_point}:")
                for mode in modes:
                    if mode in dp["modes"]:
                        slo = dp["modes"][mode]["slo_attainment"]["overall"]
                        latency = dp["modes"][mode]["latency_stats"]
                        ttft = latency.get("avg_ttft_ms", 0)
                        tpot = latency.get("avg_tpot_ms", 0)
                        e2e = latency.get("avg_e2e_ms", 0)
                        print(f"    {mode:10s}: SLO={slo:5.1f}% | TTFT={ttft:7.1f}ms | TPOT={tpot:5.1f}ms | E2E={e2e:7.1f}ms")

                best_e2e = dp.get('best_e2e', 0)
                print(f"  Best mode: {dp.get('best_mode', 'N/A')} (E2E={best_e2e:.0f}ms)")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[panel] = {"error": str(e)}

        # Restore original values
        cfg["x_values"] = original_values

    # Final cleanup
    stop_all_servers()

    # Save results
    output_file = f"results/quick_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*85)
    print("VALIDATION SUMMARY")
    print("="*85)
    print(f"{'Panel':<8} {'Test Point':<15} {'Best Mode':<12} {'Best SLO':<10} {'Best E2E':<12} {'Opt SLO':<10}")
    print("-"*85)

    for panel, result in results.items():
        if "error" in result:
            print(f"{panel:<8} {'ERROR':<15} {'-':<12} {'-':<10} {'-':<12} {'-':<10}")
        elif result.get("data_points"):
            dp = result["data_points"][0]
            test_point = dp["x_value"]
            best = dp.get("best_mode", "N/A")
            best_slo = dp.get("best_slo", 0)
            best_e2e = dp.get("best_e2e", 0)
            opt_slo = dp["modes"].get("optimizer", {}).get("slo_attainment", {}).get("overall", 0)
            print(f"{panel:<8} {str(test_point):<15} {best:<12} {best_slo:>6.1f}%   {best_e2e:>8.0f}ms   {opt_slo:>6.1f}%")

    print("="*85)

    return results


if __name__ == "__main__":
    asyncio.run(quick_validation())
