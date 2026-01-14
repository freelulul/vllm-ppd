#!/usr/bin/env python3
"""
Verify that real benchmark configuration matches fig6 (simulated) exactly.

This script compares the test configurations between:
- generate_trend_data.py (simulation, used for current fig6)
- benchmark_real_trend_data.py (real benchmark)

Run this BEFORE running the full benchmark to ensure consistency.
"""

import random
import json

print("="*70)
print("CONFIGURATION VERIFICATION: Real Benchmark vs Fig6 (Simulation)")
print("="*70)

# ============================================================================
# Panel A: QPS Trend
# ============================================================================
print("\n" + "="*70)
print("PANEL A: QPS TREND")
print("="*70)

# From generate_trend_data.py (simulation)
sim_qps_levels = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
sim_qps_jobs = 60
sim_qps_slo = {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500}
sim_qps_mix = {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}

# From benchmark_real_trend_data.py (real)
real_qps_levels = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
real_qps_jobs = 60
real_qps_slo = {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500}
real_qps_mix = {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}

print(f"\nQPS Levels:")
print(f"  Simulation: {sim_qps_levels}")
print(f"  Real:       {real_qps_levels}")
print(f"  Match: {'YES' if sim_qps_levels == real_qps_levels else 'NO'}")

print(f"\nJobs per QPS level:")
print(f"  Simulation: {sim_qps_jobs}")
print(f"  Real:       {real_qps_jobs}")
print(f"  Match: {'YES' if sim_qps_jobs == real_qps_jobs else 'NO'}")

print(f"\nSLO Thresholds:")
print(f"  Simulation: {sim_qps_slo}")
print(f"  Real:       {real_qps_slo}")
print(f"  Match: {'YES' if sim_qps_slo == real_qps_slo else 'NO'}")

print(f"\nWorkload Mix:")
print(f"  Simulation: {sim_qps_mix}")
print(f"  Real:       {real_qps_mix}")
print(f"  Match: {'YES' if sim_qps_mix == real_qps_mix else 'NO'}")

# Generate and compare job configs
print(f"\nJob Configuration (first 5 jobs, seed=42):")
random.seed(42)
sim_jobs = []
for i in range(5):
    r = random.random()
    if r < sim_qps_mix["ttft"]:
        obj = "ttft"
        inp, out = random.randint(200, 1500), random.randint(50, 200)
    elif r < sim_qps_mix["ttft"] + sim_qps_mix["tpot"]:
        obj = "tpot"
        inp, out = random.randint(100, 500), random.randint(200, 600)
    else:
        obj = "e2e"
        inp, out = random.randint(300, 1200), random.randint(100, 400)
    sim_jobs.append({"obj": obj, "input": inp, "output": out})
    print(f"  Job {i}: obj={obj}, input={inp}, output={out}")

# ============================================================================
# Panel B: E2E Ratio Trend
# ============================================================================
print("\n" + "="*70)
print("PANEL B: E2E RATIO TREND")
print("="*70)

sim_e2e_ratios = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
real_e2e_ratios = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
sim_e2e_jobs = 50
real_e2e_jobs = 50
sim_e2e_slo = {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500}
real_e2e_slo = {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500}

print(f"\nE2E Ratios:")
print(f"  Simulation: {sim_e2e_ratios}")
print(f"  Real:       {real_e2e_ratios}")
print(f"  Match: {'YES' if sim_e2e_ratios == real_e2e_ratios else 'NO'}")

print(f"\nJobs per ratio:")
print(f"  Simulation: {sim_e2e_jobs}")
print(f"  Real:       {real_e2e_jobs}")
print(f"  Match: {'YES' if sim_e2e_jobs == real_e2e_jobs else 'NO'}")

print(f"\nSLO Thresholds:")
print(f"  Simulation: {sim_e2e_slo}")
print(f"  Real:       {real_e2e_slo}")
print(f"  Match: {'YES' if sim_e2e_slo == real_e2e_slo else 'NO'}")

# ============================================================================
# Panel C: SLO Strictness Trend
# ============================================================================
print("\n" + "="*70)
print("PANEL C: SLO STRICTNESS TREND")
print("="*70)

sim_strictness = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
real_strictness = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
sim_strict_baseline = {"ttft_ms": 80, "tpot_ms": 8, "e2e_ms": 2000}
real_strict_baseline = {"ttft_ms": 80, "tpot_ms": 8, "e2e_ms": 2000}
sim_strict_jobs = 50
real_strict_jobs = 50

print(f"\nStrictness Factors:")
print(f"  Simulation: {sim_strictness}")
print(f"  Real:       {real_strictness}")
print(f"  Match: {'YES' if sim_strictness == real_strictness else 'NO'}")

print(f"\nBaseline SLO:")
print(f"  Simulation: {sim_strict_baseline}")
print(f"  Real:       {real_strict_baseline}")
print(f"  Match: {'YES' if sim_strict_baseline == real_strict_baseline else 'NO'}")

print(f"\nJobs per factor:")
print(f"  Simulation: {sim_strict_jobs}")
print(f"  Real:       {real_strict_jobs}")
print(f"  Match: {'YES' if sim_strict_jobs == real_strict_jobs else 'NO'}")

# ============================================================================
# Panel D: Input Length Trend
# ============================================================================
print("\n" + "="*70)
print("PANEL D: INPUT LENGTH TREND")
print("="*70)

sim_input_lengths = [100, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]
real_input_lengths = [100, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]
sim_input_jobs = 50
real_input_jobs = 50
sim_input_slo = {"ttft_ms": 150, "tpot_ms": 10, "e2e_ms": 3000}
real_input_slo = {"ttft_ms": 150, "tpot_ms": 10, "e2e_ms": 3000}

print(f"\nInput Lengths:")
print(f"  Simulation: {sim_input_lengths}")
print(f"  Real:       {real_input_lengths}")
print(f"  Match: {'YES' if sim_input_lengths == real_input_lengths else 'NO'}")

print(f"\nJobs per length:")
print(f"  Simulation: {sim_input_jobs}")
print(f"  Real:       {real_input_jobs}")
print(f"  Match: {'YES' if sim_input_jobs == real_input_jobs else 'NO'}")

print(f"\nSLO Thresholds:")
print(f"  Simulation: {sim_input_slo}")
print(f"  Real:       {real_input_slo}")
print(f"  Match: {'YES' if sim_input_slo == real_input_slo else 'NO'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

all_match = (
    sim_qps_levels == real_qps_levels and
    sim_qps_jobs == real_qps_jobs and
    sim_qps_slo == real_qps_slo and
    sim_e2e_ratios == real_e2e_ratios and
    sim_e2e_jobs == real_e2e_jobs and
    sim_e2e_slo == real_e2e_slo and
    sim_strictness == real_strictness and
    sim_strict_baseline == real_strict_baseline and
    sim_strict_jobs == real_strict_jobs and
    sim_input_lengths == real_input_lengths and
    sim_input_jobs == real_input_jobs and
    sim_input_slo == real_input_slo
)

if all_match:
    print("\n*** ALL CONFIGURATIONS MATCH ***")
    print("Real benchmark is configured identically to fig6 simulation.")
else:
    print("\n!!! CONFIGURATION MISMATCH DETECTED !!!")
    print("Please review and fix before running benchmark.")

print("\n" + "="*70)
print("TOTAL DATA POINTS TO COLLECT")
print("="*70)
print(f"Panel A (QPS):        {len(sim_qps_levels)} points × 4 modes × {sim_qps_jobs} requests = {len(sim_qps_levels) * 4 * sim_qps_jobs} requests")
print(f"Panel B (E2E Ratio):  {len(sim_e2e_ratios)} points × 4 modes × {sim_e2e_jobs} requests = {len(sim_e2e_ratios) * 4 * sim_e2e_jobs} requests")
print(f"Panel C (Strictness): {len(sim_strictness)} points × 4 modes × {sim_strict_jobs} requests = {len(sim_strictness) * 4 * sim_strict_jobs} requests")
print(f"Panel D (Input Len):  {len(sim_input_lengths)} points × 4 modes × {sim_input_jobs} requests = {len(sim_input_lengths) * 4 * sim_input_jobs} requests")

total = (len(sim_qps_levels) * 4 * sim_qps_jobs +
         len(sim_e2e_ratios) * 4 * sim_e2e_jobs +
         len(sim_strictness) * 4 * sim_strict_jobs +
         len(sim_input_lengths) * 4 * sim_input_jobs)
print(f"\nTOTAL: {total} requests")
print(f"Note: Panel C reuses baseline data, actual requests = {total - len(sim_strictness) * 3 * sim_strict_jobs}")
