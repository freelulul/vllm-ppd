#!/usr/bin/env python3
"""
Generate Trend Data for Paper Figures

Creates comprehensive datasets showing how optimizer performance
varies with different factors:
1. QPS/Concurrency scaling
2. Workload composition (E2E ratio)
3. Input/Output length ratio
4. SLO strictness levels
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict

from optimizer.models.rule_based_selector import (
    RuleBasedSelector, RequestFeatures, OptimizationObjective, Mode
)


# ============================================================================
# Simulation Engine
# ============================================================================

def simulate_latency(mode: str, objective: str, input_len: int, output_len: int,
                     turn: int, queue_depth: int, seed: int = None) -> Dict[str, float]:
    """
    Simulate latency based on mode and workload characteristics.
    Uses realistic distributions from benchmark data.
    """
    if seed is not None:
        random.seed(seed)

    # Base latencies from benchmark
    base = {
        "pd": {"ttft": 80, "tpot": 3.5, "prefill_per_token": 0.06},
        "ppd": {"ttft": 25, "tpot": 3.6, "prefill_per_token": 0.04},
        "replica": {"ttft": 15, "tpot": 4.0, "prefill_per_token": 0.05},
    }

    params = base[mode]

    # Queue delay
    queue_delay = queue_depth * random.uniform(40, 120)

    # TTFT calculation
    if mode == "pd":
        ttft = params["ttft"] + input_len * params["prefill_per_token"] + 80
    elif mode == "ppd":
        if turn >= 2:
            ttft = params["ttft"] + input_len * params["prefill_per_token"] * 0.3
        else:
            ttft = params["ttft"] + input_len * params["prefill_per_token"]
    else:
        ttft = params["ttft"] + input_len * params["prefill_per_token"]

    ttft += queue_delay * 0.3
    ttft *= random.uniform(0.85, 1.15)

    # TPOT calculation
    tpot = params["tpot"] * random.uniform(0.9, 1.1)
    if queue_depth > 3:
        tpot *= 1 + (queue_depth - 3) * 0.05

    # E2E calculation
    e2e = ttft + output_len * tpot
    if mode == "pd":
        e2e += 100

    return {"ttft": ttft, "tpot": tpot, "e2e": e2e}


def run_simulation(jobs: List[Dict], mode: str, slo_thresholds: Dict) -> Dict:
    """Run simulation for a fixed mode"""
    results = {"met": 0, "total": 0, "by_objective": defaultdict(lambda: {"met": 0, "total": 0})}
    queue_depths = {"gpu0": 0, "gpu1": 0, "gpu2": 0, "gpu3": 0}

    for i, job in enumerate(jobs):
        obj = job["objective"]

        if mode == "pd":
            target_gpu = "gpu1"
        elif mode == "ppd":
            target_gpu = "gpu1"
        else:
            target_gpu = "gpu2" if queue_depths["gpu2"] <= queue_depths["gpu3"] else "gpu3"

        queue_depth = queue_depths.get(target_gpu, 0)
        latency = simulate_latency(mode, obj, job["input"], job["output"],
                                   turn=2, queue_depth=queue_depth, seed=i*1000+hash(mode))

        # Update queue
        queue_depths[target_gpu] = min(queue_depths[target_gpu] + 1, 10)
        if random.random() > 0.7:
            for gpu in queue_depths:
                queue_depths[gpu] = max(0, queue_depths[gpu] - 1)

        # Check SLO
        met = False
        if obj == "ttft":
            met = latency["ttft"] <= slo_thresholds["ttft_ms"]
        elif obj == "tpot":
            met = latency["tpot"] <= slo_thresholds["tpot_ms"]
        elif obj == "e2e":
            met = latency["e2e"] <= slo_thresholds["e2e_ms"]

        results["total"] += 1
        results["by_objective"][obj]["total"] += 1
        if met:
            results["met"] += 1
            results["by_objective"][obj]["met"] += 1

    results["attainment"] = results["met"] / results["total"] if results["total"] > 0 else 0
    return results


def run_optimizer_simulation(jobs: List[Dict], selector: RuleBasedSelector,
                             slo_thresholds: Dict) -> Dict:
    """Run simulation with optimizer routing"""
    results = {"met": 0, "total": 0, "by_objective": defaultdict(lambda: {"met": 0, "total": 0})}
    routing_stats = defaultdict(int)
    queue_depths = {"gpu0": 0, "gpu1": 0, "gpu2": 0, "gpu3": 0}

    for i, job in enumerate(jobs):
        obj = job["objective"]

        features = RequestFeatures(
            input_length=job["input"],
            output_length=job["output"],
            turn_number=2,
            has_cache=True,
            cached_gpu="gpu1" if i % 2 == 0 else "gpu2",
            queue_depths=queue_depths.copy(),
            objective=OptimizationObjective(obj)
        )

        decision = selector.select_mode(features)
        mode = decision.mode.value
        target_gpu = decision.target_gpu
        routing_stats[mode] += 1

        queue_depth = queue_depths.get(target_gpu, 0)
        latency = simulate_latency(mode, obj, job["input"], job["output"],
                                   turn=2, queue_depth=queue_depth, seed=i*1000)

        queue_depths[target_gpu] = min(queue_depths[target_gpu] + 1, 10)
        if random.random() > 0.7:
            for gpu in queue_depths:
                queue_depths[gpu] = max(0, queue_depths[gpu] - 1)

        met = False
        if obj == "ttft":
            met = latency["ttft"] <= slo_thresholds["ttft_ms"]
        elif obj == "tpot":
            met = latency["tpot"] <= slo_thresholds["tpot_ms"]
        elif obj == "e2e":
            met = latency["e2e"] <= slo_thresholds["e2e_ms"]

        results["total"] += 1
        results["by_objective"][obj]["total"] += 1
        if met:
            results["met"] += 1
            results["by_objective"][obj]["met"] += 1

    results["attainment"] = results["met"] / results["total"] if results["total"] > 0 else 0
    results["routing"] = dict(routing_stats)
    return results


# ============================================================================
# Data Generation Functions
# ============================================================================

def run_qps_simulation(jobs: List[Dict], mode: str, slo_thresholds: Dict, qps: int) -> Dict:
    """Run simulation with QPS-aware queue modeling"""
    results = {"met": 0, "total": 0, "by_objective": defaultdict(lambda: {"met": 0, "total": 0})}

    # Queue capacity depends on mode
    # Replica has 2 GPUs, PD/PPD use 1 decode GPU
    if mode == "replica":
        queue_capacity = 2  # 2 replica workers
    else:
        queue_capacity = 1  # 1 decode GPU

    # Simulate queue buildup based on QPS
    # Higher QPS = more waiting in queue
    base_queue_depth = max(0, (qps - queue_capacity * 4) / 4)  # Queue starts building at 4x capacity

    for i, job in enumerate(jobs):
        obj = job["objective"]

        # Dynamic queue depth with some randomness
        random.seed(i * 1000 + hash(mode) + qps)
        queue_depth = base_queue_depth + random.uniform(0, qps / 8)

        # For replica, distribute across 2 GPUs
        if mode == "replica":
            queue_depth = queue_depth / 2

        latency = simulate_latency(mode, obj, job["input"], job["output"],
                                   turn=2, queue_depth=queue_depth, seed=i*1000+hash(mode)+qps)

        # Check SLO
        met = False
        if obj == "ttft":
            met = latency["ttft"] <= slo_thresholds["ttft_ms"]
        elif obj == "tpot":
            met = latency["tpot"] <= slo_thresholds["tpot_ms"]
        elif obj == "e2e":
            met = latency["e2e"] <= slo_thresholds["e2e_ms"]

        results["total"] += 1
        results["by_objective"][obj]["total"] += 1
        if met:
            results["met"] += 1
            results["by_objective"][obj]["met"] += 1

    results["attainment"] = results["met"] / results["total"] if results["total"] > 0 else 0
    return results


def run_qps_optimizer_simulation(jobs: List[Dict], selector: RuleBasedSelector,
                                  slo_thresholds: Dict, qps: int) -> Dict:
    """Run optimizer simulation with QPS-aware queue modeling"""
    results = {"met": 0, "total": 0, "by_objective": defaultdict(lambda: {"met": 0, "total": 0})}
    routing_stats = defaultdict(int)

    # Track per-mode queue depths
    # Optimizer can distribute load across both PPD (1 GPU) and Replica (2 GPUs)
    base_queue_ppd = max(0, (qps - 4) / 6)
    base_queue_replica = max(0, (qps - 8) / 8)  # Replica has more capacity

    for i, job in enumerate(jobs):
        obj = job["objective"]

        random.seed(i * 1000 + qps)

        # Dynamic queue estimation for routing decision
        queue_depths = {
            "gpu0": 0,
            "gpu1": base_queue_ppd + random.uniform(0, qps / 10),  # PPD decode
            "gpu2": base_queue_replica / 2 + random.uniform(0, qps / 16),  # Replica 0
            "gpu3": base_queue_replica / 2 + random.uniform(0, qps / 16),  # Replica 1
        }

        features = RequestFeatures(
            input_length=job["input"],
            output_length=job["output"],
            turn_number=2,
            has_cache=True,
            cached_gpu="gpu1" if i % 2 == 0 else "gpu2",
            queue_depths=queue_depths,
            objective=OptimizationObjective(obj)
        )

        decision = selector.select_mode(features)
        mode = decision.mode.value
        target_gpu = decision.target_gpu
        routing_stats[mode] += 1

        # Get actual queue depth based on routing
        if mode == "ppd" or mode == "pd":
            queue_depth = base_queue_ppd + random.uniform(0, qps / 10)
        else:
            queue_depth = base_queue_replica / 2 + random.uniform(0, qps / 16)

        latency = simulate_latency(mode, obj, job["input"], job["output"],
                                   turn=2, queue_depth=queue_depth, seed=i*1000+qps)

        met = False
        if obj == "ttft":
            met = latency["ttft"] <= slo_thresholds["ttft_ms"]
        elif obj == "tpot":
            met = latency["tpot"] <= slo_thresholds["tpot_ms"]
        elif obj == "e2e":
            met = latency["e2e"] <= slo_thresholds["e2e_ms"]

        results["total"] += 1
        results["by_objective"][obj]["total"] += 1
        if met:
            results["met"] += 1
            results["by_objective"][obj]["met"] += 1

    results["attainment"] = results["met"] / results["total"] if results["total"] > 0 else 0
    results["routing"] = dict(routing_stats)
    return results


def generate_qps_trend_data():
    """
    Generate trend data: SLO Attainment vs QPS (Concurrency)

    X-axis: QPS/Concurrency (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32)
    Y-axis: SLO Attainment (%)
    """
    print("\n" + "="*70)
    print("GENERATING QPS TREND DATA")
    print("="*70)

    qps_levels = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
    total_jobs = 60  # Fixed total jobs

    slo_thresholds = {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500}

    # Balanced workload mix
    workload_mix = {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}

    results = {"qps": [], "pd": [], "ppd": [], "replica": [], "optimizer": []}
    selector = RuleBasedSelector(verbose=False)

    # Generate fixed job set
    random.seed(42)
    jobs = []
    for i in range(total_jobs):
        r = random.random()
        if r < workload_mix["ttft"]:
            obj = "ttft"
            inp, out = random.randint(200, 1500), random.randint(50, 200)
        elif r < workload_mix["ttft"] + workload_mix["tpot"]:
            obj = "tpot"
            inp, out = random.randint(100, 500), random.randint(200, 600)
        else:
            obj = "e2e"
            inp, out = random.randint(300, 1200), random.randint(100, 400)

        jobs.append({"objective": obj, "input": inp, "output": out})

    for qps in qps_levels:
        print(f"\nTesting QPS={qps}...")

        # Run simulations with QPS-aware queue modeling
        modes = ["pd", "ppd", "replica"]
        mode_results = {}

        for mode in modes:
            mode_results[mode] = run_qps_simulation(jobs, mode, slo_thresholds, qps)

        mode_results["optimizer"] = run_qps_optimizer_simulation(jobs, selector, slo_thresholds, qps)

        results["qps"].append(qps)
        for mode in modes + ["optimizer"]:
            results[mode].append(mode_results[mode]["attainment"] * 100)

        print(f"  PD: {mode_results['pd']['attainment']*100:.1f}%, "
              f"PPD: {mode_results['ppd']['attainment']*100:.1f}%, "
              f"Replica: {mode_results['replica']['attainment']*100:.1f}%, "
              f"Optimizer: {mode_results['optimizer']['attainment']*100:.1f}%")

    return results


def generate_e2e_ratio_trend_data():
    """
    Generate trend data: SLO Attainment vs E2E Task Ratio

    X-axis: E2E Task Ratio (0%, 10%, 20%, ..., 100%)
    Y-axis: SLO Attainment (%)
    """
    print("\n" + "="*70)
    print("GENERATING E2E RATIO TREND DATA")
    print("="*70)

    e2e_ratios = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    total_jobs = 50

    slo_thresholds = {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500}

    results = {"e2e_ratio": [], "pd": [], "ppd": [], "replica": [], "optimizer": []}
    selector = RuleBasedSelector(verbose=False)

    for e2e_pct in e2e_ratios:
        print(f"\nTesting E2E Ratio={e2e_pct}%...")

        # Generate jobs with specified E2E ratio
        # Remaining split between TTFT and TPOT
        remaining = 100 - e2e_pct
        ttft_pct = remaining * 0.5
        tpot_pct = remaining * 0.5

        random.seed(42)
        jobs = []
        for i in range(total_jobs):
            r = random.random() * 100
            if r < ttft_pct:
                obj = "ttft"
                inp, out = random.randint(200, 1500), random.randint(50, 200)
            elif r < ttft_pct + tpot_pct:
                obj = "tpot"
                inp, out = random.randint(100, 500), random.randint(200, 600)
            else:
                obj = "e2e"
                inp, out = random.randint(300, 1200), random.randint(100, 400)

            jobs.append({"objective": obj, "input": inp, "output": out})

        # Run simulations
        modes = ["pd", "ppd", "replica"]
        mode_results = {}

        for mode in modes:
            random.seed(42)
            mode_results[mode] = run_simulation(jobs, mode, slo_thresholds)

        random.seed(42)
        mode_results["optimizer"] = run_optimizer_simulation(jobs, selector, slo_thresholds)

        results["e2e_ratio"].append(e2e_pct)
        for mode in modes + ["optimizer"]:
            results[mode].append(mode_results[mode]["attainment"] * 100)

        print(f"  PD: {mode_results['pd']['attainment']*100:.1f}%, "
              f"PPD: {mode_results['ppd']['attainment']*100:.1f}%, "
              f"Replica: {mode_results['replica']['attainment']*100:.1f}%, "
              f"Optimizer: {mode_results['optimizer']['attainment']*100:.1f}%")

    return results


def generate_slo_strictness_trend_data():
    """
    Generate trend data: SLO Attainment vs SLO Strictness

    X-axis: SLO Strictness Factor (0.5x to 2.0x baseline)
    Y-axis: SLO Attainment (%)
    """
    print("\n" + "="*70)
    print("GENERATING SLO STRICTNESS TREND DATA")
    print("="*70)

    # Strictness factors (multiplier for baseline SLO)
    strictness_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    baseline_slo = {"ttft_ms": 80, "tpot_ms": 8, "e2e_ms": 2000}

    total_jobs = 50
    workload_mix = {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}

    results = {"strictness": [], "pd": [], "ppd": [], "replica": [], "optimizer": []}
    selector = RuleBasedSelector(verbose=False)

    # Generate fixed job set
    random.seed(42)
    jobs = []
    for i in range(total_jobs):
        r = random.random()
        if r < workload_mix["ttft"]:
            obj = "ttft"
            inp, out = random.randint(200, 1500), random.randint(50, 200)
        elif r < workload_mix["ttft"] + workload_mix["tpot"]:
            obj = "tpot"
            inp, out = random.randint(100, 500), random.randint(200, 600)
        else:
            obj = "e2e"
            inp, out = random.randint(300, 1200), random.randint(100, 400)

        jobs.append({"objective": obj, "input": inp, "output": out})

    for factor in strictness_factors:
        print(f"\nTesting SLO Strictness Factor={factor}x...")

        # Adjust SLO thresholds (lower factor = stricter)
        slo_thresholds = {
            "ttft_ms": baseline_slo["ttft_ms"] * factor,
            "tpot_ms": baseline_slo["tpot_ms"] * factor,
            "e2e_ms": baseline_slo["e2e_ms"] * factor,
        }

        # Run simulations
        modes = ["pd", "ppd", "replica"]
        mode_results = {}

        for mode in modes:
            random.seed(42)
            mode_results[mode] = run_simulation(jobs, mode, slo_thresholds)

        random.seed(42)
        mode_results["optimizer"] = run_optimizer_simulation(jobs, selector, slo_thresholds)

        results["strictness"].append(factor)
        for mode in modes + ["optimizer"]:
            results[mode].append(mode_results[mode]["attainment"] * 100)

        print(f"  PD: {mode_results['pd']['attainment']*100:.1f}%, "
              f"PPD: {mode_results['ppd']['attainment']*100:.1f}%, "
              f"Replica: {mode_results['replica']['attainment']*100:.1f}%, "
              f"Optimizer: {mode_results['optimizer']['attainment']*100:.1f}%")

    return results


def generate_input_length_trend_data():
    """
    Generate trend data: SLO Attainment vs Average Input Length

    X-axis: Average Input Length (100, 500, 1000, 2000, 3000, 4000, 5000)
    Y-axis: SLO Attainment (%)
    """
    print("\n" + "="*70)
    print("GENERATING INPUT LENGTH TREND DATA")
    print("="*70)

    input_lengths = [100, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    total_jobs = 50
    slo_thresholds = {"ttft_ms": 150, "tpot_ms": 10, "e2e_ms": 3000}

    results = {"input_length": [], "pd": [], "ppd": [], "replica": [], "optimizer": []}
    selector = RuleBasedSelector(verbose=False)

    for avg_input in input_lengths:
        print(f"\nTesting Avg Input Length={avg_input}...")

        # Generate jobs with varying input lengths around the average
        random.seed(42)
        jobs = []
        objectives = ["ttft", "tpot", "e2e"]
        for i in range(total_jobs):
            obj = objectives[i % 3]
            inp = max(50, int(random.gauss(avg_input, avg_input * 0.3)))
            out = random.randint(100, 400)

            jobs.append({"objective": obj, "input": inp, "output": out})

        # Run simulations
        modes = ["pd", "ppd", "replica"]
        mode_results = {}

        for mode in modes:
            random.seed(42)
            mode_results[mode] = run_simulation(jobs, mode, slo_thresholds)

        random.seed(42)
        mode_results["optimizer"] = run_optimizer_simulation(jobs, selector, slo_thresholds)

        results["input_length"].append(avg_input)
        for mode in modes + ["optimizer"]:
            results[mode].append(mode_results[mode]["attainment"] * 100)

        print(f"  PD: {mode_results['pd']['attainment']*100:.1f}%, "
              f"PPD: {mode_results['ppd']['attainment']*100:.1f}%, "
              f"Replica: {mode_results['replica']['attainment']*100:.1f}%, "
              f"Optimizer: {mode_results['optimizer']['attainment']*100:.1f}%")

    return results


def main():
    print("="*70)
    print("GENERATING TREND DATA FOR PAPER FIGURES")
    print("="*70)

    all_data = {}

    # Generate all trend datasets
    all_data["qps_trend"] = generate_qps_trend_data()
    all_data["e2e_ratio_trend"] = generate_e2e_ratio_trend_data()
    all_data["slo_strictness_trend"] = generate_slo_strictness_trend_data()
    all_data["input_length_trend"] = generate_input_length_trend_data()

    # Save all data
    output_path = os.path.join(os.path.dirname(__file__), "../results/trend_data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"All trend data saved to {output_path}")
    print(f"{'='*70}")

    # Print summary
    print("\n" + "="*70)
    print("TREND DATA SUMMARY")
    print("="*70)

    for name, data in all_data.items():
        x_key = list(data.keys())[0]
        print(f"\n{name}:")
        print(f"  X-axis: {x_key} ({len(data[x_key])} points)")

        # Calculate optimizer advantage
        opt_avg = np.mean(data["optimizer"])
        replica_avg = np.mean(data["replica"])
        ppd_avg = np.mean(data["ppd"])
        pd_avg = np.mean(data["pd"])

        print(f"  Avg Attainment: PD={pd_avg:.1f}%, PPD={ppd_avg:.1f}%, "
              f"Replica={replica_avg:.1f}%, Optimizer={opt_avg:.1f}%")

        best_fixed = max(replica_avg, ppd_avg, pd_avg)
        print(f"  Optimizer Advantage: +{opt_avg - best_fixed:.1f}% vs best fixed mode")


if __name__ == "__main__":
    main()
