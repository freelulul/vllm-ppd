#!/usr/bin/env python3
"""
Production Scenario Test

Simulates realistic production workload patterns:
1. Dynamic workload shift - composition changes over time
2. Burst traffic - sudden spikes in request rate
3. Mixed SLO priorities - different jobs have different strictness
4. Diurnal patterns - varying load throughout "day"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict

from optimizer.models.rule_based_selector import (
    RuleBasedSelector, RequestFeatures, OptimizationObjective, Mode
)


@dataclass
class Job:
    """A multi-turn conversation job"""
    job_id: int
    objective: OptimizationObjective
    t1_input: int
    t1_output: int
    t2_input: int
    t2_output: int
    slo_strictness: str  # "strict", "normal", "relaxed"
    arrival_phase: str   # "normal", "burst", "low"


def get_slo_thresholds(strictness: str) -> Dict[str, float]:
    """Get SLO thresholds based on strictness level"""
    thresholds = {
        "strict": {"ttft_ms": 50, "tpot_ms": 7, "e2e_ms": 1500},
        "normal": {"ttft_ms": 100, "tpot_ms": 10, "e2e_ms": 2500},
        "relaxed": {"ttft_ms": 200, "tpot_ms": 15, "e2e_ms": 4000},
    }
    return thresholds.get(strictness, thresholds["normal"])


def simulate_latency(mode: str, objective: str, input_len: int, output_len: int,
                     turn: int, queue_depth: int) -> Dict[str, float]:
    """
    Simulate latency based on mode and workload characteristics.
    Uses realistic distributions from benchmark data.
    """
    random.seed(hash((mode, objective, input_len, output_len, turn, queue_depth)))

    # Base latencies from benchmark (with some variation)
    base = {
        "pd": {"ttft": 80, "tpot": 3.5, "prefill_per_token": 0.06},
        "ppd": {"ttft": 25, "tpot": 3.6, "prefill_per_token": 0.04},
        "replica": {"ttft": 15, "tpot": 4.0, "prefill_per_token": 0.05},
    }

    params = base[mode]

    # Queue delay (realistic: ~50-200ms per queued request)
    queue_delay = queue_depth * random.uniform(50, 150)

    # TTFT calculation
    if mode == "pd":
        # PD: prefill on P, then transfer to D
        ttft = params["ttft"] + input_len * params["prefill_per_token"] + 80  # transfer overhead
    elif mode == "ppd":
        # PPD: Turn 2 benefits from cache
        if turn >= 2:
            ttft = params["ttft"] + input_len * params["prefill_per_token"] * 0.3  # cache benefit
        else:
            ttft = params["ttft"] + input_len * params["prefill_per_token"]
    else:  # replica
        ttft = params["ttft"] + input_len * params["prefill_per_token"]

    ttft += queue_delay * 0.3  # partial queue impact on TTFT
    ttft *= random.uniform(0.85, 1.15)  # variance

    # TPOT calculation
    tpot = params["tpot"] * random.uniform(0.9, 1.1)
    if queue_depth > 3:
        tpot *= 1 + (queue_depth - 3) * 0.05  # congestion impact

    # E2E calculation
    e2e = ttft + output_len * tpot
    if mode == "pd":
        e2e += 100  # KV transfer overhead impacts total

    return {"ttft": ttft, "tpot": tpot, "e2e": e2e}


def generate_production_workload(scenario: str) -> List[Job]:
    """Generate realistic production workload"""
    jobs = []
    job_id = 0

    if scenario == "diurnal":
        # Morning: TTFT-heavy (quick queries)
        # Afternoon: Balanced
        # Evening: E2E-heavy (long tasks)
        phases = [
            ("morning", 20, {"ttft": 0.6, "tpot": 0.2, "e2e": 0.2}, "normal"),
            ("afternoon", 30, {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}, "normal"),
            ("evening", 20, {"ttft": 0.2, "tpot": 0.2, "e2e": 0.6}, "normal"),
        ]
    elif scenario == "burst":
        # Normal load with burst periods
        phases = [
            ("pre_burst", 15, {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}, "normal"),
            ("burst", 25, {"ttft": 0.4, "tpot": 0.3, "e2e": 0.3}, "burst"),
            ("post_burst", 15, {"ttft": 0.33, "tpot": 0.34, "e2e": 0.33}, "normal"),
        ]
    elif scenario == "mixed_slo":
        # Mix of strict and relaxed SLO requirements
        phases = [
            ("strict_phase", 20, {"ttft": 0.5, "tpot": 0.3, "e2e": 0.2}, "normal"),
            ("relaxed_phase", 20, {"ttft": 0.3, "tpot": 0.3, "e2e": 0.4}, "normal"),
        ]
    else:  # "realistic"
        # Realistic mixed workload
        phases = [
            ("steady", 40, {"ttft": 0.35, "tpot": 0.30, "e2e": 0.35}, "normal"),
        ]

    # Token length distributions (realistic ranges)
    length_profiles = {
        "ttft": {"input": (200, 2000), "output": (50, 300)},
        "tpot": {"input": (100, 500), "output": (200, 800)},
        "e2e": {"input": (300, 1500), "output": (100, 500)},
    }

    for phase_name, num_jobs, mix, arrival_phase in phases:
        for _ in range(num_jobs):
            # Randomly select objective based on mix
            r = random.random()
            if r < mix["ttft"]:
                obj = OptimizationObjective.TTFT
            elif r < mix["ttft"] + mix["tpot"]:
                obj = OptimizationObjective.TPOT
            else:
                obj = OptimizationObjective.E2E

            # Get length profile
            profile = length_profiles[obj.value]
            t1_input = random.randint(*profile["input"])
            t1_output = random.randint(*profile["output"])
            t2_input = random.randint(50, 500)  # Turn 2 typically shorter input
            t2_output = random.randint(*profile["output"])

            # SLO strictness varies
            if scenario == "mixed_slo":
                strictness = random.choice(["strict", "strict", "normal", "relaxed"])
            else:
                strictness = random.choice(["strict", "normal", "normal", "relaxed"])

            jobs.append(Job(
                job_id=job_id,
                objective=obj,
                t1_input=t1_input,
                t1_output=t1_output,
                t2_input=t2_input,
                t2_output=t2_output,
                slo_strictness=strictness,
                arrival_phase=arrival_phase,
            ))
            job_id += 1

    return jobs


def run_mode_simulation(jobs: List[Job], mode: str) -> Dict:
    """Run simulation for a fixed mode"""
    results = {"met": 0, "total": 0, "by_objective": defaultdict(lambda: {"met": 0, "total": 0})}

    # Simulate queue depths dynamically
    queue_depths = {"gpu0": 0, "gpu1": 0, "gpu2": 0, "gpu3": 0}

    for job in jobs:
        thresholds = get_slo_thresholds(job.slo_strictness)
        obj = job.objective.value

        # Simulate Turn 2 latency (main measurement)
        if mode == "pd":
            target_gpu = "gpu1"
        elif mode == "ppd":
            target_gpu = "gpu1"
        else:  # replica - round robin
            target_gpu = "gpu2" if queue_depths["gpu2"] <= queue_depths["gpu3"] else "gpu3"

        queue_depth = queue_depths.get(target_gpu, 0)
        latency = simulate_latency(mode, obj, job.t2_input, job.t2_output, turn=2, queue_depth=queue_depth)

        # Update queue (simplified)
        queue_depths[target_gpu] = min(queue_depths[target_gpu] + 1, 10)
        if random.random() > 0.7:  # Some requests complete
            for gpu in queue_depths:
                queue_depths[gpu] = max(0, queue_depths[gpu] - 1)

        # Check SLO
        met = False
        if obj == "ttft":
            met = latency["ttft"] <= thresholds["ttft_ms"]
        elif obj == "tpot":
            met = latency["tpot"] <= thresholds["tpot_ms"]
        elif obj == "e2e":
            met = latency["e2e"] <= thresholds["e2e_ms"]

        results["total"] += 1
        results["by_objective"][obj]["total"] += 1
        if met:
            results["met"] += 1
            results["by_objective"][obj]["met"] += 1

    results["attainment"] = results["met"] / results["total"] if results["total"] > 0 else 0
    return results


def run_optimizer_simulation(jobs: List[Job], selector: RuleBasedSelector) -> Dict:
    """Run simulation with dynamic optimizer routing"""
    results = {"met": 0, "total": 0, "by_objective": defaultdict(lambda: {"met": 0, "total": 0})}
    routing_stats = defaultdict(int)

    # Simulate queue depths dynamically
    queue_depths = {"gpu0": 0, "gpu1": 0, "gpu2": 0, "gpu3": 0}

    for job in jobs:
        thresholds = get_slo_thresholds(job.slo_strictness)
        obj = job.objective.value

        # Get routing decision from optimizer
        features = RequestFeatures(
            input_length=job.t2_input,
            output_length=job.t2_output,
            turn_number=2,
            has_cache=True,
            cached_gpu="gpu1" if random.random() > 0.5 else "gpu2",  # Simulate cache location
            queue_depths=queue_depths.copy(),
            objective=job.objective
        )

        decision = selector.select_mode(features)
        mode = decision.mode.value
        target_gpu = decision.target_gpu
        routing_stats[mode] += 1

        queue_depth = queue_depths.get(target_gpu, 0)
        latency = simulate_latency(mode, obj, job.t2_input, job.t2_output, turn=2, queue_depth=queue_depth)

        # Update queue
        queue_depths[target_gpu] = min(queue_depths[target_gpu] + 1, 10)
        if random.random() > 0.7:
            for gpu in queue_depths:
                queue_depths[gpu] = max(0, queue_depths[gpu] - 1)

        # Check SLO
        met = False
        if obj == "ttft":
            met = latency["ttft"] <= thresholds["ttft_ms"]
        elif obj == "tpot":
            met = latency["tpot"] <= thresholds["tpot_ms"]
        elif obj == "e2e":
            met = latency["e2e"] <= thresholds["e2e_ms"]

        results["total"] += 1
        results["by_objective"][obj]["total"] += 1
        if met:
            results["met"] += 1
            results["by_objective"][obj]["met"] += 1

    results["attainment"] = results["met"] / results["total"] if results["total"] > 0 else 0
    results["routing"] = dict(routing_stats)
    return results


def run_scenario_test(scenario: str):
    """Run test for a specific scenario"""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{'='*70}")

    # Generate workload
    random.seed(42)  # Reproducibility
    jobs = generate_production_workload(scenario)

    print(f"\nWorkload Statistics:")
    obj_counts = defaultdict(int)
    slo_counts = defaultdict(int)
    phase_counts = defaultdict(int)
    for job in jobs:
        obj_counts[job.objective.value] += 1
        slo_counts[job.slo_strictness] += 1
        phase_counts[job.arrival_phase] += 1

    print(f"  Total jobs: {len(jobs)}")
    print(f"  By objective: {dict(obj_counts)}")
    print(f"  By SLO strictness: {dict(slo_counts)}")
    print(f"  By arrival phase: {dict(phase_counts)}")

    # Run simulations
    selector = RuleBasedSelector(verbose=False)

    modes = ["pd", "ppd", "replica"]
    results = {}

    for mode in modes:
        random.seed(42)  # Same seed for fair comparison
        results[mode] = run_mode_simulation(jobs, mode)

    random.seed(42)
    results["optimizer"] = run_optimizer_simulation(jobs, selector)

    # Print results
    print(f"\n{'─'*70}")
    print("SLO Attainment Results:")
    print(f"{'─'*70}")

    for mode in modes + ["optimizer"]:
        r = results[mode]
        print(f"  {mode:12}: {r['attainment']*100:5.1f}% ({r['met']}/{r['total']})")
        if mode == "optimizer" and "routing" in r:
            print(f"    Routing: {r['routing']}")

    # Find best
    best_fixed = max(modes, key=lambda m: results[m]["attainment"])
    opt_att = results["optimizer"]["attainment"]
    best_fixed_att = results[best_fixed]["attainment"]

    diff = (opt_att - best_fixed_att) * 100
    if diff > 0:
        verdict = f"+{diff:.1f}% vs {best_fixed}"
    elif diff < 0:
        verdict = f"{diff:.1f}% vs {best_fixed}"
    else:
        verdict = f"持平 {best_fixed}"

    print(f"\n  Optimizer vs Best-Fixed: {verdict}")

    # Per-objective breakdown
    print(f"\n{'─'*70}")
    print("Per-Objective Breakdown:")
    print(f"{'─'*70}")

    for obj in ["ttft", "tpot", "e2e"]:
        print(f"\n  {obj.upper()}:")
        for mode in modes + ["optimizer"]:
            r = results[mode]["by_objective"].get(obj, {"met": 0, "total": 0})
            if r["total"] > 0:
                att = r["met"] / r["total"] * 100
                print(f"    {mode:12}: {att:5.1f}% ({r['met']}/{r['total']})")

    return {
        "scenario": scenario,
        "jobs": len(jobs),
        "results": {m: {"attainment": results[m]["attainment"], "met": results[m]["met"], "total": results[m]["total"]}
                   for m in modes + ["optimizer"]},
        "optimizer_routing": results["optimizer"].get("routing", {}),
        "verdict": verdict
    }


def main():
    print("="*70)
    print("PRODUCTION SCENARIO TEST")
    print("="*70)
    print("\nSimulating realistic production workload patterns...")

    scenarios = ["diurnal", "burst", "mixed_slo", "realistic"]
    all_results = {}

    for scenario in scenarios:
        result = run_scenario_test(scenario)
        all_results[scenario] = result

    # Summary
    print("\n" + "="*70)
    print("PRODUCTION SCENARIO SUMMARY")
    print("="*70)

    print(f"\n{'Scenario':<15} {'PD':>8} {'PPD':>8} {'Replica':>8} {'Optimizer':>10} {'Verdict':<20}")
    print("-"*70)

    for scenario, result in all_results.items():
        r = result["results"]
        print(f"{scenario:<15} {r['pd']['attainment']*100:7.1f}% {r['ppd']['attainment']*100:7.1f}% "
              f"{r['replica']['attainment']*100:7.1f}% {r['optimizer']['attainment']*100:9.1f}% "
              f"{result['verdict']:<20}")

    # Calculate overall winner counts
    print(f"\n{'─'*70}")
    optimizer_wins = sum(1 for r in all_results.values() if "+" in r["verdict"])
    ties = sum(1 for r in all_results.values() if "持平" in r["verdict"])
    losses = len(all_results) - optimizer_wins - ties

    print(f"Optimizer: {optimizer_wins} wins, {ties} ties, {losses} losses")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "../results/production_scenario.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
