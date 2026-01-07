#!/usr/bin/env python3
"""
Average benchmark results across multiple runs.

Usage: python scripts/average_runs.py [run1,run2,run3]
       Default: averages run1, run2, run3

This script:
1. Loads merged results from each run directory
2. Groups results by (workload, mode, qps)
3. Calculates mean and std for all metrics
4. Saves averaged results to results/final/
5. Runs analysis on the final averaged data
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


def load_run_data(run_num: int):
    """Load merged data from a specific run."""
    run_dir = RESULTS_DIR / f"run{run_num}"

    qps_files = sorted(run_dir.glob("qps_benchmark_v3_merged_*.json"))
    rep_files = sorted(run_dir.glob("replication_benchmark_v3_merged_*.json"))

    if not qps_files or not rep_files:
        print(f"  Warning: Run {run_num} data incomplete")
        return None, None

    qps_data = json.load(open(qps_files[-1]))
    rep_data = json.load(open(rep_files[-1]))

    print(f"  Run {run_num}: {len(qps_data['results'])} QPS, {len(rep_data['results'])} Rep experiments")

    return qps_data, rep_data


def group_results(all_runs_qps, all_runs_rep):
    """Group results by (workload, mode, qps) across all runs."""
    # Structure: groups[workload][mode][qps] = [result1, result2, ...]
    qps_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rep_groups = defaultdict(lambda: defaultdict(list))

    for qps_data in all_runs_qps:
        if qps_data is None:
            continue
        for r in qps_data['results']:
            wk = r['workload']
            mode = r['mode']
            qps = r['target_qps']
            qps_groups[wk][mode][qps].append(r)

    for rep_data in all_runs_rep:
        if rep_data is None:
            continue
        for r in rep_data['results']:
            wk = r['workload']
            qps = r['target_qps']
            rep_groups[wk][qps].append(r)

    return qps_groups, rep_groups


def average_metrics(results_list):
    """Calculate average metrics from a list of result dicts."""
    if not results_list:
        return None

    # Take the first result as template
    avg_result = results_list[0].copy()

    # Metrics to average in turn2_metrics
    t2_metrics = ['avg_ttft', 'p50_ttft', 'p90_ttft', 'p99_ttft',
                  'avg_tpot', 'p50_tpot', 'p99_tpot',
                  'avg_e2e', 'p50_e2e', 'p99_e2e',
                  'avg_throughput_tps', 'total_tokens']

    # Metrics to average in turn1_metrics
    t1_metrics = ['avg_ttft', 'p50_ttft', 'p90_ttft', 'p99_ttft',
                  'avg_e2e', 'p50_e2e', 'p99_e2e']

    # Top-level metrics to average
    top_metrics = ['avg_total_e2e', 'p99_total_e2e', 'success_count', 'sample_count']

    # Initialize averaged turn2_metrics
    if 'turn2_metrics' in avg_result and avg_result['turn2_metrics']:
        avg_t2 = {}
        for metric in t2_metrics:
            values = [r['turn2_metrics'].get(metric, 0) for r in results_list
                     if r.get('turn2_metrics')]
            if values:
                avg_t2[metric] = float(np.mean(values))
                avg_t2[f'{metric}_std'] = float(np.std(values))
        avg_t2['turn'] = 2
        avg_t2['sample_count'] = int(np.mean([r['turn2_metrics'].get('sample_count', 0)
                                              for r in results_list if r.get('turn2_metrics')]))
        avg_t2['success_count'] = int(np.mean([r['turn2_metrics'].get('success_count', 0)
                                               for r in results_list if r.get('turn2_metrics')]))
        avg_result['turn2_metrics'] = avg_t2

    # Initialize averaged turn1_metrics
    if 'turn1_metrics' in avg_result and avg_result['turn1_metrics']:
        avg_t1 = {}
        for metric in t1_metrics:
            values = [r['turn1_metrics'].get(metric, 0) for r in results_list
                     if r.get('turn1_metrics')]
            if values:
                avg_t1[metric] = float(np.mean(values))
                avg_t1[f'{metric}_std'] = float(np.std(values))
        avg_t1['turn'] = 1
        avg_result['turn1_metrics'] = avg_t1

    # Average top-level metrics
    for metric in top_metrics:
        values = [r.get(metric, 0) for r in results_list]
        if values:
            avg_result[metric] = float(np.mean(values))
            avg_result[f'{metric}_std'] = float(np.std(values))

    # Add metadata
    avg_result['num_runs_averaged'] = len(results_list)

    return avg_result


def create_averaged_data(qps_groups, rep_groups, runs_used):
    """Create final averaged data structures."""

    # QPS averaged data
    qps_averaged = {
        "benchmark_type": "qps_comparison_v3_averaged",
        "runs_averaged": runs_used,
        "merge_time": datetime.now().isoformat(),
        "results": []
    }

    for wk in sorted(qps_groups.keys()):
        for mode in sorted(qps_groups[wk].keys()):
            for qps in sorted(qps_groups[wk][mode].keys()):
                results_list = qps_groups[wk][mode][qps]
                avg_result = average_metrics(results_list)
                if avg_result:
                    qps_averaged["results"].append(avg_result)

    # Replication averaged data
    rep_averaged = {
        "benchmark_type": "replication_v3_averaged",
        "runs_averaged": runs_used,
        "merge_time": datetime.now().isoformat(),
        "results": []
    }

    for wk in sorted(rep_groups.keys()):
        for qps in sorted(rep_groups[wk].keys()):
            results_list = rep_groups[wk][qps]
            avg_result = average_metrics(results_list)
            if avg_result:
                rep_averaged["results"].append(avg_result)

    return qps_averaged, rep_averaged


def main():
    # Parse run numbers from command line
    if len(sys.argv) > 1:
        runs = [int(x) for x in sys.argv[1].split(',')]
    else:
        runs = [1, 2, 3]  # Default

    print("=" * 60)
    print("Averaging Benchmark Results Across Runs")
    print("=" * 60)
    print(f"Runs to average: {runs}")

    # Create final output directory
    final_dir = RESULTS_DIR / "final"
    final_dir.mkdir(exist_ok=True)

    # Load all run data
    print("\nLoading run data...")
    all_qps = []
    all_rep = []
    runs_loaded = []

    for run_num in runs:
        qps_data, rep_data = load_run_data(run_num)
        if qps_data and rep_data:
            all_qps.append(qps_data)
            all_rep.append(rep_data)
            runs_loaded.append(run_num)

    if not runs_loaded:
        print("Error: No valid run data found!")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(runs_loaded)} runs: {runs_loaded}")

    # Group and average
    print("\nGrouping results...")
    qps_groups, rep_groups = group_results(all_qps, all_rep)

    print(f"  QPS groups: {sum(len(qps_groups[wk][m]) for wk in qps_groups for m in qps_groups[wk])} (workload, mode) combinations")
    print(f"  Rep groups: {sum(len(rep_groups[wk]) for wk in rep_groups)} (workload) combinations")

    print("\nCalculating averages...")
    qps_averaged, rep_averaged = create_averaged_data(qps_groups, rep_groups, runs_loaded)

    # Save averaged results
    print("\nSaving averaged results...")

    qps_output = final_dir / f"qps_benchmark_v3_averaged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(qps_output, 'w') as f:
        json.dump(qps_averaged, f, indent=2)
    print(f"  {qps_output.name}: {len(qps_averaged['results'])} experiments")

    rep_output = final_dir / f"replication_benchmark_v3_averaged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(rep_output, 'w') as f:
        json.dump(rep_averaged, f, indent=2)
    print(f"  {rep_output.name}: {len(rep_averaged['results'])} experiments")

    # Create symlinks for analyze_results.py compatibility
    # (analyze_results.py looks for qps_benchmark_v3_merged_*.json)
    qps_link = final_dir / f"qps_benchmark_v3_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    rep_link = final_dir / f"replication_benchmark_v3_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Just copy the files with the expected naming convention
    with open(qps_link, 'w') as f:
        json.dump(qps_averaged, f, indent=2)
    with open(rep_link, 'w') as f:
        json.dump(rep_averaged, f, indent=2)

    print("\n" + "=" * 60)
    print("Averaging Complete!")
    print("=" * 60)
    print(f"Output directory: {final_dir}")
    print(f"Runs averaged: {runs_loaded}")
    print(f"QPS experiments: {len(qps_averaged['results'])}")
    print(f"Replication experiments: {len(rep_averaged['results'])}")
    print("\nTo generate final analysis plots:")
    print("  python scripts/analyze_results.py final")
    print("=" * 60)


if __name__ == "__main__":
    main()
