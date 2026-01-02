#!/usr/bin/env python3
"""
Merge QPS Benchmark Results from Multiple Sources

This script merges results from:
- PD/PPD benchmark (qps_benchmark_*.json)
- Replication benchmark (replication_benchmark_*.json)

Into a single combined file for 3-mode comparison plotting.

Features:
- Filters out anomalous results (high failure rate, zero samples)
- Averages across multiple runs
- Handles missing data gracefully

Usage:
    python scripts/merge_results.py results/qps_benchmark_*.json results/replication_benchmark_*.json
    python scripts/merge_results.py --auto  # Auto-detect latest files
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import statistics

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

# Anomaly thresholds
MAX_FAILURE_RATE = 0.3  # Filter out results with >30% failure rate
MIN_SAMPLES = 1  # Require at least 1 sample


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def is_anomalous(result: dict) -> tuple[bool, str]:
    """Check if a result entry is anomalous and should be filtered."""
    success = result.get('success_count', 0)
    failure = result.get('failure_count', 0)
    total = success + failure

    # Zero samples
    if success == 0:
        return True, "zero samples"

    # High failure rate
    if total > 0:
        failure_rate = failure / total
        if failure_rate > MAX_FAILURE_RATE:
            return True, f"high failure rate ({failure_rate:.1%})"

    return False, ""


def average_metrics(results: list[dict]) -> dict:
    """Average metrics across multiple runs, filtering anomalies."""
    valid_results = []
    filtered_reasons = []

    for r in results:
        is_anom, reason = is_anomalous(r)
        if is_anom:
            filtered_reasons.append(reason)
        else:
            valid_results.append(r)

    if not valid_results:
        return None  # All results were anomalous

    # If we filtered some results, note it
    if filtered_reasons:
        print(f"    Filtered {len(filtered_reasons)} runs: {', '.join(filtered_reasons)}")

    # Average the metrics
    avg_result = valid_results[0].copy()

    # Metrics to average
    numeric_keys = [
        'sample_count', 'success_count', 'failure_count', 'real_qps',
        'avg_ttft', 'p50_ttft', 'p90_ttft', 'p99_ttft', 'min_ttft', 'max_ttft',
        'avg_e2e', 'p50_e2e', 'p90_e2e', 'p99_e2e',
        'avg_throughput_tps', 'total_tokens'
    ]

    for key in numeric_keys:
        values = [r.get(key, 0) for r in valid_results if r.get(key) is not None]
        if values:
            avg_result[key] = statistics.mean(values)

    # Average turn metrics
    for turn_key in ['turn1_metrics', 'turn2_metrics']:
        if turn_key in valid_results[0]:
            turn_metrics = avg_result[turn_key].copy()
            turn_numeric_keys = [
                'sample_count', 'success_count',
                'avg_ttft', 'p50_ttft', 'p90_ttft', 'p99_ttft',
                'avg_tpot', 'p50_tpot', 'p99_tpot',
                'avg_e2e', 'p50_e2e', 'p99_e2e',
                'avg_throughput_tps', 'total_tokens'
            ]
            for key in turn_numeric_keys:
                values = [r[turn_key].get(key, 0) for r in valid_results if r.get(turn_key) and r[turn_key].get(key) is not None]
                if values:
                    turn_metrics[key] = statistics.mean(values)
            avg_result[turn_key] = turn_metrics

    # Add metadata about averaging
    avg_result['runs_used'] = len(valid_results)
    avg_result['runs_filtered'] = len(filtered_reasons)

    return avg_result


def find_latest_files() -> tuple[str, str]:
    """Find the latest PD/PPD and Replication result files."""
    pdppd_files = sorted(RESULTS_DIR.glob("qps_benchmark_*.json"))
    repl_files = sorted(RESULTS_DIR.glob("replication_benchmark_*.json"))

    if not pdppd_files:
        raise FileNotFoundError("No qps_benchmark_*.json files found in results/")
    if not repl_files:
        raise FileNotFoundError("No replication_benchmark_*.json files found in results/")

    return str(pdppd_files[-1]), str(repl_files[-1])


def merge_results(pdppd_file: str, repl_file: str, average_runs: bool = True) -> dict:
    """Merge PD/PPD and Replication results into a single dataset."""
    pdppd_data = load_results(pdppd_file)
    repl_data = load_results(repl_file)

    print(f"PD/PPD file: {pdppd_file}")
    print(f"  Timestamp: {pdppd_data.get('timestamp', 'unknown')}")
    print(f"  Results: {len(pdppd_data['results'])} entries")
    print(f"  Runs: {pdppd_data['config'].get('runs', 1)}")

    print(f"Replication file: {repl_file}")
    print(f"  Timestamp: {repl_data.get('timestamp', 'unknown')}")
    print(f"  Results: {len(repl_data['results'])} entries")
    print(f"  Runs: {repl_data['config'].get('runs', 1)}")

    # Validate workloads match
    pdppd_workloads = set(r['workload'] for r in pdppd_data['results'])
    repl_workloads = set(r['workload'] for r in repl_data['results'])

    if pdppd_workloads != repl_workloads:
        print(f"\nWARNING: Workload mismatch!")
        print(f"  PD/PPD: {sorted(pdppd_workloads)}")
        print(f"  Repl:   {sorted(repl_workloads)}")

    # Validate QPS values match
    pdppd_qps = set(r['target_qps'] for r in pdppd_data['results'])
    repl_qps = set(r['target_qps'] for r in repl_data['results'])

    if pdppd_qps != repl_qps:
        print(f"\nWARNING: QPS mismatch!")
        print(f"  PD/PPD: {sorted(pdppd_qps)}")
        print(f"  Repl:   {sorted(repl_qps)}")

    # Combine all results
    all_results = pdppd_data['results'] + repl_data['results']

    # Group by (workload, qps, mode) for averaging
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['workload'], r['target_qps'], r['mode'])
        grouped[key].append(r)

    # Average across runs (filtering anomalies)
    if average_runs:
        print(f"\nAveraging across runs (filtering anomalies with >{MAX_FAILURE_RATE:.0%} failure rate)...")
        averaged_results = []
        skipped = 0

        for (workload, qps, mode), results in sorted(grouped.items()):
            print(f"  {workload} QPS={qps} {mode.upper()}: {len(results)} runs")
            avg_result = average_metrics(results)
            if avg_result is not None:
                averaged_results.append(avg_result)
            else:
                print(f"    SKIPPED: All runs were anomalous")
                skipped += 1

        final_results = averaged_results
        print(f"\nFinal: {len(final_results)} entries ({skipped} skipped due to anomalies)")
    else:
        final_results = all_results

    # Check for mode distribution
    modes = {}
    for r in final_results:
        mode = r['mode']
        modes[mode] = modes.get(mode, 0) + 1

    print(f"\nMerged results by mode:")
    for mode, count in sorted(modes.items()):
        print(f"  {mode}: {count} entries")

    # Create merged output
    merged = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_files": {
            "pdppd": pdppd_file,
            "replication": repl_file,
        },
        "config": {
            "workloads": pdppd_data['config']['workloads'],
            "qps_sweep": sorted(pdppd_qps | repl_qps),
            "duration_s": pdppd_data['config'].get('duration_s', 45),
            "averaged_runs": average_runs,
        },
        "results": final_results,
    }

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge QPS benchmark results")
    parser.add_argument("files", nargs="*", help="Result files to merge")
    parser.add_argument("--auto", action="store_true", help="Auto-detect latest files")
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    if args.auto:
        pdppd_file, repl_file = find_latest_files()
    elif len(args.files) >= 2:
        # Separate files by type
        pdppd_file = None
        repl_file = None
        for f in args.files:
            if "replication" in f:
                repl_file = f
            else:
                pdppd_file = f

        if not pdppd_file or not repl_file:
            print("ERROR: Need both qps_benchmark_*.json and replication_benchmark_*.json")
            return
    else:
        print("Usage: python merge_results.py --auto")
        print("   or: python merge_results.py qps_benchmark.json replication_benchmark.json")
        return

    # Merge results
    merged = merge_results(pdppd_file, repl_file)

    # Save merged file
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or str(RESULTS_DIR / f"merged_3mode_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged results saved to: {output_file}")
    print(f"\nTo generate plots, run:")
    print(f"  python scripts/plot_qps_curves.py {output_file}")


if __name__ == "__main__":
    main()
