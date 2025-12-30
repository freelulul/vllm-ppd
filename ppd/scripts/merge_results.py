#!/usr/bin/env python3
"""
Merge QPS Benchmark Results from Multiple Sources

This script merges results from:
- PD/PPD benchmark (qps_benchmark_*.json)
- Replication benchmark (replication_benchmark_*.json)

Into a single combined file for 3-mode comparison plotting.

Usage:
    python scripts/merge_results.py results/qps_benchmark_*.json results/replication_benchmark_*.json
    python scripts/merge_results.py --auto  # Auto-detect latest files
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def find_latest_files() -> tuple[str, str]:
    """Find the latest PD/PPD and Replication result files."""
    pdppd_files = sorted(RESULTS_DIR.glob("qps_benchmark_*.json"))
    repl_files = sorted(RESULTS_DIR.glob("replication_benchmark_*.json"))

    if not pdppd_files:
        raise FileNotFoundError("No qps_benchmark_*.json files found in results/")
    if not repl_files:
        raise FileNotFoundError("No replication_benchmark_*.json files found in results/")

    return str(pdppd_files[-1]), str(repl_files[-1])


def merge_results(pdppd_file: str, repl_file: str) -> dict:
    """Merge PD/PPD and Replication results into a single dataset."""
    pdppd_data = load_results(pdppd_file)
    repl_data = load_results(repl_file)

    print(f"PD/PPD file: {pdppd_file}")
    print(f"  Timestamp: {pdppd_data.get('timestamp', 'unknown')}")
    print(f"  Results: {len(pdppd_data['results'])} entries")

    print(f"Replication file: {repl_file}")
    print(f"  Timestamp: {repl_data.get('timestamp', 'unknown')}")
    print(f"  Results: {len(repl_data['results'])} entries")

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

    # Merge results
    all_results = pdppd_data['results'] + repl_data['results']

    # Check for mode distribution
    modes = {}
    for r in all_results:
        mode = r['mode']
        modes[mode] = modes.get(mode, 0) + 1

    print(f"\nMerged results:")
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
        },
        "results": all_results,
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
