#!/usr/bin/env python3
"""
Merge benchmark results from individual workload files (V3).

Usage: python scripts/merge_benchmark_results.py [RUN_NUM]
       RUN_NUM: 1, 2, 3, ... (merges results from results/run{N}/)

V3 Workload Naming: {T1}_{T2}
- T1: S, M, L, XL (context size)
- T2: a, b, c, d (request type)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent

# V3 workload names: 4 T1 x 4 T2 = 16 combinations
WORKLOADS = [
    "S_a", "S_b", "S_c", "S_d",
    "M_a", "M_b", "M_c", "M_d",
    "L_a", "L_b", "L_c", "L_d",
    "XL_a", "XL_b", "XL_c", "XL_d"
]


def merge_qps_results(results_dir: Path):
    """Merge QPS benchmark results (one file per workload)."""
    merged = {
        "benchmark_type": "qps_comparison_v3",
        "merged_from": [],
        "merge_time": datetime.now().isoformat(),
        "results": []
    }

    for wk in WORKLOADS:
        file_path = results_dir / f"qps_{wk}.json"
        if file_path.exists():
            print(f"  Loading qps_{wk}.json...")
            with open(file_path) as f:
                data = json.load(f)
            merged["merged_from"].append(file_path.name)
            merged["results"].extend(data.get("results", []))
        else:
            print(f"  Warning: qps_{wk}.json not found")

    if merged["results"]:
        output_file = results_dir / f"qps_benchmark_v3_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"QPS results: {len(merged['merged_from'])}/16 files, {len(merged['results'])} experiments -> {output_file.name}")
        return output_file
    else:
        print("No QPS results to merge")
        return None


def merge_replication_results(results_dir: Path):
    """Merge replication benchmark results (one file per workload)."""
    merged = {
        "benchmark_type": "replication_v3",
        "merged_from": [],
        "merge_time": datetime.now().isoformat(),
        "results": []
    }

    for wk in WORKLOADS:
        file_path = results_dir / f"replication_{wk}.json"
        if file_path.exists():
            print(f"  Loading replication_{wk}.json...")
            with open(file_path) as f:
                data = json.load(f)
            merged["merged_from"].append(file_path.name)
            merged["results"].extend(data.get("results", []))
        else:
            print(f"  Warning: replication_{wk}.json not found")

    if merged["results"]:
        output_file = results_dir / f"replication_benchmark_v3_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Replication results: {len(merged['merged_from'])}/16 files, {len(merged['results'])} experiments -> {output_file.name}")
        return output_file
    else:
        print("No replication results to merge")
        return None


def check_status(results_dir: Path):
    """Check which result files exist."""
    print("\n" + "=" * 60)
    print("Current Status")
    print("=" * 60)

    qps_done = sum(1 for wk in WORKLOADS if (results_dir / f"qps_{wk}.json").exists())
    rep_done = sum(1 for wk in WORKLOADS if (results_dir / f"replication_{wk}.json").exists())

    print(f"QPS (PD/PPD):  {qps_done}/16 workloads complete")
    print(f"Replication:   {rep_done}/16 workloads complete")

    if qps_done < 16:
        missing = [wk for wk in WORKLOADS if not (results_dir / f"qps_{wk}.json").exists()]
        print(f"  Missing QPS: {', '.join(missing)}")

    if rep_done < 16:
        missing = [wk for wk in WORKLOADS if not (results_dir / f"replication_{wk}.json").exists()]
        print(f"  Missing Replication: {', '.join(missing)}")


if __name__ == "__main__":
    # Get run number from command line
    run_num = int(sys.argv[1]) if len(sys.argv) > 1 else None

    if run_num:
        results_dir = PROJECT_DIR / "results" / f"run{run_num}"
    else:
        results_dir = PROJECT_DIR / "results"

    print("=" * 60)
    print(f"Merging Benchmark Results")
    print(f"Directory: {results_dir}")
    print("=" * 60)

    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    check_status(results_dir)

    print("\n" + "=" * 60)
    print("Merging QPS Results")
    print("=" * 60)
    merge_qps_results(results_dir)

    print("\n" + "=" * 60)
    print("Merging Replication Results")
    print("=" * 60)
    merge_replication_results(results_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
