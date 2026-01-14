#!/usr/bin/env python3
"""
Merge benchmark results from multiple runs and compute averages.

Usage:
    python scripts/analysis/merge_results.py [--input-dir results] [--output merged_results.json]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np


def load_results(input_dir: Path):
    """Load all JSON result files from directory."""
    all_results = []

    for mode_dir in ["ppd", "pd", "replica"]:
        mode_path = input_dir / mode_dir
        if not mode_path.exists():
            continue

        for json_file in sorted(mode_path.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Handle new format: {"benchmark_type": ..., "results": [...]}
                if isinstance(data, dict) and "results" in data:
                    for r in data["results"]:
                        r["_source_file"] = str(json_file)
                    all_results.extend(data["results"])
                # Handle old format: [...]
                elif isinstance(data, list):
                    for r in data:
                        r["_source_file"] = str(json_file)
                    all_results.extend(data)
                else:
                    data["_source_file"] = str(json_file)
                    all_results.append(data)

                print(f"  Loaded: {json_file.name}")
            except Exception as e:
                print(f"  Error loading {json_file}: {e}")

    return all_results


def safe_mean(values):
    """Compute mean of non-empty list, return 0 if empty."""
    values = [v for v in values if v is not None and v > 0]
    return float(np.mean(values)) if values else 0.0


def safe_std(values):
    """Compute std of list with >1 elements, return 0 otherwise."""
    values = [v for v in values if v is not None and v > 0]
    return float(np.std(values)) if len(values) > 1 else 0.0


def merge_results(results: list) -> dict:
    """Merge results by (mode, workload, qps) and compute averages."""

    # Group by (mode, workload, qps)
    grouped = defaultdict(list)
    for r in results:
        key = (r["mode"], r["workload"], r["target_qps"])
        grouped[key].append(r)

    merged = []

    for (mode, workload, qps), group in sorted(grouped.items()):
        num_runs = len(group)

        merged_result = {
            "mode": mode,
            "workload": workload,
            "target_qps": qps,
            "num_runs": num_runs,
            "t1_input": group[0].get("t1_input", 0),
            "t1_output": group[0].get("t1_output", 0),
            "t2_input": group[0].get("t2_input", 0),
            "t2_output": group[0].get("t2_output", 0),
            "num_turns": group[0].get("num_turns", 2),
        }

        # Merge Turn 1 metrics
        t1_metrics = {
            "turn": 1,
            "sample_count": sum(r.get("turn1_metrics", {}).get("sample_count", 0) for r in group),
            "success_count": sum(r.get("turn1_metrics", {}).get("success_count", 0) for r in group),
        }
        t1_ttfts = [r.get("turn1_metrics", {}).get("avg_ttft", 0) for r in group]
        t1_e2es = [r.get("turn1_metrics", {}).get("avg_e2e", 0) for r in group]
        t1_p99_ttfts = [r.get("turn1_metrics", {}).get("p99_ttft", 0) for r in group]
        t1_metrics["avg_ttft"] = safe_mean(t1_ttfts)
        t1_metrics["std_ttft"] = safe_std(t1_ttfts)
        t1_metrics["p99_ttft"] = safe_mean(t1_p99_ttfts)
        t1_metrics["avg_e2e"] = safe_mean(t1_e2es)

        # Merge Turn 1 cache
        t1_cache = {
            "local_cache_queries": sum(r.get("turn1_metrics", {}).get("cache", {}).get("local_cache_queries", 0) for r in group),
            "local_cache_hits": sum(r.get("turn1_metrics", {}).get("cache", {}).get("local_cache_hits", 0) for r in group),
            "external_cache_queries": sum(r.get("turn1_metrics", {}).get("cache", {}).get("external_cache_queries", 0) for r in group),
            "external_cache_hits": sum(r.get("turn1_metrics", {}).get("cache", {}).get("external_cache_hits", 0) for r in group),
        }
        t1_cache["local_hit_rate_pct"] = (t1_cache["local_cache_hits"] / t1_cache["local_cache_queries"] * 100) if t1_cache["local_cache_queries"] > 0 else 0
        t1_cache["external_hit_rate_pct"] = (t1_cache["external_cache_hits"] / t1_cache["external_cache_queries"] * 100) if t1_cache["external_cache_queries"] > 0 else 0
        t1_metrics["cache"] = t1_cache

        # Merge Turn 2 metrics
        t2_metrics = {
            "turn": 2,
            "sample_count": sum(r.get("turn2_metrics", {}).get("sample_count", 0) for r in group),
            "success_count": sum(r.get("turn2_metrics", {}).get("success_count", 0) for r in group),
        }
        t2_ttfts = [r.get("turn2_metrics", {}).get("avg_ttft", 0) for r in group]
        t2_tpots = [r.get("turn2_metrics", {}).get("avg_tpot", 0) for r in group]
        t2_e2es = [r.get("turn2_metrics", {}).get("avg_e2e", 0) for r in group]
        t2_throughputs = [r.get("turn2_metrics", {}).get("avg_throughput_tps", 0) for r in group]

        t2_metrics["avg_ttft"] = safe_mean(t2_ttfts)
        t2_metrics["std_ttft"] = safe_std(t2_ttfts)
        t2_metrics["avg_tpot"] = safe_mean(t2_tpots)
        t2_metrics["std_tpot"] = safe_std(t2_tpots)
        t2_metrics["avg_e2e"] = safe_mean(t2_e2es)
        t2_metrics["avg_throughput_tps"] = safe_mean(t2_throughputs)

        # P99 metrics (important for SLA)
        t2_p99_ttfts = [r.get("turn2_metrics", {}).get("p99_ttft", 0) for r in group]
        t2_p99_tpots = [r.get("turn2_metrics", {}).get("p99_tpot", 0) for r in group]
        t2_metrics["p99_ttft"] = safe_mean(t2_p99_ttfts)
        t2_metrics["p99_tpot"] = safe_mean(t2_p99_tpots)

        # KV reuse metrics (for model fitting)
        kv_reuse_pcts = [r.get("turn2_metrics", {}).get("kv_reuse_pct", 0) for r in group]
        t2_metrics["kv_reuse_pct"] = safe_mean(kv_reuse_pcts)
        t2_metrics["kv_reuse_tokens"] = group[0].get("turn2_metrics", {}).get("kv_reuse_tokens", 0)

        # Merge Turn 2 cache
        t2_cache = {
            "local_cache_queries": sum(r.get("turn2_metrics", {}).get("cache", {}).get("local_cache_queries", 0) for r in group),
            "local_cache_hits": sum(r.get("turn2_metrics", {}).get("cache", {}).get("local_cache_hits", 0) for r in group),
            "external_cache_queries": sum(r.get("turn2_metrics", {}).get("cache", {}).get("external_cache_queries", 0) for r in group),
            "external_cache_hits": sum(r.get("turn2_metrics", {}).get("cache", {}).get("external_cache_hits", 0) for r in group),
        }
        t2_cache["local_hit_rate_pct"] = (t2_cache["local_cache_hits"] / t2_cache["local_cache_queries"] * 100) if t2_cache["local_cache_queries"] > 0 else 0
        t2_cache["external_hit_rate_pct"] = (t2_cache["external_cache_hits"] / t2_cache["external_cache_queries"] * 100) if t2_cache["external_cache_queries"] > 0 else 0
        t2_metrics["cache"] = t2_cache

        merged_result["turn1_metrics"] = t1_metrics
        merged_result["turn2_metrics"] = t2_metrics

        # Aggregate top-level metrics
        merged_result["sample_count"] = sum(r.get("sample_count", 0) for r in group)
        merged_result["success_count"] = sum(r.get("success_count", 0) for r in group)
        merged_result["failure_count"] = sum(r.get("failure_count", 0) for r in group)

        merged_result["avg_ttft"] = safe_mean([r.get("avg_ttft", 0) for r in group])
        merged_result["std_ttft"] = safe_std([r.get("avg_ttft", 0) for r in group])
        merged_result["avg_e2e"] = safe_mean([r.get("avg_e2e", 0) for r in group])
        merged_result["avg_tpot"] = safe_mean([r.get("turn2_metrics", {}).get("avg_tpot", 0) for r in group])
        merged_result["avg_throughput_tps"] = safe_mean([r.get("avg_throughput_tps", 0) for r in group])
        merged_result["real_qps"] = safe_mean([r.get("real_qps", 0) for r in group])

        # Total cache stats
        total_cache = {
            "local_cache_queries": t1_cache["local_cache_queries"] + t2_cache["local_cache_queries"],
            "local_cache_hits": t1_cache["local_cache_hits"] + t2_cache["local_cache_hits"],
            "external_cache_queries": t1_cache["external_cache_queries"] + t2_cache["external_cache_queries"],
            "external_cache_hits": t1_cache["external_cache_hits"] + t2_cache["external_cache_hits"],
        }
        total_cache["local_hit_rate_pct"] = (total_cache["local_cache_hits"] / total_cache["local_cache_queries"] * 100) if total_cache["local_cache_queries"] > 0 else 0
        total_cache["external_hit_rate_pct"] = (total_cache["external_cache_hits"] / total_cache["external_cache_queries"] * 100) if total_cache["external_cache_queries"] > 0 else 0
        merged_result["cache_stats"] = total_cache

        merged.append(merged_result)

    return {
        "merged_at": datetime.now().isoformat(),
        "total_results": len(merged),
        "results": merged
    }


def generate_summary(merged_data: dict) -> str:
    """Generate a text summary of results."""
    lines = []
    lines.append("=" * 100)
    lines.append("BENCHMARK RESULTS SUMMARY (Averaged across runs)")
    lines.append("=" * 100)
    lines.append("")

    # Group by workload
    by_workload = defaultdict(list)
    for r in merged_data["results"]:
        by_workload[r["workload"]].append(r)

    for workload in sorted(by_workload.keys()):
        results = by_workload[workload]
        first = results[0]

        lines.append(f"\n## Workload: {workload}")
        lines.append(f"   T1: {first['t1_input']}→{first['t1_output']}, T2: {first['t2_input']}→{first['t2_output']}")
        lines.append("-" * 100)

        # Header
        lines.append(f"{'Mode':<8} {'QPS':<5} {'Runs':<5} │ {'T1_TTFT':<10} {'T2_TTFT':<10} {'T2_TPOT':<10} │ {'T2_Cache%':<10} {'Success%':<10}")
        lines.append("-" * 100)

        for r in sorted(results, key=lambda x: (x["mode"], x["target_qps"])):
            mode = r["mode"]
            qps = r["target_qps"]
            runs = r["num_runs"]

            t1_ttft = r["turn1_metrics"].get("avg_ttft", 0)
            t2_ttft = r["turn2_metrics"].get("avg_ttft", 0)
            t2_tpot = r["turn2_metrics"].get("avg_tpot", 0)

            t2_cache_rate = r["turn2_metrics"]["cache"]["local_hit_rate_pct"]
            success_rate = r["success_count"] / r["sample_count"] * 100 if r["sample_count"] > 0 else 0

            lines.append(f"{mode:<8} {qps:<5.1f} {runs:<5} │ {t1_ttft:<10.1f} {t2_ttft:<10.1f} {t2_tpot:<10.2f} │ {t2_cache_rate:<10.1f} {success_rate:<10.1f}")

    # PPD vs PD comparison
    lines.append("\n" + "=" * 100)
    lines.append("PPD vs PD COMPARISON (T2 TTFT)")
    lines.append("=" * 100)

    ppd_results = {(r["workload"], r["target_qps"]): r for r in merged_data["results"] if r["mode"] == "ppd"}
    pd_results = {(r["workload"], r["target_qps"]): r for r in merged_data["results"] if r["mode"] == "pd"}

    lines.append(f"{'Workload':<10} {'QPS':<6} │ {'PPD_TTFT':<12} {'PD_TTFT':<12} │ {'Speedup':<10} {'Winner':<8}")
    lines.append("-" * 70)

    for key in sorted(ppd_results.keys()):
        if key in pd_results:
            workload, qps = key
            ppd_ttft = ppd_results[key]["turn2_metrics"].get("avg_ttft", 0)
            pd_ttft = pd_results[key]["turn2_metrics"].get("avg_ttft", 0)

            if ppd_ttft > 0 and pd_ttft > 0:
                speedup = pd_ttft / ppd_ttft
                winner = "PPD" if speedup > 1 else "PD"
                lines.append(f"{workload:<10} {qps:<6.1f} │ {ppd_ttft:<12.1f} {pd_ttft:<12.1f} │ {speedup:<10.2f}x {winner:<8}")

    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Merge benchmark results")
    parser.add_argument("--input-dir", type=str, default="results",
                        help="Directory containing result subdirectories (ppd/, pd/, replica/)")
    parser.add_argument("--output", type=str, default="results/merged_results.json",
                        help="Output file for merged results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory {input_dir} does not exist")
        return

    print(f"Loading results from: {input_dir}")
    print("-" * 50)
    results = load_results(input_dir)

    if not results:
        print("No results found!")
        return

    print(f"\nLoaded {len(results)} result entries")

    print("\nMerging results...")
    merged = merge_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged results saved to: {output_path}")

    # Generate and save summary
    summary = generate_summary(merged)
    summary_path = output_path.with_suffix(".txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to: {summary_path}")

    # Print summary
    print("\n" + summary)


if __name__ == "__main__":
    main()
