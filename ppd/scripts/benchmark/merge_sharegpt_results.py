#!/usr/bin/env python3
"""
Merge ShareGPT and WildChat benchmark results from all batches.
Generates summary CSV and JSON files for analysis.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import csv
from datetime import datetime


def load_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all result files from a directory."""
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Warning: Directory not found: {results_dir}")
        return results

    for config_dir in results_path.iterdir():
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name
        results[config_name] = {}

        for json_file in config_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                qps = data.get('qps', 'unknown')
                results[config_name][f"qps_{qps}"] = data
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

    return results


def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a single result."""
    metrics = {
        'config': result.get('config', 'unknown'),
        'qps': result.get('qps', 0),
        'num_conversations': result.get('num_conversations', 0),
        'total_turns': result.get('total_turns', 0),
        'success_count': result.get('success_count', 0),
        'failed_count': result.get('failed_count', 0),
    }

    # Turn 2+ metrics (main focus)
    # Note: sharegpt_benchmark.py uses flat keys like 'avg_ttft_ms' not nested 'ttft_ms.avg'
    t2plus = result.get('turn2plus', result.get('turn2plus_stats', {}))
    metrics['t2plus_ttft_avg'] = t2plus.get('avg_ttft_ms', 0)
    metrics['t2plus_ttft_p50'] = t2plus.get('p50_ttft_ms', 0)
    metrics['t2plus_ttft_p99'] = t2plus.get('p99_ttft_ms', 0)
    metrics['t2plus_tpot_avg'] = t2plus.get('avg_tpot_ms', 0)
    metrics['t2plus_tpot_p50'] = t2plus.get('p50_tpot_ms', 0)
    metrics['t2plus_tpot_p99'] = t2plus.get('p99_tpot_ms', 0)
    metrics['t2plus_e2e_avg'] = t2plus.get('avg_e2e_ms', 0)
    metrics['t2plus_e2e_p50'] = t2plus.get('p50_e2e_ms', 0)
    metrics['t2plus_e2e_p99'] = t2plus.get('p99_e2e_ms', 0)

    # Throughput
    throughput = result.get('throughput', {})
    metrics['requests_per_sec'] = throughput.get('requests_per_sec', 0)
    metrics['tokens_per_sec'] = throughput.get('tokens_per_sec', 0)

    return metrics


def merge_and_save(sharegpt_results: Dict, wildchat_results: Dict, output_dir: str):
    """Merge results and save to CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare merged data
    all_metrics = []
    merged_json = {
        'generated_at': datetime.now().isoformat(),
        'sharegpt': {},
        'wildchat': {}
    }

    # Process ShareGPT results
    for config, qps_results in sharegpt_results.items():
        merged_json['sharegpt'][config] = qps_results
        for qps_key, result in qps_results.items():
            metrics = extract_metrics(result)
            metrics['dataset'] = 'sharegpt'
            all_metrics.append(metrics)

    # Process WildChat results
    for config, qps_results in wildchat_results.items():
        merged_json['wildchat'][config] = qps_results
        for qps_key, result in qps_results.items():
            metrics = extract_metrics(result)
            metrics['dataset'] = 'wildchat'
            all_metrics.append(metrics)

    # Save merged JSON
    json_path = output_path / 'all_results.json'
    with open(json_path, 'w') as f:
        json.dump(merged_json, f, indent=2)
    print(f"Saved merged JSON: {json_path}")

    # Save CSV
    if all_metrics:
        csv_path = output_path / 'all_results.csv'
        fieldnames = list(all_metrics[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"Saved CSV: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Benchmark Results Summary")
    print("=" * 60)

    print(f"\nShareGPT Results: {len(sharegpt_results)} configs")
    for config in sorted(sharegpt_results.keys()):
        qps_count = len(sharegpt_results[config])
        print(f"  {config}: {qps_count} QPS points")

    print(f"\nWildChat Results: {len(wildchat_results)} configs")
    for config in sorted(wildchat_results.keys()):
        qps_count = len(wildchat_results[config])
        print(f"  {config}: {qps_count} QPS points")

    # Generate comparison table
    print("\n" + "=" * 60)
    print("Turn 2+ TTFT Comparison (avg, ms)")
    print("=" * 60)
    print(f"{'Config':<15} {'QPS':>5} {'ShareGPT':>12} {'WildChat':>12} {'Diff':>10}")
    print("-" * 60)

    all_configs = set(sharegpt_results.keys()) | set(wildchat_results.keys())
    for config in sorted(all_configs):
        sg_qps = sharegpt_results.get(config, {})
        wc_qps = wildchat_results.get(config, {})

        all_qps_keys = set(sg_qps.keys()) | set(wc_qps.keys())
        for qps_key in sorted(all_qps_keys, key=lambda x: float(x.replace('qps_', ''))):
            sg_data = sg_qps.get(qps_key, {})
            wc_data = wc_qps.get(qps_key, {})
            sg_t2 = sg_data.get('turn2plus', sg_data.get('turn2plus_stats', {}))
            wc_t2 = wc_data.get('turn2plus', wc_data.get('turn2plus_stats', {}))
            sg_ttft = sg_t2.get('avg_ttft_ms', 0)
            wc_ttft = wc_t2.get('avg_ttft_ms', 0)

            qps_val = qps_key.replace('qps_', '')
            diff = ""
            if sg_ttft > 0 and wc_ttft > 0:
                diff_pct = (wc_ttft - sg_ttft) / sg_ttft * 100
                diff = f"{diff_pct:+.1f}%"

            print(f"{config:<15} {qps_val:>5} {sg_ttft:>12.1f} {wc_ttft:>12.1f} {diff:>10}")

    return merged_json


def main():
    """Main function."""
    base_dir = Path("/net/projects2/ds3lab/zongzel/vllm/ppd")

    sharegpt_dir = base_dir / "results" / "sharegpt"
    wildchat_dir = base_dir / "results" / "wildchat"
    output_dir = base_dir / "results" / "analysis"

    print("Loading ShareGPT results...")
    sharegpt_results = load_results(str(sharegpt_dir))

    print("Loading WildChat results...")
    wildchat_results = load_results(str(wildchat_dir))

    if not sharegpt_results and not wildchat_results:
        print("No results found. Run benchmarks first.")
        sys.exit(1)

    merged = merge_and_save(sharegpt_results, wildchat_results, str(output_dir))

    print("\n" + "=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print(f"\nTo generate analysis figures:")
    print(f"  python scripts/analysis/ppd_analysis.py \\")
    print(f"      --sharegpt-dir results/sharegpt \\")
    print(f"      --output-dir results/analysis/figures/sharegpt")
    print(f"\n  python scripts/analysis/ppd_analysis.py \\")
    print(f"      --sharegpt-dir results/wildchat \\")
    print(f"      --output-dir results/analysis/figures/wildchat")


if __name__ == "__main__":
    main()
