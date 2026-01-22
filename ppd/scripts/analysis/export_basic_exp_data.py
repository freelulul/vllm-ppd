#!/usr/bin/env python3
"""
Export all basic_exp results to a comprehensive Excel file.

Exports all configs, QPS points, and metrics for both ShareGPT and WildChat datasets.
Can be re-run after more data is collected.

Usage:
    python scripts/analysis/export_basic_exp_data.py
    python scripts/analysis/export_basic_exp_data.py --output results/analysis/basic_exp_data.xlsx
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Try to import pandas and openpyxl
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not installed. Will output CSV instead of Excel.")

# Project root
PROJECT_DIR = Path(__file__).parent.parent.parent

# All 17 expected configurations
ALL_CONFIGS = [
    # Pure Replica
    "4R",
    # Pure PD
    "1P_3D", "2P_2D", "3P_1D",
    # Pure PPD
    "1P_3pD", "2P_2pD", "3P_1pD",
    # Mixed D/pD
    "1P_2D_1pD", "1P_1D_2pD", "2P_1D_1pD",
    # Hybrid (R + P + D/pD)
    "1R_1P_2D", "1R_1P_2pD", "1R_1P_1D_1pD",
    "1R_2P_1D", "1R_2P_1pD",
    "2R_1P_1D", "2R_1P_1pD",
]

# Expected QPS points
QPS_POINTS = [2.0, 4.0, 8.0, 12.0]

# Datasets
DATASETS = ["sharegpt", "wildchat"]


def load_result_file(filepath: Path) -> Optional[dict]:
    """Load a single basic_exp result file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def extract_metrics(data: dict) -> dict:
    """Extract all metrics from a basic_exp result."""
    metrics = {
        # Basic info
        "config": data.get("config", ""),
        "ppd_enabled": data.get("ppd_enabled", False),
        "qps": data.get("qps", 0),
        "num_conversations": data.get("num_conversations", 0),
        "duration_sec": data.get("duration_sec", 0),
        "timestamp": data.get("timestamp", ""),
    }

    # Counts
    counts = data.get("counts", {})
    metrics["total_conversations"] = counts.get("total_conversations", 0)
    metrics["successful_conversations"] = counts.get("successful_conversations", 0)
    metrics["total_turns"] = counts.get("total_turns", 0)
    metrics["turn1_count"] = counts.get("turn1_count", 0)
    metrics["turn2plus_count"] = counts.get("turn2plus_count", 0)
    metrics["successful_turns"] = counts.get("successful_turns", 0)

    # Success rates
    sr = data.get("success_rate", {})
    metrics["success_rate_turns"] = sr.get("turns", 0)
    metrics["success_rate_conversations"] = sr.get("conversations", 0)

    # Throughput
    tp = data.get("throughput", {})
    metrics["conversations_per_sec"] = tp.get("conversations_per_sec", 0)
    metrics["requests_per_sec"] = tp.get("requests_per_sec", 0)
    metrics["tokens_per_sec"] = tp.get("tokens_per_sec", 0)

    # Turn 1 metrics
    t1 = data.get("turn1", {})
    metrics["t1_count"] = t1.get("count", 0)
    metrics["t1_avg_ttft_ms"] = t1.get("avg_ttft_ms", 0)
    metrics["t1_p50_ttft_ms"] = t1.get("p50_ttft_ms", 0)
    metrics["t1_p90_ttft_ms"] = t1.get("p90_ttft_ms", 0)
    metrics["t1_p99_ttft_ms"] = t1.get("p99_ttft_ms", 0)
    metrics["t1_avg_tpot_ms"] = t1.get("avg_tpot_ms", 0)
    metrics["t1_p50_tpot_ms"] = t1.get("p50_tpot_ms", 0)
    metrics["t1_p90_tpot_ms"] = t1.get("p90_tpot_ms", 0)
    metrics["t1_p99_tpot_ms"] = t1.get("p99_tpot_ms", 0)
    metrics["t1_avg_e2e_ms"] = t1.get("avg_e2e_ms", 0)
    metrics["t1_p50_e2e_ms"] = t1.get("p50_e2e_ms", 0)
    metrics["t1_p90_e2e_ms"] = t1.get("p90_e2e_ms", 0)
    metrics["t1_p99_e2e_ms"] = t1.get("p99_e2e_ms", 0)

    # Turn 2+ metrics (main focus)
    t2 = data.get("turn2plus", data.get("turn2plus_stats", {}))
    metrics["t2plus_count"] = t2.get("count", 0)
    metrics["t2plus_avg_ttft_ms"] = t2.get("avg_ttft_ms", 0)
    metrics["t2plus_p50_ttft_ms"] = t2.get("p50_ttft_ms", 0)
    metrics["t2plus_p90_ttft_ms"] = t2.get("p90_ttft_ms", 0)
    metrics["t2plus_p99_ttft_ms"] = t2.get("p99_ttft_ms", 0)
    metrics["t2plus_avg_tpot_ms"] = t2.get("avg_tpot_ms", 0)
    metrics["t2plus_p50_tpot_ms"] = t2.get("p50_tpot_ms", 0)
    metrics["t2plus_p90_tpot_ms"] = t2.get("p90_tpot_ms", 0)
    metrics["t2plus_p99_tpot_ms"] = t2.get("p99_tpot_ms", 0)
    metrics["t2plus_avg_e2e_ms"] = t2.get("avg_e2e_ms", 0)
    metrics["t2plus_p50_e2e_ms"] = t2.get("p50_e2e_ms", 0)
    metrics["t2plus_p90_e2e_ms"] = t2.get("p90_e2e_ms", 0)
    metrics["t2plus_p99_e2e_ms"] = t2.get("p99_e2e_ms", 0)

    # All turns metrics
    at = data.get("all_turns", {})
    metrics["all_count"] = at.get("count", 0)
    metrics["all_avg_ttft_ms"] = at.get("avg_ttft_ms", 0)
    metrics["all_p50_ttft_ms"] = at.get("p50_ttft_ms", 0)
    metrics["all_p90_ttft_ms"] = at.get("p90_ttft_ms", 0)
    metrics["all_p99_ttft_ms"] = at.get("p99_ttft_ms", 0)
    metrics["all_avg_tpot_ms"] = at.get("avg_tpot_ms", 0)
    metrics["all_p50_tpot_ms"] = at.get("p50_tpot_ms", 0)
    metrics["all_p90_tpot_ms"] = at.get("p90_tpot_ms", 0)
    metrics["all_p99_tpot_ms"] = at.get("p99_tpot_ms", 0)
    metrics["all_avg_e2e_ms"] = at.get("avg_e2e_ms", 0)
    metrics["all_p50_e2e_ms"] = at.get("p50_e2e_ms", 0)
    metrics["all_p90_e2e_ms"] = at.get("p90_e2e_ms", 0)
    metrics["all_p99_e2e_ms"] = at.get("p99_e2e_ms", 0)

    # SLO attainment
    slo = data.get("slo_attainment", {})
    metrics["slo_ttft_100ms"] = slo.get("ttft_100ms", 0)
    metrics["slo_ttft_200ms"] = slo.get("ttft_200ms", 0)
    metrics["slo_ttft_500ms"] = slo.get("ttft_500ms", 0)
    metrics["slo_e2e_5000ms"] = slo.get("e2e_5000ms", 0)
    metrics["slo_e2e_10000ms"] = slo.get("e2e_10000ms", 0)

    return metrics


def collect_all_results(results_base: Path) -> List[dict]:
    """Collect all basic_exp results from all datasets."""
    all_results = []

    for dataset in DATASETS:
        dataset_dir = results_base / dataset
        if not dataset_dir.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            continue

        print(f"\nCollecting {dataset} results...")

        for config in ALL_CONFIGS:
            config_dir = dataset_dir / config
            if not config_dir.exists():
                print(f"  Missing: {config}")
                # Add placeholder for missing config
                for qps in QPS_POINTS:
                    all_results.append({
                        "dataset": dataset,
                        "config": config,
                        "qps": qps,
                        "status": "MISSING",
                        **{k: None for k in extract_metrics({}).keys() if k not in ["config", "qps"]}
                    })
                continue

            # Find all result files
            result_files = list(config_dir.glob("*.json"))
            found_qps = set()

            for filepath in result_files:
                data = load_result_file(filepath)
                if data is None:
                    continue

                qps = data.get("qps", 0)
                found_qps.add(qps)

                metrics = extract_metrics(data)
                metrics["dataset"] = dataset

                # Determine status based on success rate
                success_rate = metrics.get("success_rate_turns", 0)
                t2_count = metrics.get("t2plus_count", 0)

                if success_rate >= 95 and t2_count >= 100:
                    metrics["status"] = "OK"
                elif success_rate >= 50 and t2_count >= 50:
                    metrics["status"] = "PARTIAL"
                elif t2_count > 0:
                    metrics["status"] = "FAILED"
                else:
                    metrics["status"] = "EMPTY"

                all_results.append(metrics)
                print(f"  {config} QPS={qps}: {metrics['status']} (T2+: {t2_count}, success: {success_rate:.1f}%)")

            # Check for missing QPS points
            for qps in QPS_POINTS:
                if qps not in found_qps:
                    all_results.append({
                        "dataset": dataset,
                        "config": config,
                        "qps": qps,
                        "status": "MISSING",
                        **{k: None for k in extract_metrics({}).keys() if k not in ["config", "qps"]}
                    })
                    print(f"  {config} QPS={qps}: MISSING")

    return all_results


def generate_summary(results: List[dict]) -> List[dict]:
    """Generate summary statistics for PPD vs PD comparison."""
    summary = []

    # Define PD/PPD pairs
    pairs = [
        ("1P_3D", "1P_3pD"),
        ("2P_2D", "2P_2pD"),
        ("3P_1D", "3P_1pD"),
        ("1R_1P_2D", "1R_1P_2pD"),
        ("1R_2P_1D", "1R_2P_1pD"),
        ("2R_1P_1D", "2R_1P_1pD"),
    ]

    for dataset in DATASETS:
        dataset_results = [r for r in results if r.get("dataset") == dataset]

        for pd_config, ppd_config in pairs:
            for qps in QPS_POINTS:
                pd_data = next((r for r in dataset_results
                               if r.get("config") == pd_config and r.get("qps") == qps), None)
                ppd_data = next((r for r in dataset_results
                                if r.get("config") == ppd_config and r.get("qps") == qps), None)

                row = {
                    "dataset": dataset,
                    "pd_config": pd_config,
                    "ppd_config": ppd_config,
                    "qps": qps,
                }

                if pd_data and ppd_data:
                    pd_ttft = pd_data.get("t2plus_avg_ttft_ms", 0) or 0
                    ppd_ttft = ppd_data.get("t2plus_avg_ttft_ms", 0) or 0
                    pd_tpot = pd_data.get("t2plus_avg_tpot_ms", 0) or 0
                    ppd_tpot = ppd_data.get("t2plus_avg_tpot_ms", 0) or 0
                    pd_e2e = pd_data.get("t2plus_avg_e2e_ms", 0) or 0
                    ppd_e2e = ppd_data.get("t2plus_avg_e2e_ms", 0) or 0

                    row["pd_status"] = pd_data.get("status", "UNKNOWN")
                    row["ppd_status"] = ppd_data.get("status", "UNKNOWN")
                    row["pd_t2plus_count"] = pd_data.get("t2plus_count", 0)
                    row["ppd_t2plus_count"] = ppd_data.get("t2plus_count", 0)

                    row["pd_ttft_avg"] = pd_ttft
                    row["ppd_ttft_avg"] = ppd_ttft
                    row["ttft_improvement_pct"] = ((pd_ttft - ppd_ttft) / pd_ttft * 100) if pd_ttft > 0 else None

                    row["pd_tpot_avg"] = pd_tpot
                    row["ppd_tpot_avg"] = ppd_tpot
                    row["tpot_change_pct"] = ((ppd_tpot - pd_tpot) / pd_tpot * 100) if pd_tpot > 0 else None

                    row["pd_e2e_avg"] = pd_e2e
                    row["ppd_e2e_avg"] = ppd_e2e
                    row["e2e_improvement_pct"] = ((pd_e2e - ppd_e2e) / pd_e2e * 100) if pd_e2e > 0 else None

                    # PPD score
                    ttft_imp = row["ttft_improvement_pct"] or 0
                    tpot_deg = row["tpot_change_pct"] or 0
                    row["ppd_score"] = (ttft_imp / 100) - (tpot_deg / 100)
                    row["ppd_wins"] = row["ppd_score"] > 0 if row["pd_status"] == "OK" and row["ppd_status"] == "OK" else None
                else:
                    row["pd_status"] = pd_data.get("status", "MISSING") if pd_data else "MISSING"
                    row["ppd_status"] = ppd_data.get("status", "MISSING") if ppd_data else "MISSING"

                summary.append(row)

    return summary


def export_to_excel(results: List[dict], summary: List[dict], output_path: Path):
    """Export results to Excel file with multiple sheets."""
    if not HAS_PANDAS:
        # Fallback to CSV
        df = pd.DataFrame(results) if HAS_PANDAS else None
        if df is None:
            import csv
            csv_path = output_path.with_suffix('.csv')
            with open(csv_path, 'w', newline='') as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
            print(f"\nExported to CSV: {csv_path}")
            return

    # Create DataFrames
    df_all = pd.DataFrame(results)
    df_summary = pd.DataFrame(summary)

    # Separate by dataset
    df_sharegpt = df_all[df_all['dataset'] == 'sharegpt'].copy()
    df_wildchat = df_all[df_all['dataset'] == 'wildchat'].copy()

    # Create pivot tables for easier viewing
    pivot_cols = ['t2plus_avg_ttft_ms', 't2plus_avg_tpot_ms', 't2plus_avg_e2e_ms', 'success_rate_turns', 't2plus_count']

    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # All data sheet
        df_all.to_excel(writer, sheet_name='All_Data', index=False)

        # Dataset-specific sheets
        if not df_sharegpt.empty:
            df_sharegpt.to_excel(writer, sheet_name='ShareGPT_Raw', index=False)

            # Pivot table for ShareGPT
            for metric in ['t2plus_avg_ttft_ms', 't2plus_avg_tpot_ms']:
                try:
                    pivot = df_sharegpt.pivot(index='config', columns='qps', values=metric)
                    pivot.to_excel(writer, sheet_name=f'SG_{metric.replace("t2plus_avg_", "").replace("_ms", "").upper()}')
                except:
                    pass

        if not df_wildchat.empty:
            df_wildchat.to_excel(writer, sheet_name='WildChat_Raw', index=False)

            # Pivot table for WildChat
            for metric in ['t2plus_avg_ttft_ms', 't2plus_avg_tpot_ms']:
                try:
                    pivot = df_wildchat.pivot(index='config', columns='qps', values=metric)
                    pivot.to_excel(writer, sheet_name=f'WC_{metric.replace("t2plus_avg_", "").replace("_ms", "").upper()}')
                except:
                    pass

        # PPD vs PD Summary
        df_summary.to_excel(writer, sheet_name='PPD_vs_PD_Summary', index=False)

        # Failed/Missing tests
        df_issues = df_all[df_all['status'].isin(['FAILED', 'MISSING', 'PARTIAL', 'EMPTY'])].copy()
        if not df_issues.empty:
            df_issues.to_excel(writer, sheet_name='Issues', index=False)

    print(f"\nExported to Excel: {output_path}")
    print(f"  Sheets: All_Data, ShareGPT_Raw, WildChat_Raw, SG_TTFT, SG_TPOT, WC_TTFT, WC_TPOT, PPD_vs_PD_Summary, Issues")


def print_status_summary(results: List[dict]):
    """Print a summary of data status."""
    print("\n" + "=" * 80)
    print("DATA STATUS SUMMARY")
    print("=" * 80)

    for dataset in DATASETS:
        dataset_results = [r for r in results if r.get("dataset") == dataset]

        ok_count = sum(1 for r in dataset_results if r.get("status") == "OK")
        partial_count = sum(1 for r in dataset_results if r.get("status") == "PARTIAL")
        failed_count = sum(1 for r in dataset_results if r.get("status") == "FAILED")
        missing_count = sum(1 for r in dataset_results if r.get("status") == "MISSING")
        empty_count = sum(1 for r in dataset_results if r.get("status") == "EMPTY")
        total = len(dataset_results)

        print(f"\n{dataset.upper()}:")
        print(f"  OK:      {ok_count:3d} / {total}")
        print(f"  PARTIAL: {partial_count:3d} / {total}")
        print(f"  FAILED:  {failed_count:3d} / {total}")
        print(f"  MISSING: {missing_count:3d} / {total}")
        print(f"  EMPTY:   {empty_count:3d} / {total}")

        # List issues
        issues = [r for r in dataset_results if r.get("status") in ["FAILED", "MISSING", "PARTIAL", "EMPTY"]]
        if issues:
            print(f"\n  Issues to fix:")
            for r in sorted(issues, key=lambda x: (x.get("config", ""), x.get("qps", 0))):
                print(f"    {r.get('config'):<15} QPS={r.get('qps'):>4} -> {r.get('status')}")


def main():
    parser = argparse.ArgumentParser(description="Export basic_exp data to Excel")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Base results directory")
    parser.add_argument("--output", type=str, default="results/analysis/basic_exp_data.xlsx",
                        help="Output Excel file path")

    args = parser.parse_args()

    results_base = PROJECT_DIR / args.results_dir
    output_path = PROJECT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("basic_exp DATA EXPORT")
    print("=" * 80)
    print(f"Results directory: {results_base}")
    print(f"Output file: {output_path}")
    print(f"Expected configs: {len(ALL_CONFIGS)}")
    print(f"Expected QPS points: {QPS_POINTS}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Collect all results
    results = collect_all_results(results_base)

    # Generate summary
    summary = generate_summary(results)

    # Print status summary
    print_status_summary(results)

    # Export
    export_to_excel(results, summary, output_path)

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
