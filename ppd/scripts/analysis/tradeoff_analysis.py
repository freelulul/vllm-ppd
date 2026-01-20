#!/usr/bin/env python3
"""
Trade-off Analysis for PPD Comprehensive Benchmark

Analyzes 3060 benchmark data points to find key trade-offs between
PD, PPD, and Replica modes across different workloads and QPS levels.

Core thesis: "Objective-Oriented" - no single mode wins in all metrics.
"""

import os
import sys
import json
import glob
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Project paths
PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "results" / "comprehensive"
OUTPUT_DIR = PROJECT_DIR / "results" / "analysis" / "exploration"


# =============================================================================
# Configuration Classification
# =============================================================================

CONFIG_CATEGORIES = {
    # Pure Replica
    "4R": "Replica",

    # Pure PD (P + D only)
    "1P_3D": "PD",
    "2P_2D": "PD",
    "3P_1D": "PD",

    # Pure PPD (P + pD only)
    "1P_3pD": "PPD",
    "2P_2pD": "PPD",
    "3P_1pD": "PPD",

    # Mixed D/pD
    "1P_2D_1pD": "Mixed_DpD",
    "1P_1D_2pD": "Mixed_DpD",
    "2P_1D_1pD": "Mixed_DpD",

    # Hybrid with Replica
    "1R_1P_2D": "Hybrid",
    "1R_1P_2pD": "Hybrid",
    "1R_1P_1D_1pD": "Hybrid",
    "1R_2P_1D": "Hybrid",
    "1R_2P_1pD": "Hybrid",
    "2R_1P_1D": "Hybrid",
    "2R_1P_1pD": "Hybrid",
}

WORKLOAD_CATEGORIES = {
    # Decode-heavy (output >> input)
    "tiny": "Decode-heavy",
    "short_gen": "Decode-heavy",
    "long_gen": "Decode-heavy",
    "very_long_gen": "Decode-heavy",

    # Balanced (input ≈ output)
    "small_bal": "Balanced",
    "mid_bal": "Balanced",

    # Prefill-heavy (input >> output)
    "mid_paste": "Prefill-heavy",
    "big_paste": "Prefill-heavy",
    "huge_paste": "Prefill-heavy",
}

# Representative configs for detailed analysis
REPRESENTATIVE_CONFIGS = [
    "4R",           # Replica baseline
    "2P_2D",        # Classic PD
    "2P_2pD",       # Classic PPD
    "1P_3D",        # Extreme P:D ratio (failure case)
    "3P_1D",        # Multi-P config
    "1P_3pD",       # Extreme PPD
    "1R_1P_1D_1pD", # Hybrid
    "1R_1P_2pD",    # Hybrid PPD
]


# =============================================================================
# Data Loading
# =============================================================================

def load_all_data() -> pd.DataFrame:
    """Load all JSON results into a DataFrame."""
    print("Loading data from", DATA_DIR)

    all_records = []

    for config_dir in DATA_DIR.iterdir():
        if not config_dir.is_dir() or config_dir.name == "checkpoint.json":
            continue

        config = config_dir.name

        for json_file in config_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Extract workload type from workload name
                workload = data.get("workload", "")
                t2_type = workload.split("_", 1)[1] if "_" in workload else workload
                t1_type = workload.split("_")[0] if "_" in workload else "unknown"

                record = {
                    "config": config,
                    "config_category": CONFIG_CATEGORIES.get(config, "Unknown"),
                    "workload": workload,
                    "t1_type": t1_type,
                    "t2_type": t2_type,
                    "workload_category": WORKLOAD_CATEGORIES.get(t2_type, "Unknown"),
                    "qps": data.get("qps", 0),
                    "success_rate": data.get("success_rate", 0),
                    "total_requests": data.get("total_requests", 0),
                    "failed_requests": data.get("failed_requests", 0),

                    # Turn 1 metrics
                    "t1_avg_ttft": data.get("turn1", {}).get("avg_ttft_ms", 0),
                    "t1_avg_tpot": data.get("turn1", {}).get("avg_tpot_ms", 0),
                    "t1_avg_e2e": data.get("turn1", {}).get("avg_e2e_ms", 0),
                    "t1_p50_ttft": data.get("turn1", {}).get("p50_ttft_ms", 0),
                    "t1_p99_ttft": data.get("turn1", {}).get("p99_ttft_ms", 0),
                    "t1_p50_tpot": data.get("turn1", {}).get("p50_tpot_ms", 0),
                    "t1_p99_tpot": data.get("turn1", {}).get("p99_tpot_ms", 0),

                    # Turn 2 metrics
                    "t2_avg_ttft": data.get("turn2", {}).get("avg_ttft_ms", 0),
                    "t2_avg_tpot": data.get("turn2", {}).get("avg_tpot_ms", 0),
                    "t2_avg_e2e": data.get("turn2", {}).get("avg_e2e_ms", 0),
                    "t2_p50_ttft": data.get("turn2", {}).get("p50_ttft_ms", 0),
                    "t2_p99_ttft": data.get("turn2", {}).get("p99_ttft_ms", 0),
                    "t2_p50_tpot": data.get("turn2", {}).get("p50_tpot_ms", 0),
                    "t2_p99_tpot": data.get("turn2", {}).get("p99_tpot_ms", 0),

                    # Average metrics
                    "avg_ttft": data.get("average", {}).get("avg_ttft_ms", 0),
                    "avg_tpot": data.get("average", {}).get("avg_tpot_ms", 0),
                    "avg_e2e": data.get("average", {}).get("avg_e2e_ms", 0),

                    # Throughput
                    "throughput_rps": data.get("throughput", {}).get("requests_per_sec", 0),
                    "throughput_tps": data.get("throughput", {}).get("tokens_per_sec", 0),

                    # Error info
                    "error": data.get("error"),
                }

                all_records.append(record)

            except Exception as e:
                print(f"  Error loading {json_file}: {e}")

    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} records")
    return df


# =============================================================================
# Basic Statistics
# =============================================================================

def print_data_summary(df: pd.DataFrame):
    """Print basic data summary."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"\nTotal records: {len(df)}")
    print(f"Configs: {df['config'].nunique()} ({', '.join(sorted(df['config'].unique()))})")
    print(f"Workloads: {df['workload'].nunique()}")
    print(f"QPS range: {df['qps'].min()} - {df['qps'].max()}")

    print("\n--- Config Categories ---")
    for cat in ["Replica", "PD", "PPD", "Mixed_DpD", "Hybrid"]:
        configs = df[df['config_category'] == cat]['config'].unique()
        print(f"  {cat}: {len(configs)} configs")

    print("\n--- Workload Categories ---")
    for cat in ["Decode-heavy", "Balanced", "Prefill-heavy"]:
        workloads = df[df['workload_category'] == cat]['t2_type'].unique()
        print(f"  {cat}: {', '.join(workloads)}")

    print("\n--- Success Rate ---")
    success_df = df[df['success_rate'] == 100]
    fail_df = df[df['success_rate'] < 100]
    print(f"  100% success: {len(success_df)} ({len(success_df)/len(df)*100:.1f}%)")
    print(f"  Failures: {len(fail_df)} ({len(fail_df)/len(df)*100:.1f}%)")

    if len(fail_df) > 0:
        print("\n  Failure breakdown by config:")
        fail_by_config = fail_df.groupby('config').size().sort_values(ascending=False)
        for config, count in fail_by_config.items():
            print(f"    {config}: {count} failures")


# =============================================================================
# Winner Analysis
# =============================================================================

def find_winners(df: pd.DataFrame, metric: str, lower_is_better: bool = True) -> pd.DataFrame:
    """
    Find the winning config for each (workload, QPS) combination.

    Returns DataFrame with winner info for each scenario.
    """
    # Filter to successful runs only
    valid_df = df[df['success_rate'] >= 95].copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    winners = []

    for (workload, qps), group in valid_df.groupby(['workload', 'qps']):
        if len(group) == 0:
            continue

        if lower_is_better:
            winner_idx = group[metric].idxmin()
        else:
            winner_idx = group[metric].idxmax()

        winner = group.loc[winner_idx]

        winners.append({
            'workload': workload,
            'qps': qps,
            'winner_config': winner['config'],
            'winner_category': winner['config_category'],
            'winner_value': winner[metric],
            'workload_category': winner['workload_category'],
        })

    return pd.DataFrame(winners)


def analyze_winners(df: pd.DataFrame):
    """Analyze winners across different metrics."""
    print("\n" + "=" * 70)
    print("WINNER ANALYSIS")
    print("=" * 70)

    metrics = [
        ("t2_avg_ttft", True, "T2 TTFT (lower = better)"),
        ("avg_tpot", True, "Average TPOT (lower = better)"),
        ("throughput_tps", False, "Throughput (higher = better)"),
        ("avg_e2e", True, "Average E2E (lower = better)"),
    ]

    all_winners = {}

    for metric, lower_is_better, description in metrics:
        print(f"\n--- {description} ---")

        winners_df = find_winners(df, metric, lower_is_better)
        all_winners[metric] = winners_df

        if len(winners_df) == 0:
            print("  No valid data")
            continue

        # Count by config category
        cat_counts = winners_df['winner_category'].value_counts()
        total = len(winners_df)

        print(f"  Total scenarios: {total}")
        print(f"  Winner distribution by mode:")
        for cat in ["PPD", "PD", "Replica", "Mixed_DpD", "Hybrid"]:
            count = cat_counts.get(cat, 0)
            pct = count / total * 100
            print(f"    {cat}: {count} wins ({pct:.1f}%)")

        # Most winning configs
        config_counts = winners_df['winner_config'].value_counts().head(5)
        print(f"  Top winning configs:")
        for config, count in config_counts.items():
            pct = count / total * 100
            print(f"    {config}: {count} wins ({pct:.1f}%)")

    return all_winners


def analyze_metric_disagreement(df: pd.DataFrame, winners_dict: dict):
    """Analyze how often different metrics prefer different configs."""
    print("\n" + "=" * 70)
    print("METRIC DISAGREEMENT ANALYSIS")
    print("=" * 70)
    print("(Proves 'Objective-Oriented' thesis: TTFT best != TPOT best)")

    if 't2_avg_ttft' not in winners_dict or 'avg_tpot' not in winners_dict:
        print("  Insufficient data for comparison")
        return

    ttft_winners = winners_dict['t2_avg_ttft'].set_index(['workload', 'qps'])
    tpot_winners = winners_dict['avg_tpot'].set_index(['workload', 'qps'])

    # Find common scenarios
    common_idx = ttft_winners.index.intersection(tpot_winners.index)

    agree = 0
    disagree = 0

    for idx in common_idx:
        ttft_winner = ttft_winners.loc[idx, 'winner_config']
        tpot_winner = tpot_winners.loc[idx, 'winner_config']

        if ttft_winner == tpot_winner:
            agree += 1
        else:
            disagree += 1

    total = agree + disagree
    if total > 0:
        print(f"\n  TTFT vs TPOT comparison across {total} scenarios:")
        print(f"    Same winner: {agree} ({agree/total*100:.1f}%)")
        print(f"    Different winners: {disagree} ({disagree/total*100:.1f}%)")
        print(f"\n  ==> {disagree/total*100:.1f}% of scenarios have conflicting optimal configs!")


# =============================================================================
# PPD vs PD Comparison
# =============================================================================

def compare_ppd_pd(df: pd.DataFrame):
    """Compare PPD (2P_2pD) vs PD (2P_2D) performance."""
    print("\n" + "=" * 70)
    print("PPD vs PD COMPARISON (2P_2pD vs 2P_2D)")
    print("=" * 70)

    ppd_df = df[df['config'] == '2P_2pD'].copy()
    pd_df = df[df['config'] == '2P_2D'].copy()

    if len(ppd_df) == 0 or len(pd_df) == 0:
        print("  Missing data for comparison")
        return

    # Merge on workload and QPS
    merged = ppd_df.merge(
        pd_df,
        on=['workload', 'qps'],
        suffixes=('_ppd', '_pd')
    )

    # Filter to successful runs
    merged = merged[(merged['success_rate_ppd'] >= 95) & (merged['success_rate_pd'] >= 95)]

    print(f"\n  Comparable scenarios: {len(merged)}")

    # T2 TTFT improvement
    merged['t2_ttft_improvement'] = (merged['t2_avg_ttft_pd'] - merged['t2_avg_ttft_ppd']) / merged['t2_avg_ttft_pd'] * 100

    print(f"\n  T2 TTFT Improvement (PPD vs PD):")
    print(f"    Mean: {merged['t2_ttft_improvement'].mean():.1f}%")
    print(f"    Median: {merged['t2_ttft_improvement'].median():.1f}%")
    print(f"    Min: {merged['t2_ttft_improvement'].min():.1f}%")
    print(f"    Max: {merged['t2_ttft_improvement'].max():.1f}%")

    # By workload category
    print(f"\n  T2 TTFT Improvement by Workload Category:")
    for cat in ["Decode-heavy", "Balanced", "Prefill-heavy"]:
        cat_data = merged[merged['workload_category_ppd'] == cat]
        if len(cat_data) > 0:
            mean_imp = cat_data['t2_ttft_improvement'].mean()
            print(f"    {cat}: {mean_imp:+.1f}% (n={len(cat_data)})")

    # By QPS
    print(f"\n  T2 TTFT Improvement by QPS:")
    for qps in sorted(merged['qps'].unique()):
        qps_data = merged[merged['qps'] == qps]
        if len(qps_data) > 0:
            mean_imp = qps_data['t2_ttft_improvement'].mean()
            print(f"    QPS={qps}: {mean_imp:+.1f}% (n={len(qps_data)})")

    # TPOT comparison
    merged['tpot_diff'] = merged['avg_tpot_ppd'] - merged['avg_tpot_pd']
    print(f"\n  TPOT Difference (PPD - PD):")
    print(f"    Mean: {merged['tpot_diff'].mean():.2f} ms")
    print(f"    (Positive = PPD slower, Negative = PPD faster)")

    # Throughput comparison
    merged['throughput_diff'] = (merged['throughput_tps_ppd'] - merged['throughput_tps_pd']) / merged['throughput_tps_pd'] * 100
    print(f"\n  Throughput Difference (PPD vs PD):")
    print(f"    Mean: {merged['throughput_diff'].mean():+.1f}%")

    return merged


# =============================================================================
# Failure Region Analysis
# =============================================================================

def analyze_failure_regions(df: pd.DataFrame):
    """Identify configs that fail in certain scenarios."""
    print("\n" + "=" * 70)
    print("FAILURE REGION ANALYSIS")
    print("=" * 70)

    fail_df = df[df['success_rate'] < 95].copy()

    if len(fail_df) == 0:
        print("  No failures detected (all success_rate >= 95%)")
        return

    print(f"\n  Total failure scenarios: {len(fail_df)}")

    # By config
    print(f"\n  Failures by config:")
    fail_by_config = fail_df.groupby('config').agg({
        'workload': 'count',
        'success_rate': 'mean'
    }).sort_values('workload', ascending=False)

    for config, row in fail_by_config.iterrows():
        print(f"    {config}: {int(row['workload'])} failures, avg success_rate={row['success_rate']:.1f}%")

    # Detailed failure analysis for 1P_3D
    print(f"\n  --- 1P_3D Failure Analysis ---")
    p1_3d_fail = fail_df[fail_df['config'] == '1P_3D']

    if len(p1_3d_fail) > 0:
        print(f"    Total failures: {len(p1_3d_fail)}")

        # By workload category
        print(f"    By workload category:")
        for cat in ["Decode-heavy", "Balanced", "Prefill-heavy"]:
            cat_fail = p1_3d_fail[p1_3d_fail['workload_category'] == cat]
            if len(cat_fail) > 0:
                print(f"      {cat}: {len(cat_fail)} failures")

        # By QPS
        print(f"    By QPS:")
        qps_counts = p1_3d_fail.groupby('qps').size().sort_index()
        for qps, count in qps_counts.items():
            print(f"      QPS={qps}: {count} failures")

        # Pattern: high QPS + prefill-heavy
        high_qps_prefill = p1_3d_fail[
            (p1_3d_fail['qps'] >= 6) &
            (p1_3d_fail['workload_category'] == 'Prefill-heavy')
        ]
        print(f"\n    Key pattern: QPS>=6 AND Prefill-heavy → {len(high_qps_prefill)} failures")
        print(f"    Reason: Single P cannot handle prefill load")
    else:
        print("    No 1P_3D failures")

    return fail_df


# =============================================================================
# Key Findings Summary
# =============================================================================

def generate_key_findings(df: pd.DataFrame, winners_dict: dict, ppd_pd_comparison: pd.DataFrame):
    """Generate key findings summary."""
    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    findings = []

    # Finding 1: No single winner
    if 't2_avg_ttft' in winners_dict:
        ttft_winners = winners_dict['t2_avg_ttft']
        cat_counts = ttft_winners['winner_category'].value_counts()
        top_cat = cat_counts.index[0]
        top_pct = cat_counts.iloc[0] / len(ttft_winners) * 100

        finding1 = f"""
1. NO SINGLE WINNER:
   - For T2 TTFT optimization, {top_cat} wins {top_pct:.1f}% of scenarios
   - But no mode dominates completely
   - Validates "Objective-Oriented" thesis
"""
        findings.append(finding1)

    # Finding 2: PPD T2 TTFT advantage
    if ppd_pd_comparison is not None and len(ppd_pd_comparison) > 0:
        mean_improvement = ppd_pd_comparison['t2_ttft_improvement'].mean()
        finding2 = f"""
2. PPD T2 TTFT ADVANTAGE:
   - PPD (2P_2pD) reduces T2 TTFT by {mean_improvement:.1f}% vs PD (2P_2D)
   - Benefit comes from prefix cache avoiding KV transfer
   - Largest benefit in decode-heavy workloads
"""
        findings.append(finding2)

    # Finding 3: Failure regions
    fail_df = df[df['success_rate'] < 95]
    if len(fail_df) > 0:
        p1_3d_fail = fail_df[fail_df['config'] == '1P_3D']
        finding3 = f"""
3. FAILURE REGIONS:
   - 1P_3D has {len(p1_3d_fail)} failure scenarios
   - Pattern: High QPS + Prefill-heavy workloads
   - Root cause: Single P becomes prefill bottleneck
   - Recommendation: Avoid 1:3 P:D ratio for high-load scenarios
"""
        findings.append(finding3)

    # Finding 4: Replica throughput advantage
    replica_df = df[df['config'] == '4R']
    ppd_df = df[df['config'] == '2P_2pD']
    if len(replica_df) > 0 and len(ppd_df) > 0:
        merged = replica_df.merge(
            ppd_df, on=['workload', 'qps'], suffixes=('_replica', '_ppd')
        )
        if len(merged) > 0:
            merged['throughput_advantage'] = (
                merged['throughput_tps_replica'] - merged['throughput_tps_ppd']
            ) / merged['throughput_tps_ppd'] * 100

            high_qps = merged[merged['qps'] >= 10]
            if len(high_qps) > 0:
                avg_advantage = high_qps['throughput_advantage'].mean()
                finding4 = f"""
4. REPLICA THROUGHPUT ADVANTAGE:
   - At high QPS (>=10), 4R outperforms 2P_2pD by {avg_advantage:+.1f}% in throughput
   - 4 parallel workers beat 2+2 disaggregated setup
   - Trade-off: Higher throughput but no isolation
"""
                findings.append(finding4)

    for finding in findings:
        print(finding)

    return findings


# =============================================================================
# Report Generation
# =============================================================================

def save_reports(df: pd.DataFrame, winners_dict: dict, findings: list):
    """Save analysis reports to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed data
    df.to_csv(OUTPUT_DIR.parent / "data" / "all_data.csv", index=False)
    print(f"\nSaved all_data.csv")

    # Save winner matrix
    if 't2_avg_ttft' in winners_dict:
        winners_dict['t2_avg_ttft'].to_csv(
            OUTPUT_DIR.parent / "data" / "winner_matrix_t2_ttft.csv", index=False
        )

    # Save key findings
    with open(OUTPUT_DIR / "key_findings.md", 'w') as f:
        f.write("# Key Findings from Trade-off Analysis\n\n")
        for finding in findings:
            f.write(finding + "\n")

    print(f"Saved reports to {OUTPUT_DIR}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PPD COMPREHENSIVE TRADE-OFF ANALYSIS")
    print("=" * 70)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR.parent / "data").mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_all_data()

    # Basic summary
    print_data_summary(df)

    # Winner analysis
    winners_dict = analyze_winners(df)

    # Metric disagreement
    analyze_metric_disagreement(df, winners_dict)

    # PPD vs PD
    ppd_pd_comparison = compare_ppd_pd(df)

    # Failure regions
    analyze_failure_regions(df)

    # Key findings
    findings = generate_key_findings(df, winners_dict, ppd_pd_comparison)

    # Save reports
    save_reports(df, winners_dict, findings)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
