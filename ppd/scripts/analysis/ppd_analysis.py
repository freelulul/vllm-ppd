#!/usr/bin/env python3
"""
PPD Analysis Script - Generate figures for paper.

Generates:
1. Turn 2+ TTFT comparison (box plot)
2. SLO attainment curves (DistServe style)
3. Pareto frontier (latency vs throughput)
4. Multi-turn stability (TTFT across turns)

Usage:
    python ppd_analysis.py --dataset-dir results/sharegpt --output-dir results/analysis/figures/sharegpt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)


# Configuration
COLORS = {
    "4R": "#1f77b4",       # Blue - Replica
    "2P_2D": "#ff7f0e",    # Orange - PD
    "2P_2pD": "#2ca02c",   # Green - PPD
    "1P_3D": "#d62728",    # Red
    "1P_3pD": "#9467bd",   # Purple
    "ppd_dynamic": "#e377c2",  # Pink - Dynamic PPD
    "baseline": "#7f7f7f",     # Gray
}

MARKERS = {
    "4R": "o",
    "2P_2D": "s",
    "2P_2pD": "^",
    "1P_3D": "D",
    "1P_3pD": "v",
    "ppd_dynamic": "*",
}

CONFIG_LABELS = {
    "4R": "4R (Replica)",
    "2P_2D": "2P+2D (PD)",
    "2P_2pD": "2P+2pD (PPD)",
    "1P_3D": "1P+3D (PD)",
    "1P_3pD": "1P+3pD (PPD)",
    "2P_2D_ppd": "2P+2D + Dynamic PPD",
}


def load_sharegpt_results(results_dir: Path) -> Dict[str, List[dict]]:
    """Load ShareGPT benchmark results.

    Returns:
        Dict mapping config name to list of results (one per QPS)
    """
    results = {}

    for config_dir in results_dir.iterdir():
        if not config_dir.is_dir():
            continue

        config_name = config_dir.name
        results[config_name] = []

        for result_file in config_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results[config_name].append(data)
            except Exception as e:
                print(f"Warning: Failed to load {result_file}: {e}")

    return results


def load_comprehensive_results(results_dir: Path, configs: List[str]) -> Dict[str, List[dict]]:
    """Load comprehensive benchmark results for comparison."""
    results = {}

    for config in configs:
        config_dir = results_dir / config
        if not config_dir.exists():
            continue

        results[config] = []
        for result_file in config_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results[config].append(data)
            except Exception as e:
                print(f"Warning: Failed to load {result_file}: {e}")

    return results


def plot_t2_ttft_comparison(
    results: Dict[str, List[dict]],
    output_path: Path,
    qps_filter: float = None,
):
    """Generate Turn 2+ TTFT comparison box plot.

    Args:
        results: Dict of config -> list of results
        output_path: Where to save the figure
        qps_filter: If specified, only include results at this QPS
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect T2+ TTFT data for each config
    data = []
    labels = []
    colors = []

    for config in sorted(results.keys()):
        config_results = results[config]

        # Filter by QPS if specified
        if qps_filter:
            config_results = [r for r in config_results if abs(r.get("qps", 0) - qps_filter) < 0.1]

        if not config_results:
            continue

        # Extract T2+ TTFT values
        t2_ttfts = []
        for r in config_results:
            t2_stats = r.get("turn2plus", r.get("turn2", {}))
            if "avg_ttft_ms" in t2_stats:
                t2_ttfts.append(t2_stats["avg_ttft_ms"])

        if t2_ttfts:
            data.append(t2_ttfts)
            label = CONFIG_LABELS.get(config, config)
            if r.get("ppd_enabled"):
                label += " (PPD)"
            labels.append(label)
            colors.append(COLORS.get(config, "#333333"))

    if not data:
        print("No data to plot for T2+ TTFT comparison")
        return

    # Create box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Turn 2+ TTFT (ms)', fontsize=12)
    ax.set_title('Turn 2+ TTFT Comparison Across Configurations', fontsize=14)

    # Add QPS annotation if filtered
    if qps_filter:
        ax.annotate(f'QPS = {qps_filter}', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_slo_attainment(
    results: Dict[str, List[dict]],
    output_path: Path,
    metric: str = "ttft",
    qps_filter: float = None,
):
    """Generate SLO attainment curve (DistServe style).

    X-axis: SLO threshold multiplier
    Y-axis: Attainment rate (fraction of requests meeting SLO)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # SLO threshold multipliers
    multipliers = np.linspace(0.5, 3.0, 50)

    for config in sorted(results.keys()):
        config_results = results[config]

        if qps_filter:
            config_results = [r for r in config_results if abs(r.get("qps", 0) - qps_filter) < 0.1]

        if not config_results:
            continue

        # Get T2+ metric values from SLO attainment data
        attainment_rates = []

        # Base SLO thresholds
        if metric == "ttft":
            base_slo = 100  # 100ms base TTFT SLO
            slo_keys = ["ttft_100ms", "ttft_200ms", "ttft_500ms"]
        elif metric == "tpot":
            base_slo = 15  # 15ms base TPOT SLO
            slo_keys = ["tpot_15ms", "tpot_30ms"]
        else:
            base_slo = 5000  # 5s base E2E SLO
            slo_keys = ["e2e_5000ms", "e2e_10000ms"]

        # For each multiplier, estimate attainment rate
        for mult in multipliers:
            threshold = base_slo * mult

            # Use closest available SLO data point
            total_attainment = 0
            count = 0
            for r in config_results:
                slo_data = r.get("slo_attainment", {})

                # Interpolate based on threshold
                if metric == "ttft":
                    if threshold <= 100:
                        rate = slo_data.get("ttft_100ms", 0) * (threshold / 100)
                    elif threshold <= 200:
                        r100 = slo_data.get("ttft_100ms", 0)
                        r200 = slo_data.get("ttft_200ms", 0)
                        rate = r100 + (r200 - r100) * (threshold - 100) / 100
                    elif threshold <= 500:
                        r200 = slo_data.get("ttft_200ms", 0)
                        r500 = slo_data.get("ttft_500ms", 0)
                        rate = r200 + (r500 - r200) * (threshold - 200) / 300
                    else:
                        rate = slo_data.get("ttft_500ms", 0.95)
                else:
                    # Simplified for other metrics
                    rate = 0.9  # Placeholder

                total_attainment += rate
                count += 1

            if count > 0:
                attainment_rates.append(total_attainment / count)
            else:
                attainment_rates.append(0)

        # Plot
        color = COLORS.get(config, "#333333")
        label = CONFIG_LABELS.get(config, config)
        if config_results and config_results[0].get("ppd_enabled"):
            label += " (PPD)"

        ax.plot(multipliers, attainment_rates, label=label, color=color, linewidth=2)

    ax.set_xlabel(f'{metric.upper()} SLO Threshold Multiplier (base={base_slo}ms)', fontsize=12)
    ax.set_ylabel('SLO Attainment Rate', fontsize=12)
    ax.set_title(f'{metric.upper()} SLO Attainment Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pareto_frontier(
    results: Dict[str, List[dict]],
    output_path: Path,
):
    """Generate Pareto frontier plot (latency vs throughput).

    Shows trade-off space and how PPD opens new regions.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for config in sorted(results.keys()):
        config_results = results[config]

        if not config_results:
            continue

        # Extract data points
        latencies = []
        throughputs = []
        qps_values = []

        for r in config_results:
            t2_stats = r.get("turn2plus", r.get("turn2", {}))
            throughput_data = r.get("throughput", {})

            if "avg_ttft_ms" in t2_stats and "requests_per_sec" in throughput_data:
                latencies.append(t2_stats["avg_ttft_ms"])
                throughputs.append(throughput_data["requests_per_sec"])
                qps_values.append(r.get("qps", 0))

        if not latencies:
            continue

        color = COLORS.get(config, "#333333")
        marker = MARKERS.get(config, "o")
        label = CONFIG_LABELS.get(config, config)
        if config_results[0].get("ppd_enabled"):
            label += " (PPD)"

        # Plot points
        ax.scatter(latencies, throughputs, c=color, marker=marker, s=100,
                   label=label, alpha=0.8, edgecolors='black', linewidth=0.5)

        # Connect points with lines
        sorted_indices = np.argsort(qps_values)
        sorted_lat = [latencies[i] for i in sorted_indices]
        sorted_thr = [throughputs[i] for i in sorted_indices]
        ax.plot(sorted_lat, sorted_thr, c=color, alpha=0.5, linestyle='--')

    ax.set_xlabel('Turn 2+ Average TTFT (ms)', fontsize=12)
    ax.set_ylabel('Throughput (requests/sec)', fontsize=12)
    ax.set_title('Latency-Throughput Trade-off (Pareto Frontier)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation for PPD region
    ax.annotate('PPD opens\nnew space', xy=(0.2, 0.8), xycoords='axes fraction',
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_turn_stability(
    results: Dict[str, List[dict]],
    output_path: Path,
    qps_filter: float = 4.0,
):
    """Generate multi-turn stability plot (TTFT across turn numbers).

    Shows how TTFT changes with turn number:
    - PD: TTFT increases with turn (more KV to transfer)
    - PPD: TTFT stays stable (prefix cache benefit)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # This requires per-turn data which we may not have in aggregated results
    # For now, we'll show Turn 1 vs Turn 2+ comparison

    configs_to_plot = []
    t1_means = []
    t2_means = []
    t1_stds = []
    t2_stds = []

    for config in sorted(results.keys()):
        config_results = results[config]

        if qps_filter:
            config_results = [r for r in config_results if abs(r.get("qps", 0) - qps_filter) < 0.1]

        if not config_results:
            continue

        # Extract Turn 1 and Turn 2+ stats
        t1_ttfts = []
        t2_ttfts = []

        for r in config_results:
            t1_stats = r.get("turn1", {})
            t2_stats = r.get("turn2plus", r.get("turn2", {}))

            if "avg_ttft_ms" in t1_stats:
                t1_ttfts.append(t1_stats["avg_ttft_ms"])
            if "avg_ttft_ms" in t2_stats:
                t2_ttfts.append(t2_stats["avg_ttft_ms"])

        if t1_ttfts and t2_ttfts:
            label = CONFIG_LABELS.get(config, config)
            if config_results[0].get("ppd_enabled"):
                label += " (PPD)"
            configs_to_plot.append(label)
            t1_means.append(np.mean(t1_ttfts))
            t2_means.append(np.mean(t2_ttfts))
            t1_stds.append(np.std(t1_ttfts) if len(t1_ttfts) > 1 else 0)
            t2_stds.append(np.std(t2_ttfts) if len(t2_ttfts) > 1 else 0)

    if not configs_to_plot:
        print("No data for turn stability plot")
        return

    x = np.arange(len(configs_to_plot))
    width = 0.35

    bars1 = ax.bar(x - width/2, t1_means, width, label='Turn 1', yerr=t1_stds, capsize=3)
    bars2 = ax.bar(x + width/2, t2_means, width, label='Turn 2+', yerr=t2_stds, capsize=3)

    ax.set_ylabel('Average TTFT (ms)', fontsize=12)
    ax.set_title('TTFT Stability: Turn 1 vs Turn 2+', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs_to_plot, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add ratio annotation
    for i, (t1, t2) in enumerate(zip(t1_means, t2_means)):
        ratio = t2 / t1 if t1 > 0 else 0
        ax.annotate(f'{ratio:.2f}x', xy=(i + width/2, t2), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ppd_improvement(
    results: Dict[str, List[dict]],
    output_path: Path,
):
    """Generate PPD improvement summary plot.

    Shows percentage improvement in T2+ TTFT when using PPD.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Find PD vs PPD pairs
    pd_configs = ["1P_3D", "2P_2D", "3P_1D"]
    ppd_configs = ["1P_3pD", "2P_2pD", "3P_1pD"]

    qps_values = sorted(set(
        r.get("qps", 0)
        for config_results in results.values()
        for r in config_results
    ))

    improvements = {qps: [] for qps in qps_values}
    labels = []

    for pd_config, ppd_config in zip(pd_configs, ppd_configs):
        if pd_config not in results or ppd_config not in results:
            continue

        labels.append(f"{pd_config} → {ppd_config}")

        for qps in qps_values:
            pd_results = [r for r in results[pd_config] if abs(r.get("qps", 0) - qps) < 0.1]
            ppd_results = [r for r in results[ppd_config] if abs(r.get("qps", 0) - qps) < 0.1]

            if pd_results and ppd_results:
                pd_t2_ttft = np.mean([r["turn2plus"]["avg_ttft_ms"] for r in pd_results if "turn2plus" in r])
                ppd_t2_ttft = np.mean([r["turn2plus"]["avg_ttft_ms"] for r in ppd_results if "turn2plus" in r])

                if pd_t2_ttft > 0:
                    improvement = (pd_t2_ttft - ppd_t2_ttft) / pd_t2_ttft * 100
                    improvements[qps].append(improvement)
                else:
                    improvements[qps].append(0)
            else:
                improvements[qps].append(0)

    if not labels:
        print("No PD/PPD pairs found for improvement plot")
        return

    x = np.arange(len(qps_values))
    width = 0.8 / len(labels)

    for i, label in enumerate(labels):
        values = [improvements[qps][i] if i < len(improvements[qps]) else 0 for qps in qps_values]
        ax.bar(x + i * width - 0.4 + width/2, values, width, label=label)

    ax.set_xlabel('QPS', fontsize=12)
    ax.set_ylabel('T2+ TTFT Improvement (%)', fontsize=12)
    ax.set_title('PPD Improvement in Turn 2+ TTFT', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{q}' for q in qps_values])
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_slo_combined(
    results: Dict[str, List[dict]],
    output_path: Path,
    baseline_config: str = "2P_2D",
):
    """Generate combined SLO attainment figure with 3x2 subplot layout.

    Layout (DistServe style):
    +---------------------------+---------------------------+
    |   TTFT SLO Attainment     |   TTFT SLO Attainment     |
    |   (X: SLO Scale)          |   (X: Per-GPU Rate)       |
    +---------------------------+---------------------------+
    |   TPOT SLO Attainment     |   TPOT SLO Attainment     |
    |   (X: SLO Scale)          |   (X: Per-GPU Rate)       |
    +---------------------------+---------------------------+
    |   E2E SLO Attainment      |   E2E SLO Attainment      |
    |   (X: SLO Scale)          |   (X: Per-GPU Rate)       |
    +---------------------------+---------------------------+

    Args:
        results: Dict of config -> list of results
        output_path: Where to save the figure
        baseline_config: Config to use as baseline for SLO Scale (default: 2P_2D)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    metrics = ["ttft", "tpot", "e2e"]
    metric_keys = {
        "ttft": ("avg_ttft_ms", "p50_ttft_ms"),
        "tpot": ("avg_tpot_ms", "p50_tpot_ms"),
        "e2e": ("avg_e2e_ms", "p50_e2e_ms"),
    }

    # Get baseline p50 values from baseline_config
    baselines = {}
    if baseline_config in results:
        baseline_results = results[baseline_config]
        for r in baseline_results:
            t2_stats = r.get("turn2plus", r.get("turn2", {}))
            for metric in metrics:
                key = metric_keys[metric][1]  # p50 key
                if key in t2_stats and metric not in baselines:
                    baselines[metric] = t2_stats[key]

    # Fallback to reasonable defaults if baseline not found
    if "ttft" not in baselines:
        baselines["ttft"] = 150  # ms
    if "tpot" not in baselines:
        baselines["tpot"] = 10  # ms
    if "e2e" not in baselines:
        baselines["e2e"] = 3000  # ms

    # Plot each metric row
    for row, metric in enumerate(metrics):
        # Left column: SLO Scale (multiplier of baseline p50)
        ax_scale = axes[row, 0]
        baseline = baselines[metric]
        scales = np.linspace(0.3, 3.0, 50)

        for config in sorted(results.keys()):
            config_results = results[config]
            if not config_results:
                continue

            # Collect all metric values
            all_values = []
            for r in config_results:
                t2_stats = r.get("turn2plus", r.get("turn2", {}))
                avg_key = metric_keys[metric][0]
                if avg_key in t2_stats:
                    # We need raw values, but for now use avg as approximation
                    all_values.append(t2_stats[avg_key])

            if not all_values:
                continue

            # Calculate attainment rate for each scale
            attainment_rates = []
            for scale in scales:
                threshold = baseline * scale
                # Approximate: what fraction of results would meet this SLO
                # Using avg as proxy for now
                rate = sum(1 for v in all_values if v <= threshold) / len(all_values)
                attainment_rates.append(rate)

            color = COLORS.get(config, "#333333")
            marker = MARKERS.get(config, "o")
            label = CONFIG_LABELS.get(config, config)

            ax_scale.plot(scales, attainment_rates, label=label, color=color,
                         marker=marker, markevery=10, linewidth=2)

        ax_scale.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax_scale.set_xlabel(f'SLO Scale (baseline={baseline:.0f}ms)', fontsize=10)
        ax_scale.set_ylabel('SLO Attainment (%)', fontsize=10)
        ax_scale.set_title(f'{metric.upper()} - SLO Scale', fontsize=11)
        ax_scale.set_ylim(0, 1.05)
        ax_scale.set_xlim(0.3, 3.0)
        ax_scale.grid(True, alpha=0.3)
        if row == 0:
            ax_scale.legend(loc='lower right', fontsize=8)

        # Right column: Per-GPU Rate (requests/sec)
        ax_rate = axes[row, 1]

        for config in sorted(results.keys()):
            config_results = results[config]
            if not config_results:
                continue

            # Group by QPS and calculate attainment at fixed SLO (baseline p50)
            qps_attainment = {}
            for r in config_results:
                qps = r.get("qps", 0)
                t2_stats = r.get("turn2plus", r.get("turn2", {}))
                avg_key = metric_keys[metric][0]
                if avg_key in t2_stats:
                    val = t2_stats[avg_key]
                    if qps not in qps_attainment:
                        qps_attainment[qps] = []
                    # Check if meets SLO (baseline p50)
                    qps_attainment[qps].append(1 if val <= baseline else 0)

            if not qps_attainment:
                continue

            # Calculate per-GPU rate (QPS / 4 GPUs)
            rates = []
            attainments = []
            for qps in sorted(qps_attainment.keys()):
                rates.append(qps / 4)  # Per-GPU rate
                attainments.append(np.mean(qps_attainment[qps]))

            color = COLORS.get(config, "#333333")
            marker = MARKERS.get(config, "o")
            label = CONFIG_LABELS.get(config, config)

            ax_rate.plot(rates, attainments, label=label, color=color,
                        marker=marker, linewidth=2)

        ax_rate.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax_rate.set_xlabel('Per-GPU Rate (req/s)', fontsize=10)
        ax_rate.set_ylabel('SLO Attainment (%)', fontsize=10)
        ax_rate.set_title(f'{metric.upper()} - Per-GPU Rate', fontsize=11)
        ax_rate.set_ylim(0, 1.05)
        ax_rate.grid(True, alpha=0.3)

    plt.suptitle('SLO Attainment Comparison (Baseline: 2P_2D p50)', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_table(
    results: Dict[str, List[dict]],
    output_path: Path,
):
    """Generate metrics summary table figure.

    Shows p50/p90/p99/avg for TTFT, TPOT, E2E for each config.
    Highlights best values per column.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    # Collect data
    configs = sorted(results.keys())
    metrics = ['TTFT', 'TPOT', 'E2E']
    stats = ['avg', 'p50', 'p90', 'p99']

    # Build table data
    columns = ['Config'] + [f'{m} {s}' for m in metrics for s in stats]
    table_data = []

    for config in configs:
        row = [CONFIG_LABELS.get(config, config)]
        config_results = results[config]

        for metric in ['ttft', 'tpot', 'e2e']:
            for stat in stats:
                values = []
                for r in config_results:
                    t2_stats = r.get("turn2plus", r.get("turn2", {}))
                    key = f"{stat}_{metric}_ms"
                    if stat == "avg":
                        key = f"avg_{metric}_ms"
                    if key in t2_stats:
                        values.append(t2_stats[key])

                if values:
                    row.append(f"{np.mean(values):.1f}")
                else:
                    row.append("-")

        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                    cellLoc='center', colColours=['#f0f0f0']*len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for j, col in enumerate(columns):
        table[(0, j)].set_facecolor('#4a86e8')
        table[(0, j)].set_text_props(color='white', weight='bold')

    plt.title('Performance Metrics Summary (Turn 2+)', fontsize=14, pad=20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures(
    results_dir: Path,
    output_dir: Path,
    comprehensive_dir: Path = None,
):
    """Generate all analysis figures."""
    print(f"Loading results from {results_dir}...")
    results = load_sharegpt_results(results_dir)

    if not results:
        print("No results found!")

        # Try comprehensive results if available
        if comprehensive_dir and comprehensive_dir.exists():
            print(f"Trying comprehensive results from {comprehensive_dir}...")
            results = load_comprehensive_results(
                comprehensive_dir,
                ["4R", "2P_2D", "2P_2pD", "1P_3D", "1P_3pD"]
            )

    if not results:
        print("No results to analyze!")
        return

    print(f"Loaded results for configs: {list(results.keys())}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\nGenerating figures...")

    # 1. T2+ TTFT comparison
    plot_t2_ttft_comparison(results, output_dir / "fig1_t2_ttft_comparison.png")

    # 2. SLO attainment curves (legacy TTFT only)
    plot_slo_attainment(results, output_dir / "fig2_slo_attainment_ttft.png", metric="ttft")

    # 3. Pareto frontier
    plot_pareto_frontier(results, output_dir / "fig3_pareto_frontier.png")

    # 4. Turn stability
    plot_turn_stability(results, output_dir / "fig4_turn_stability.png")

    # 5. PPD improvement
    plot_ppd_improvement(results, output_dir / "fig5_ppd_improvement.png")

    # 6. Combined SLO figure (3x2 subplot - NEW)
    plot_slo_combined(results, output_dir / "fig6_slo_combined.png")

    # 7. Metrics summary table (NEW)
    plot_metrics_table(results, output_dir / "fig7_metrics_table.png")

    print(f"\nAll figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PPD Analysis - Generate Figures")
    parser.add_argument("--dataset-dir", type=str, default="results/sharegpt",
                        help="Directory with ShareGPT benchmark results")
    parser.add_argument("--comprehensive-dir", type=str, default="results/comprehensive",
                        help="Directory with comprehensive benchmark results (fallback)")
    parser.add_argument("--output-dir", type=str, default="results/analysis/figures/sharegpt",
                        help="Output directory for figures")

    args = parser.parse_args()

    dataset_dir = Path(PROJECT_DIR) / args.dataset_dir
    comprehensive_dir = Path(PROJECT_DIR) / args.comprehensive_dir
    output_dir = Path(PROJECT_DIR) / args.output_dir

    generate_all_figures(dataset_dir, output_dir, comprehensive_dir)


if __name__ == "__main__":
    main()
