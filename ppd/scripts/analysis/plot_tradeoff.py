#!/usr/bin/env python3
"""
Comprehensive Trade-off Visualization for PPD Benchmark (Revised)

Based on user feedback:
- A1: Multiple alternative visualizations (Stacked Bar, Grouped Bar, Sankey, Simplified Heatmap)
- A3: Improved radar chart with better data point selection
- C1-C3: Retained (good examples)
- F1-F2: Moved to internal/ (for user analysis)
- H1-H3: Retained (supplementary)
- Removed: A2, B1-B3, D1-D3, E1-E2 (low information density)
- Output: PNG only (no PDF)
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.sankey import Sankey
import seaborn as sns

# Project paths
PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "results" / "comprehensive"
TURN_SCALING_DIR = PROJECT_DIR / "results" / "turn_scaling"
MODEL_SCALING_DIR = PROJECT_DIR / "results" / "model_scaling"
OUTPUT_DIR = PROJECT_DIR / "results" / "analysis" / "figures"

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color schemes
MODE_COLORS = {
    'Replica': '#2ecc71',    # Green
    'PD': '#3498db',         # Blue
    'PPD': '#e74c3c',        # Red
    'Mixed_DpD': '#9b59b6',  # Purple
    'Hybrid': '#f39c12',     # Orange
}

CONFIG_COLORS = {
    '4R': '#2ecc71',
    '1P_3D': '#85c1e9',
    '2P_2D': '#3498db',
    '3P_1D': '#1a5276',
    '1P_3pD': '#f1948a',
    '2P_2pD': '#e74c3c',
    '3P_1pD': '#922b21',
    '1P_2D_1pD': '#bb8fce',
    '1P_1D_2pD': '#9b59b6',
    '2P_1D_1pD': '#6c3483',
    '1R_1P_2D': '#fad7a0',
    '1R_1P_2pD': '#f39c12',
    '1R_1P_1D_1pD': '#d68910',
    '1R_2P_1D': '#f5b041',
    '1R_2P_1pD': '#e67e22',
    '2R_1P_1D': '#fdebd0',
    '2R_1P_1pD': '#f8c471',
}

# Classification
CONFIG_CATEGORIES = {
    "4R": "Replica",
    "1P_3D": "PD", "2P_2D": "PD", "3P_1D": "PD",
    "1P_3pD": "PPD", "2P_2pD": "PPD", "3P_1pD": "PPD",
    "1P_2D_1pD": "Mixed_DpD", "1P_1D_2pD": "Mixed_DpD", "2P_1D_1pD": "Mixed_DpD",
    "1R_1P_2D": "Hybrid", "1R_1P_2pD": "Hybrid", "1R_1P_1D_1pD": "Hybrid",
    "1R_2P_1D": "Hybrid", "1R_2P_1pD": "Hybrid", "2R_1P_1D": "Hybrid", "2R_1P_1pD": "Hybrid",
}

WORKLOAD_CATEGORIES = {
    "tiny": "Decode-heavy", "short_gen": "Decode-heavy",
    "long_gen": "Decode-heavy", "very_long_gen": "Decode-heavy",
    "small_bal": "Balanced", "mid_bal": "Balanced",
    "mid_paste": "Prefill-heavy", "big_paste": "Prefill-heavy", "huge_paste": "Prefill-heavy",
}

# Representative configs
REPRESENTATIVE_CONFIGS = ["4R", "2P_2D", "2P_2pD", "1P_3D", "3P_1D", "1P_3pD", "1R_1P_1D_1pD", "1R_1P_2pD"]
PD_CONFIGS = ["1P_3D", "2P_2D", "3P_1D"]
PPD_CONFIGS = ["1P_3pD", "2P_2pD", "3P_1pD"]


# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load all benchmark data."""
    print("Loading data...")

    all_records = []

    for config_dir in DATA_DIR.iterdir():
        if not config_dir.is_dir() or config_dir.name == "checkpoint.json":
            continue

        config = config_dir.name

        for json_file in config_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

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

                    # Turn metrics
                    "t1_avg_ttft": data.get("turn1", {}).get("avg_ttft_ms", 0),
                    "t1_avg_tpot": data.get("turn1", {}).get("avg_tpot_ms", 0),
                    "t1_avg_e2e": data.get("turn1", {}).get("avg_e2e_ms", 0),
                    "t1_p50_ttft": data.get("turn1", {}).get("p50_ttft_ms", 0),
                    "t1_p99_ttft": data.get("turn1", {}).get("p99_ttft_ms", 0),
                    "t1_p50_tpot": data.get("turn1", {}).get("p50_tpot_ms", 0),
                    "t1_p99_tpot": data.get("turn1", {}).get("p99_tpot_ms", 0),

                    "t2_avg_ttft": data.get("turn2", {}).get("avg_ttft_ms", 0),
                    "t2_avg_tpot": data.get("turn2", {}).get("avg_tpot_ms", 0),
                    "t2_avg_e2e": data.get("turn2", {}).get("avg_e2e_ms", 0),
                    "t2_p50_ttft": data.get("turn2", {}).get("p50_ttft_ms", 0),
                    "t2_p99_ttft": data.get("turn2", {}).get("p99_ttft_ms", 0),
                    "t2_p50_tpot": data.get("turn2", {}).get("p50_tpot_ms", 0),
                    "t2_p99_tpot": data.get("turn2", {}).get("p99_tpot_ms", 0),

                    "avg_ttft": data.get("average", {}).get("avg_ttft_ms", 0),
                    "avg_tpot": data.get("average", {}).get("avg_tpot_ms", 0),
                    "avg_e2e": data.get("average", {}).get("avg_e2e_ms", 0),

                    "throughput_rps": data.get("throughput", {}).get("requests_per_sec", 0),
                    "throughput_tps": data.get("throughput", {}).get("tokens_per_sec", 0),
                }

                all_records.append(record)
            except Exception as e:
                print(f"  Error loading {json_file}: {e}")

    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} records")
    return df


def load_turn_scaling_data() -> pd.DataFrame:
    """Load turn scaling experiment data."""
    print("Loading turn scaling data...")
    records = []

    for json_file in TURN_SCALING_DIR.glob("*.json"):
        if json_file.name == "all_results.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Parse filename: e.g., "2P_2D_8turns.json" -> config="2P_2D", num_turns=8
            parts = json_file.stem.rsplit('_', 1)
            config = parts[0]
            num_turns = int(parts[1].replace('turns', ''))

            records.append({
                "config": config,
                "num_turns": num_turns,
                "t1_ttft": data.get("t1_metrics", {}).get("avg_ttft_ms"),
                "t2plus_ttft": data.get("t2plus_metrics", {}).get("avg_ttft_ms"),
                "t1_tpot": data.get("t1_metrics", {}).get("avg_tpot_ms"),
                "t2plus_tpot": data.get("t2plus_metrics", {}).get("avg_tpot_ms"),
                "per_turn_metrics": data.get("per_turn_metrics", []),
                "workload": data.get("workload", ""),
                "qps": data.get("qps", 0),
            })
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} turn scaling records")
    return df


def load_model_scaling_data() -> pd.DataFrame:
    """Load model scaling experiment data."""
    print("Loading model scaling data...")
    records = []

    for json_file in MODEL_SCALING_DIR.glob("*.json"):
        if json_file.name == "all_results.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)

            records.append({
                "model_size": data.get("model_size", ""),
                "model_name": data.get("model_name", ""),
                "config": data.get("config", ""),
                "t1_ttft": data.get("turn1", {}).get("avg_ttft_ms"),
                "t2_ttft": data.get("turn2", {}).get("avg_ttft_ms"),
                "t1_tpot": data.get("turn1", {}).get("avg_tpot_ms"),
                "t2_tpot": data.get("turn2", {}).get("avg_tpot_ms"),
                "t1_e2e": data.get("turn1", {}).get("avg_e2e_ms"),
                "t2_e2e": data.get("turn2", {}).get("avg_e2e_ms"),
                "avg_tpot": data.get("average", {}).get("avg_tpot_ms"),
                "avg_e2e": data.get("average", {}).get("avg_e2e_ms"),
                "workload": data.get("workload", ""),
                "qps": data.get("qps", 0),
                "success_rate": data.get("success_rate", 0),
            })
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} model scaling records")
    return df


# =============================================================================
# A1 Alternatives: Winner Visualization
# =============================================================================

def compute_winners(df: pd.DataFrame) -> Dict:
    """Compute winner for each (workload, QPS) combination for each metric."""
    valid_df = df[df['success_rate'] >= 95].copy()

    metrics = [
        ('t2_avg_ttft', True, 'T2 TTFT'),
        ('avg_tpot', True, 'Avg TPOT'),
        ('throughput_tps', False, 'Throughput'),
    ]

    results = {}
    for metric, lower_is_better, name in metrics:
        winners = {}
        for (workload, qps), group in valid_df.groupby(['workload', 'qps']):
            if len(group) == 0:
                continue
            if lower_is_better:
                winner_idx = group[metric].idxmin()
            else:
                winner_idx = group[metric].idxmax()
            winner_cat = group.loc[winner_idx, 'config_category']
            winners[(workload, qps)] = winner_cat
        results[name] = winners

    return results


def plot_A1_v1_stacked_bar(df: pd.DataFrame, output_dir: Path):
    """
    A1 Version 1: Stacked Bar Chart
    Shows winner distribution by mode category for each metric.
    Directly proves Objective-Oriented thesis.
    """
    print("Plotting A1 v1: Stacked Bar Chart...")

    winners = compute_winners(df)

    # Count wins by mode for each metric
    metrics = ['T2 TTFT', 'Avg TPOT', 'Throughput']
    modes = ['Replica', 'PD', 'PPD', 'Mixed_DpD', 'Hybrid']

    data = {mode: [] for mode in modes}
    for metric in metrics:
        total = len(winners[metric])
        mode_counts = defaultdict(int)
        for winner in winners[metric].values():
            mode_counts[winner] += 1
        for mode in modes:
            data[mode].append(mode_counts[mode] / total * 100 if total > 0 else 0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.6
    bottom = np.zeros(len(metrics))

    for mode in modes:
        bars = ax.bar(x, data[mode], width, bottom=bottom, label=mode,
                     color=MODE_COLORS[mode], edgecolor='white', linewidth=0.5)
        # Add percentage labels
        for i, (b, v) in enumerate(zip(bars, data[mode])):
            if v > 5:  # Only show if > 5%
                ax.text(b.get_x() + b.get_width()/2, bottom[i] + v/2,
                       f'{v:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')
        bottom += data[mode]

    ax.set_ylabel('Winner Percentage (%)')
    ax.set_xlabel('Optimization Objective')
    ax.set_title('Winner Distribution by Optimization Objective\n(Proves: No Single Mode Wins All)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5)

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'A1_v1_stacked_bar.png')
    plt.close()
    print("  Saved A1_v1_stacked_bar.png")


def plot_Fig1_objective_oriented(df: pd.DataFrame, output_dir: Path):
    """
    Fig1: Objective-Oriented Configuration Selection (Enhanced)

    Shows that different optimization objectives lead to different optimal configurations.
    This is the core figure for Finding 1.

    Design improvements:
    - Professional color palette with gradients
    - Statistical annotations
    - Value labels on bars
    - Clean typography
    - Dominant mode highlighting
    """
    print("Plotting Fig1: Objective-Oriented (Enhanced)...")

    valid_df = df[df['success_rate'] >= 95].copy()

    workload_cats = ['Decode-heavy', 'Balanced', 'Prefill-heavy']
    modes = ['Replica', 'PD', 'PPD', 'Mixed_DpD', 'Hybrid']

    # Enhanced color palette (professional, distinguishable)
    enhanced_colors = {
        'Replica': '#1a73e8',    # Google Blue
        'PD': '#ea8600',         # Orange
        'PPD': '#34a853',        # Google Green
        'Mixed_DpD': '#9334e6',  # Purple
        'Hybrid': '#ea4335',     # Google Red
    }

    metrics = [
        ('t2_avg_ttft', True, 'T2 TTFT', 'Minimize Turn-2 Latency'),
        ('avg_tpot', True, 'Avg TPOT', 'Minimize Per-Token Latency'),
        ('throughput_tps', False, 'Throughput', 'Maximize Throughput'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Store global winner counts for summary
    global_winners = {metric[0]: {mode: 0 for mode in modes} for metric in metrics}
    global_totals = {metric[0]: 0 for metric in metrics}

    for ax_idx, (ax, (metric, lower_is_better, short_title, full_title)) in enumerate(zip(axes, metrics)):
        # Count wins by workload category and mode
        data = {wl_cat: {mode: 0 for mode in modes} for wl_cat in workload_cats}
        totals = {wl_cat: 0 for wl_cat in workload_cats}

        for (workload, qps), group in valid_df.groupby(['workload', 'qps']):
            if len(group) == 0:
                continue
            wl_cat = group['workload_category'].iloc[0]
            if wl_cat not in workload_cats:
                continue

            if lower_is_better:
                winner_cat = group.loc[group[metric].idxmin(), 'config_category']
            else:
                winner_cat = group.loc[group[metric].idxmax(), 'config_category']

            data[wl_cat][winner_cat] += 1
            totals[wl_cat] += 1
            global_winners[metric][winner_cat] += 1
            global_totals[metric] += 1

        # Convert to percentages
        x = np.arange(len(workload_cats))
        width = 0.14

        # Find dominant mode for this metric
        total_wins = sum(global_winners[metric].values())
        dominant_mode = max(global_winners[metric].items(), key=lambda x: x[1])[0] if total_wins > 0 else None
        dominant_pct = global_winners[metric][dominant_mode] / total_wins * 100 if total_wins > 0 else 0

        for i, mode in enumerate(modes):
            values = [data[wl][mode] / totals[wl] * 100 if totals[wl] > 0 else 0
                     for wl in workload_cats]

            # Highlight dominant mode with thicker edge
            edge_width = 2.5 if mode == dominant_mode else 0.5
            edge_color = 'black' if mode == dominant_mode else 'white'

            bars = ax.bar(x + i*width - 2*width, values, width, label=mode,
                         color=enhanced_colors[mode], edgecolor=edge_color,
                         linewidth=edge_width, alpha=0.9)

            # Add value labels for significant bars
            for bar, val in zip(bars, values):
                if val >= 15:  # Only show if >= 15%
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_ylabel('Winner Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Workload Category', fontsize=11, fontweight='bold')
        ax.set_title(f'{full_title}\n', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(workload_cats, fontsize=10)
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add annotation box showing dominant mode
        if dominant_mode:
            ax.annotate(f'Dominant: {dominant_mode}\n({dominant_pct:.1f}% overall)',
                       xy=(0.98, 0.98), xycoords='axes fraction',
                       ha='right', va='top', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=enhanced_colors[dominant_mode],
                                alpha=0.2, edgecolor=enhanced_colors[dominant_mode]))

        # Add spines styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    # Common legend at bottom
    handles = [mpatches.Patch(color=enhanced_colors[m], label=m, edgecolor='gray', linewidth=0.5)
               for m in modes]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, fontsize=10, frameon=True, fancybox=True, shadow=True)

    # Main title with key insight
    plt.suptitle('Objective-Oriented Configuration Selection\n'
                 '92.2% of scenarios have different optimal configs for TTFT vs TPOT',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_dir / 'Part1_Fig1_objective_oriented.png', facecolor='white')
    plt.close()
    print("  Saved Part1_Fig1_objective_oriented.png")


def plot_A1_v3_sankey(df: pd.DataFrame, output_dir: Path):
    """
    A1 Version 3: Sankey-like Flow Diagram (using stacked horizontal bars)
    Shows flow from workload type -> metric -> optimal mode.
    """
    print("Plotting A1 v3: Flow Diagram...")

    valid_df = df[df['success_rate'] >= 95].copy()

    workload_cats = ['Decode-heavy', 'Balanced', 'Prefill-heavy']
    metrics = ['T2 TTFT', 'Avg TPOT', 'Throughput']
    modes = ['Replica', 'PD', 'PPD', 'Mixed_DpD', 'Hybrid']

    # Create a more readable flow visualization using subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    metric_configs = [
        ('t2_avg_ttft', True, 'T2 TTFT'),
        ('avg_tpot', True, 'Avg TPOT'),
        ('throughput_tps', False, 'Throughput'),
    ]

    for ax, (metric, lower_is_better, metric_name) in zip(axes, metric_configs):
        # Count wins by workload category and mode
        data = []
        for wl_cat in workload_cats:
            wl_data = valid_df[valid_df['workload_category'] == wl_cat]
            mode_counts = defaultdict(int)
            total = 0

            for (workload, qps), group in wl_data.groupby(['workload', 'qps']):
                if len(group) == 0:
                    continue
                if lower_is_better:
                    winner_cat = group.loc[group[metric].idxmin(), 'config_category']
                else:
                    winner_cat = group.loc[group[metric].idxmax(), 'config_category']
                mode_counts[winner_cat] += 1
                total += 1

            row = {'workload': wl_cat}
            for mode in modes:
                row[mode] = mode_counts[mode] / total * 100 if total > 0 else 0
            data.append(row)

        # Plot horizontal stacked bars
        y = np.arange(len(workload_cats))
        height = 0.6
        left = np.zeros(len(workload_cats))

        for mode in modes:
            values = [d[mode] for d in data]
            bars = ax.barh(y, values, height, left=left, label=mode,
                          color=MODE_COLORS[mode], edgecolor='white', linewidth=0.5)
            # Add percentage labels
            for i, (bar, v) in enumerate(zip(bars, values)):
                if v > 8:
                    ax.text(left[i] + v/2, i, f'{v:.0f}%',
                           ha='center', va='center', fontsize=9, fontweight='bold')
            left += values

        ax.set_xlabel('Winner Percentage (%)')
        ax.set_yticks(y)
        ax.set_yticklabels(workload_cats)
        ax.set_title(f'Optimizing for {metric_name}', fontsize=11)
        ax.set_xlim(0, 100)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    # Common legend
    handles = [mpatches.Patch(color=MODE_COLORS[m], label=m) for m in modes]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=5)

    plt.suptitle('Winner Flow: Workload Category → Optimal Mode by Metric', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_dir / 'A1_v3_flow_diagram.png')
    plt.close()
    print("  Saved A1_v3_flow_diagram.png")


def plot_A1_v4_simplified_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    A1 Version 4: Simplified Heatmap + Summary Table
    Aggregates workload and QPS into categories for cleaner visualization.
    """
    print("Plotting A1 v4: Simplified Heatmap...")

    valid_df = df[df['success_rate'] >= 95].copy()

    # Define QPS categories
    def qps_category(qps):
        if qps <= 2:
            return 'Low (≤2)'
        elif qps <= 8:
            return 'Medium (4-8)'
        else:
            return 'High (>8)'

    valid_df['qps_category'] = valid_df['qps'].apply(qps_category)

    workload_cats = ['Decode-heavy', 'Balanced', 'Prefill-heavy']
    qps_cats = ['Low (≤2)', 'Medium (4-8)', 'High (>8)']
    modes = ['Replica', 'PD', 'PPD', 'Mixed_DpD', 'Hybrid']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ('t2_avg_ttft', True, 'T2 TTFT'),
        ('avg_tpot', True, 'Avg TPOT'),
        ('throughput_tps', False, 'Throughput'),
    ]

    category_to_num = {m: i for i, m in enumerate(modes)}

    for ax, (metric, lower_is_better, title) in zip(axes, metrics):
        # Find most common winner for each (workload_cat, qps_cat)
        matrix = np.zeros((len(workload_cats), len(qps_cats)))
        annotations = [['' for _ in qps_cats] for _ in workload_cats]

        for i, wl_cat in enumerate(workload_cats):
            for j, qps_cat in enumerate(qps_cats):
                subset = valid_df[
                    (valid_df['workload_category'] == wl_cat) &
                    (valid_df['qps_category'] == qps_cat)
                ]

                if len(subset) == 0:
                    matrix[i, j] = -1
                    annotations[i][j] = 'N/A'
                    continue

                # Count winners
                mode_counts = defaultdict(int)
                for (workload, qps), group in subset.groupby(['workload', 'qps']):
                    if len(group) == 0:
                        continue
                    if lower_is_better:
                        winner_cat = group.loc[group[metric].idxmin(), 'config_category']
                    else:
                        winner_cat = group.loc[group[metric].idxmax(), 'config_category']
                    mode_counts[winner_cat] += 1

                if mode_counts:
                    dominant_mode = max(mode_counts.items(), key=lambda x: x[1])[0]
                    total = sum(mode_counts.values())
                    pct = mode_counts[dominant_mode] / total * 100
                    matrix[i, j] = category_to_num[dominant_mode]
                    annotations[i][j] = f'{dominant_mode}\n({pct:.0f}%)'

        # Create heatmap
        colors = [MODE_COLORS[m] for m in modes]
        cmap = LinearSegmentedColormap.from_list('mode', colors, N=len(modes))

        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=len(modes)-1)

        # Add annotations
        for i in range(len(workload_cats)):
            for j in range(len(qps_cats)):
                text = annotations[i][j]
                ax.text(j, i, text, ha='center', va='center', fontsize=8,
                       color='white' if matrix[i,j] >= 0 else 'black', fontweight='bold')

        ax.set_xticks(range(len(qps_cats)))
        ax.set_xticklabels(qps_cats)
        ax.set_yticks(range(len(workload_cats)))
        ax.set_yticklabels(workload_cats)
        ax.set_xlabel('QPS Level')
        ax.set_ylabel('Workload Type')
        ax.set_title(f'Best for {title}')

    # Legend
    legend_patches = [mpatches.Patch(color=MODE_COLORS[m], label=m) for m in modes]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=5)

    plt.suptitle('Dominant Mode by Scenario (Simplified View)', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig(output_dir / 'A1_v4_simplified_heatmap.png')
    plt.close()
    print("  Saved A1_v4_simplified_heatmap.png")


# =============================================================================
# A3: Improved Radar Chart
# =============================================================================

def plot_A3_improved_radar(df: pd.DataFrame, output_dir: Path):
    """
    A3: Improved Radar Chart
    Multiple scenarios to show different trade-off patterns.
    """
    print("Plotting A3: Improved Radar Chart...")

    valid_df = df[df['success_rate'] >= 95].copy()

    # Select multiple representative scenarios
    scenarios = [
        ('small_tiny', 4, 'Decode-heavy @ QPS=4'),
        ('small_mid_bal', 4, 'Balanced @ QPS=4'),
        ('small_big_paste', 4, 'Prefill-heavy @ QPS=4'),
    ]

    configs_to_show = ['4R', '2P_2D', '2P_2pD', '1R_1P_2pD', '1P_3pD']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))

    metrics = ['t1_avg_ttft', 't2_avg_ttft', 'avg_tpot', 'avg_e2e']
    metric_labels = ['T1 TTFT', 'T2 TTFT', 'Avg TPOT', 'Avg E2E']

    for ax, (workload, qps, title) in zip(axes, scenarios):
        scenario_df = valid_df[(valid_df['workload'] == workload) & (valid_df['qps'] == qps)]

        if len(scenario_df) == 0:
            ax.set_title(f'{title}\n(No data)')
            continue

        # Add inverted throughput
        scenario_df = scenario_df.copy()
        max_throughput = scenario_df['throughput_tps'].max()
        scenario_df['inv_throughput'] = max_throughput - scenario_df['throughput_tps']

        current_metrics = metrics + ['inv_throughput']
        current_labels = metric_labels + ['Throughput\n(inverted)']

        # Normalize each metric to [0, 1]
        normalized = {}
        for metric in current_metrics:
            min_val = scenario_df[metric].min()
            max_val = scenario_df[metric].max()
            if max_val > min_val:
                normalized[metric] = (scenario_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized[metric] = scenario_df[metric] * 0

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(current_metrics), endpoint=False).tolist()
        angles += angles[:1]

        for config in configs_to_show:
            config_data = scenario_df[scenario_df['config'] == config]
            if len(config_data) == 0:
                continue

            values = [normalized[m].loc[config_data.index[0]] for m in current_metrics]
            values += values[:1]

            color = CONFIG_COLORS.get(config, '#333333')
            ax.plot(angles, values, 'o-', linewidth=2, label=config, color=color, markersize=4)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(current_labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=11, pad=15)

    # Common legend
    handles = [mpatches.Patch(color=CONFIG_COLORS.get(c, '#333'), label=c) for c in configs_to_show]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=5)

    plt.suptitle('Configuration Trade-offs Across Workload Types\n(Lower = Better, center = best)',
                 fontsize=12, y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_dir / 'A3_improved_radar.png')
    plt.close()
    print("  Saved A3_improved_radar.png")


def plot_A3_parallel_coordinates(df: pd.DataFrame, output_dir: Path):
    """
    A3 Alternative: Parallel Coordinates Plot
    Shows all metrics at once for multiple configurations.
    """
    print("Plotting A3: Parallel Coordinates...")

    valid_df = df[df['success_rate'] >= 95].copy()

    # Select representative scenario
    workload = 'small_mid_bal'
    qps = 4

    scenario_df = valid_df[(valid_df['workload'] == workload) & (valid_df['qps'] == qps)]

    if len(scenario_df) == 0:
        print("  No data for parallel coordinates")
        return

    configs_to_show = ['4R', '2P_2D', '2P_2pD', '1P_3D', '3P_1D', '1R_1P_2pD']
    scenario_df = scenario_df[scenario_df['config'].isin(configs_to_show)]

    metrics = ['t1_avg_ttft', 't2_avg_ttft', 'avg_tpot', 'avg_e2e', 'throughput_tps']
    labels = ['T1 TTFT', 'T2 TTFT', 'Avg TPOT', 'Avg E2E', 'Throughput']

    # Normalize (0=best, 1=worst for all except throughput)
    normalized_df = scenario_df[['config'] + metrics].copy()
    for metric in metrics:
        min_val = scenario_df[metric].min()
        max_val = scenario_df[metric].max()
        if max_val > min_val:
            if metric == 'throughput_tps':
                # Invert throughput so higher is better -> lower normalized
                normalized_df[metric] = 1 - (scenario_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[metric] = (scenario_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[metric] = 0

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))

    for _, row in normalized_df.iterrows():
        config = row['config']
        values = [row[m] for m in metrics]
        color = CONFIG_COLORS.get(config, '#333333')
        ax.plot(x, values, 'o-', linewidth=2.5, markersize=8, label=config,
               color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Normalized Value (0=Best, 1=Worst)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'Configuration Comparison: {workload} @ QPS={qps}\n(Lower is Better for All Metrics)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'A3_parallel_coordinates.png')
    plt.close()
    print("  Saved A3_parallel_coordinates.png")


# =============================================================================
# C-Series: Config Analysis Figures (Retained - Good Examples)
# =============================================================================

def plot_C1_pd_ratio_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    C1: P:D Ratio Heatmap
    Shows how different P:D ratios perform across QPS levels.
    """
    print("Plotting C1: P:D Ratio Heatmap...")

    valid_df = df[df['success_rate'] >= 95].copy()

    pd_configs = ['1P_3D', '2P_2D', '3P_1D']
    ppd_configs = ['1P_3pD', '2P_2pD', '3P_1pD']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax_row, (configs, mode_name) in enumerate([(pd_configs, 'PD'), (ppd_configs, 'PPD')]):
        for ax_col, (metric, title) in enumerate([('t2_avg_ttft', 'T2 TTFT (ms)'), ('avg_tpot', 'Avg TPOT (ms)')]):
            ax = axes[ax_row, ax_col]

            workload = 'small_mid_bal'
            wl_data = valid_df[(valid_df['workload'] == workload) & (valid_df['config'].isin(configs))]

            if len(wl_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            pivot = wl_data.pivot_table(values=metric, index='config', columns='qps', aggfunc='mean')
            pivot = pivot.reindex(configs)

            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': title})
            ax.set_title(f'{mode_name} Mode - {title}')
            ax.set_xlabel('QPS')
            ax.set_ylabel('Config (P:D ratio)')

    plt.suptitle(f'P:D Ratio Impact on Performance\n(Workload: small_mid_bal)', fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / 'Part2_Fig2_pd_ratio_heatmap.png')
    plt.close()
    print("  Saved Part2_Fig2_pd_ratio_heatmap.png")


def plot_C2_d_vs_pd_comparison(df: pd.DataFrame, output_dir: Path):
    """
    C2: D vs pD Comparison
    Compares pure D configs with pD configs.
    """
    print("Plotting C2: D vs pD Comparison...")

    valid_df = df[df['success_rate'] >= 95].copy()

    pairs = [
        ('2P_2D', '2P_2pD', '2P'),
        ('1P_3D', '1P_3pD', '1P'),
        ('3P_1D', '3P_1pD', '3P'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    metrics = [('t2_avg_ttft', 'T2 TTFT (ms)'), ('avg_tpot', 'Avg TPOT (ms)')]
    workloads = ['small_tiny', 'small_mid_bal', 'small_big_paste']

    for col, wl in enumerate(workloads):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]

            wl_data = valid_df[valid_df['workload'] == wl]

            for d_config, pd_config, label in pairs:
                d_data = wl_data[wl_data['config'] == d_config].sort_values('qps')
                pd_data = wl_data[wl_data['config'] == pd_config].sort_values('qps')

                if len(d_data) > 0:
                    ax.plot(d_data['qps'], d_data[metric], '--', label=f'{d_config} (D)',
                           color=CONFIG_COLORS.get(d_config, '#333'), linewidth=2, marker='o')
                if len(pd_data) > 0:
                    ax.plot(pd_data['qps'], pd_data[metric], '-', label=f'{pd_config} (pD)',
                           color=CONFIG_COLORS.get(pd_config, '#333'), linewidth=2, marker='s')

            ax.set_xlabel('QPS')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{wl}')
            if col == 2 and row == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('D vs pD Comparison Across Workloads', fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.88, 0.98])
    plt.savefig(output_dir / 'Part2_Fig1_d_vs_pd_comparison.png')
    plt.close()
    print("  Saved Part2_Fig1_d_vs_pd_comparison.png")


def plot_C3_hybrid_vs_pure(df: pd.DataFrame, output_dir: Path):
    """
    C3: Hybrid vs Pure Mode Comparison
    Compares hybrid configs with pure disaggregated and replica.
    """
    print("Plotting C3: Hybrid vs Pure...")

    valid_df = df[df['success_rate'] >= 95].copy()

    configs = ['4R', '2P_2D', '2P_2pD', '1R_1P_2pD', '1R_1P_1D_1pD']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    workloads = ['small_tiny', 'small_mid_bal', 'small_big_paste']
    metrics = [('t2_avg_ttft', 'T2 TTFT (ms)'), ('throughput_tps', 'Throughput (tps)')]

    for col, wl in enumerate(workloads):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            wl_data = valid_df[valid_df['workload'] == wl]

            for config in configs:
                cfg_data = wl_data[wl_data['config'] == config].sort_values('qps')
                if len(cfg_data) > 0:
                    ax.plot(cfg_data['qps'], cfg_data[metric], 'o-',
                           label=config, color=CONFIG_COLORS.get(config, '#333'),
                           linewidth=2, markersize=5)

            ax.set_xlabel('QPS')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{wl}')
            if col == 2 and row == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Hybrid vs Pure Mode Comparison', fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.88, 0.98])
    plt.savefig(output_dir / 'Part3_Fig1_hybrid_vs_pure.png')
    plt.close()
    print("  Saved Part3_Fig1_hybrid_vs_pure.png")


# =============================================================================
# New Figures: Part3_Fig2 (Throughput), Part4_Fig1 (1P Scalability)
# =============================================================================

def plot_Part3_Fig2_throughput_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Part3_Fig2: Throughput Comparison Across Modes

    Shows peak throughput by config, highlighting hybrid mode's hidden value.
    Key insight: 1R_2P_1pD achieves highest peak throughput (36.91 req/s)
    but is underrepresented in single-metric winner matrices.
    """
    print("Plotting Part3_Fig2: Throughput Comparison...")

    valid_df = df[df['success_rate'] >= 95].copy()

    # Add category if not present
    if 'config_category' not in valid_df.columns:
        valid_df['config_category'] = valid_df['config'].map(CONFIG_CATEGORIES)

    # Get peak throughput per config
    peak_throughput = valid_df.groupby('config')['throughput_tps'].max().reset_index()
    peak_throughput['category'] = peak_throughput['config'].map(CONFIG_CATEGORIES)

    # Sort by category then by throughput
    category_order = ['Replica', 'PD', 'PPD', 'Mixed_DpD', 'Hybrid']
    peak_throughput['cat_order'] = peak_throughput['category'].map(
        {c: i for i, c in enumerate(category_order)}
    )
    peak_throughput = peak_throughput.sort_values(['cat_order', 'throughput_tps'],
                                                   ascending=[True, False])

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left panel: Bar chart by config (sorted by throughput within category)
    ax1 = axes[0]
    colors = [MODE_COLORS.get(cat, '#333') for cat in peak_throughput['category']]
    y_positions = range(len(peak_throughput))

    bars = ax1.barh(y_positions, peak_throughput['throughput_tps'],
                    color=colors, edgecolor='black', linewidth=0.5)

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(peak_throughput['config'], fontsize=9)
    ax1.set_xlabel('Peak Throughput (tokens/sec)', fontsize=11, fontweight='bold')
    ax1.set_title('Peak Throughput by Configuration', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_axisbelow(True)

    # Highlight top 3 with labels
    top3 = peak_throughput.nlargest(3, 'throughput_tps')
    for rank, (idx, row) in enumerate(top3.iterrows(), 1):
        y_idx = list(peak_throughput.index).index(idx)
        ax1.text(row['throughput_tps'] + 10, y_idx, f'#{rank}',
                fontsize=10, fontweight='bold', va='center', color='darkgreen')

    # Add value labels on bars
    for i, (idx, row) in enumerate(peak_throughput.iterrows()):
        ax1.text(row['throughput_tps'] - 5, i, f'{row["throughput_tps"]:.0f}',
                ha='right', va='center', fontsize=8, color='white', fontweight='bold')

    # Right panel: Box plot by category showing distribution
    ax2 = axes[1]
    categories = []
    throughputs = []
    for cat in category_order:
        cat_data = valid_df[valid_df['config_category'] == cat]['throughput_tps']
        if len(cat_data) > 0:
            categories.append(cat)
            throughputs.append(cat_data.values)

    bp = ax2.boxplot(throughputs, labels=categories, patch_artist=True, vert=True)
    for patch, cat in zip(bp['boxes'], categories):
        patch.set_facecolor(MODE_COLORS.get(cat, '#333'))
        patch.set_alpha(0.7)

    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Mode Category', fontsize=11, fontweight='bold')
    ax2.set_title('Throughput Distribution by Mode', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)

    # Add annotation for hybrid peak
    hybrid_data = valid_df[valid_df['config_category'] == 'Hybrid']
    if len(hybrid_data) > 0:
        hybrid_max = hybrid_data['throughput_tps'].max()
        hybrid_config = hybrid_data[hybrid_data['throughput_tps'] == hybrid_max]['config'].iloc[0]
        # Find x position of Hybrid box
        hybrid_x = categories.index('Hybrid') + 1 if 'Hybrid' in categories else 5
        ax2.annotate(f'Peak: {hybrid_config}\n({hybrid_max:.0f} tps)',
                    xy=(hybrid_x, hybrid_max), xytext=(hybrid_x - 0.8, hybrid_max * 1.1),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', edgecolor='#f39c12'))

    # Suptitle
    plt.suptitle('Throughput Analysis: Hybrid Mode Achieves Highest Peak\n'
                 'Despite being underrepresented in single-metric winner counts',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'Part3_Fig2_throughput_comparison.png', facecolor='white', dpi=300)
    plt.close()
    print("  Saved Part3_Fig2_throughput_comparison.png")


def plot_Part4_Fig1_1p_scalability(df: pd.DataFrame, output_dir: Path):
    """
    Part4_Fig1: 1P Configuration Scalability Analysis

    Shows how ALL 1P configurations degrade with QPS, demonstrating:
    - pD variants degrade slower than pure D variants
    - Even with 1P bottleneck, pD provides resilience

    Key insight: 1P_3pD degrades slower than 1P_3D, proving pD value.

    SINGLE PANEL: Absolute values only (normalized panel removed per user request)
    """
    print("Plotting Part4_Fig1: 1P Scalability Comparison...")

    valid_df = df[df['success_rate'] >= 50].copy()  # Lower threshold to show degradation

    # ALL 1P configurations (ordered from pure D to pure pD)
    configs_1p = ['1P_3D', '1P_2D_1pD', '1P_1D_2pD', '1P_3pD']

    # Reference configs for comparison
    reference_configs = ['4R', '2P_2pD']

    # Color scheme: gradient from pure D (blue shades) to pure pD (red shades)
    config_colors = {
        '1P_3D': '#1a5276',      # Dark blue - pure D (3D, 0pD)
        '1P_2D_1pD': '#3498db',  # Blue - mostly D (2D, 1pD)
        '1P_1D_2pD': '#e74c3c',  # Red - mostly pD (1D, 2pD)
        '1P_3pD': '#922b21',     # Dark red - pure pD (0D, 3pD)
        '4R': '#27ae60',         # Green - reference (Replica)
        '2P_2pD': '#f39c12',     # Orange - reference (balanced PPD)
    }

    # Line styles
    line_styles = {
        '1P_3D': '-',
        '1P_2D_1pD': '--',
        '1P_1D_2pD': '-.',
        '1P_3pD': ':',
        '4R': '-',
        '2P_2pD': '-',
    }

    # Markers
    markers = {
        '1P_3D': 'o',
        '1P_2D_1pD': 's',
        '1P_1D_2pD': '^',
        '1P_3pD': 'D',
        '4R': 'o',
        '2P_2pD': 's',
    }

    # Use a consistent workload
    workload = 'small_mid_bal'
    wl_data = valid_df[valid_df['workload'] == workload]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot 1P configs with emphasis (thicker lines)
    for config in configs_1p:
        cfg_data = wl_data[wl_data['config'] == config].sort_values('qps')
        if len(cfg_data) > 0:
            ax.plot(cfg_data['qps'], cfg_data['t2_avg_ttft'],
                   marker=markers.get(config, 'o'), linestyle=line_styles.get(config, '-'),
                   label=f'{config}', color=config_colors.get(config, '#333'),
                   linewidth=3, markersize=9, alpha=0.9)

    # Plot reference configs with thinner lines
    for config in reference_configs:
        cfg_data = wl_data[wl_data['config'] == config].sort_values('qps')
        if len(cfg_data) > 0:
            ax.plot(cfg_data['qps'], cfg_data['t2_avg_ttft'],
                   marker=markers.get(config, 's'), linestyle='-',
                   label=f'{config} (ref)', color=config_colors.get(config, '#333'),
                   linewidth=1.5, markersize=6, alpha=0.6)

    ax.set_xlabel('QPS', fontsize=12, fontweight='bold')
    ax.set_ylabel('T2 TTFT (ms)', fontsize=12, fontweight='bold')
    ax.set_title('1P Configuration Scalability: pD Degrades Slower than D\n'
                 f'(Workload: {workload})', fontsize=13, fontweight='bold')

    # Create legend with clear grouping
    ax.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add shaded region showing "danger zone" (QPS >= 6)
    ymin, ymax = ax.get_ylim()
    ax.axvspan(6, 20, alpha=0.08, color='red')
    ax.text(13, ymax * 0.95, 'High Stress Zone\n(QPS ≥ 6)',
           fontsize=10, ha='center', va='top', style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))

    # Add annotation showing pD advantage at specific QPS
    d_data = wl_data[wl_data['config'] == '1P_3D'].sort_values('qps')
    pd_data = wl_data[wl_data['config'] == '1P_3pD'].sort_values('qps')

    # Find QPS=8 data for comparison
    if len(d_data) > 0 and len(pd_data) > 0:
        d_at_8 = d_data[d_data['qps'] == 8]['t2_avg_ttft'].values
        pd_at_8 = pd_data[pd_data['qps'] == 8]['t2_avg_ttft'].values
        if len(d_at_8) > 0 and len(pd_at_8) > 0:
            d_val = d_at_8[0]
            pd_val = pd_at_8[0]
            improvement = (d_val - pd_val) / d_val * 100 if d_val > 0 else 0
            mid_y = (d_val + pd_val) / 2

            # Draw annotation
            ax.annotate(f'pD advantage:\n{improvement:.0f}% lower\nat QPS=8',
                       xy=(8, mid_y), xytext=(10, mid_y * 0.6),
                       fontsize=10, ha='left',
                       arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#d5f5e3',
                                edgecolor='#27ae60', linewidth=2))

            # Draw connecting line between the two points
            ax.plot([8, 8], [d_val, pd_val], color='#27ae60', linewidth=2,
                   linestyle='--', alpha=0.7)

    # Add text box explaining the color coding
    textstr = 'Color gradient:\nBlue = more D machines\nRed = more pD machines'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_dir / 'Part4_Fig1_1p_scalability_comparison.png',
                facecolor='white', dpi=300)
    plt.close()
    print("  Saved Part4_Fig1_1p_scalability_comparison.png")


def plot_Part2_Fig3_turn_scaling(output_dir: Path):
    """
    Part2_Fig3: Turn Scaling Validation

    Validates that PPD's T2+ TTFT advantage persists across multi-turn conversations.
    Two-panel design:
    - Panel A: Per-turn TTFT progression (8 turns) - shows PD degradation vs PPD stability
    - Panel B: T2+ TTFT vs turn count (2, 4, 8, 16) - confirms trend across turn counts

    Key insight: PPD shows 68% T2+ TTFT reduction that remains stable across 2-16 turns,
    while PD's T2+ TTFT increases with context size.
    """
    print("Plotting Part2_Fig3: Turn Scaling Validation...")

    # Load turn scaling data
    turn_df = load_turn_scaling_data()

    if len(turn_df) == 0:
        print("  No turn scaling data available, skipping...")
        return

    # Config colors
    config_colors = {
        '4R': '#2ecc71',       # Green - Replica
        '2P_2D': '#3498db',    # Blue - PD
        '2P_2pD': '#e74c3c',   # Red - PPD
        '1R_1P_2pD': '#f39c12',  # Orange - Hybrid
    }

    # Markers
    config_markers = {
        '4R': 'o',
        '2P_2D': 's',
        '2P_2pD': '^',
        '1R_1P_2pD': 'D',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Panel A: Per-turn TTFT progression (8 turns) =====
    ax1 = axes[0]

    # Get 8-turn data
    turn8_data = turn_df[turn_df['num_turns'] == 8]

    for _, row in turn8_data.iterrows():
        config = row['config']
        per_turn = row['per_turn_metrics']

        if per_turn and len(per_turn) > 0:
            turns = [m['turn'] for m in per_turn]
            ttfts = [m['avg_ttft_ms'] for m in per_turn]

            ax1.plot(turns, ttfts, marker=config_markers.get(config, 'o'),
                    color=config_colors.get(config, '#333'),
                    label=config, linewidth=2.5, markersize=8, alpha=0.9)

    ax1.set_xlabel('Turn Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('TTFT (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Turn TTFT Progression (8 turns, QPS=4)\n'
                  'PD increases with context; PPD stays flat',
                  fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.set_xticks(range(1, 9))

    # Add annotations
    # Find 2P_2D and 2P_2pD data for comparison
    pd_row = turn8_data[turn8_data['config'] == '2P_2D']
    ppd_row = turn8_data[turn8_data['config'] == '2P_2pD']

    if len(pd_row) > 0 and len(ppd_row) > 0:
        pd_per_turn = pd_row.iloc[0]['per_turn_metrics']
        ppd_per_turn = ppd_row.iloc[0]['per_turn_metrics']

        if pd_per_turn and ppd_per_turn:
            # Get T8 values
            pd_t8 = pd_per_turn[-1]['avg_ttft_ms']
            ppd_t8 = ppd_per_turn[-1]['avg_ttft_ms']

            # Annotate the difference
            ax1.annotate(f'PD: {pd_t8:.0f}ms\n(grows with context)',
                        xy=(8, pd_t8), xytext=(6.5, pd_t8 + 15),
                        fontsize=9, ha='center',
                        arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#d4e6f1', edgecolor='#3498db'))

            ax1.annotate(f'PPD: {ppd_t8:.0f}ms\n(stays flat)',
                        xy=(8, ppd_t8), xytext=(6.5, ppd_t8 - 15),
                        fontsize=9, ha='center',
                        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#e74c3c'))

    # ===== Panel B: T2+ TTFT vs turn count (2, 4, 8, 16) =====
    ax2 = axes[1]

    configs_to_show = ['4R', '2P_2D', '2P_2pD', '1R_1P_2pD']
    turn_counts = sorted(turn_df['num_turns'].unique())

    for config in configs_to_show:
        config_data = turn_df[turn_df['config'] == config].sort_values('num_turns')

        if len(config_data) > 0:
            ax2.plot(config_data['num_turns'], config_data['t2plus_ttft'],
                    marker=config_markers.get(config, 'o'),
                    color=config_colors.get(config, '#333'),
                    label=config, linewidth=2.5, markersize=10, alpha=0.9)

    ax2.set_xlabel('Number of Turns', fontsize=12, fontweight='bold')
    ax2.set_ylabel('T2+ Average TTFT (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('T2+ TTFT Stability Across Turn Counts\n'
                  'PPD advantage persists from 2 to 16 turns',
                  fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.set_xticks(turn_counts)

    # Add shaded region showing PPD stability zone (0-50ms)
    ax2.axhspan(0, 50, alpha=0.1, color='#27ae60', label='_nolegend_')
    ax2.text(max(turn_counts), 25, 'PPD stable zone\n(<50ms)',
            fontsize=9, ha='right', va='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#27ae60'))

    # Calculate and annotate improvement
    pd_data = turn_df[(turn_df['config'] == '2P_2D') & (turn_df['num_turns'] == 8)]
    ppd_data = turn_df[(turn_df['config'] == '2P_2pD') & (turn_df['num_turns'] == 8)]

    if len(pd_data) > 0 and len(ppd_data) > 0:
        pd_t2plus = pd_data.iloc[0]['t2plus_ttft']
        ppd_t2plus = ppd_data.iloc[0]['t2plus_ttft']
        improvement = (pd_t2plus - ppd_t2plus) / pd_t2plus * 100 if pd_t2plus > 0 else 0

        # Add summary box
        summary_text = (f'At 8 turns:\n'
                       f'PD T2+: {pd_t2plus:.1f}ms\n'
                       f'PPD T2+: {ppd_t2plus:.1f}ms\n'
                       f'Improvement: {improvement:.0f}%')
        props = dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', alpha=0.95, edgecolor='#f39c12')
        ax2.text(0.98, 0.98, summary_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    # Suptitle
    plt.suptitle('Turn Scaling Validation: PPD Advantage is Consistent Across Multi-Turn Conversations\n'
                 '68% T2+ TTFT reduction maintained from 2 to 16 turns',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'Part2_Fig3_turn_scaling.png', facecolor='white', dpi=300)
    plt.close()
    print("  Saved Part2_Fig3_turn_scaling.png")


def plot_Part2_Fig4_model_scaling(output_dir: Path):
    """
    Part2_Fig4: Model Scaling Analysis

    Shows that PPD advantage is consistent across different model sizes (8B, 14B).
    4 configurations × 3 metrics (T2 TTFT, TPOT, E2E) × 2 model sizes.

    Configs: 4R (Replica), 2P_2D (PD), 2P_2pD (PPD), 1R_1P_1D_1pD (Hybrid)
    """
    print("Plotting Part2_Fig4: Model Scaling Analysis...")

    # Load model scaling data
    model_df = load_model_scaling_data()

    if len(model_df) == 0:
        print("  No model scaling data available, skipping...")
        return

    # 4 configs to show: Replica, PD, PPD, Hybrid
    configs_to_show = ['4R', '2P_2D', '2P_2pD', '1R_1P_1D_1pD']
    config_labels = ['4R\n(Replica)', '2P_2D\n(PD)', '2P_2pD\n(PPD)', '1R_1P_1D_1pD\n(Hybrid)']

    model_df = model_df[model_df['config'].isin(configs_to_show)]

    if len(model_df) == 0:
        print("  No matching configs in model scaling data, skipping...")
        return

    # Config colors
    config_colors = {
        '4R': '#2ecc71',           # Green - Replica
        '2P_2D': '#3498db',        # Blue - PD
        '2P_2pD': '#e74c3c',       # Red - PPD
        '1R_1P_1D_1pD': '#f39c12', # Orange - Hybrid
    }

    # Get model sizes (should be 8B, 14B)
    model_sizes = ['8B', '14B']  # Fixed order
    available_sizes = model_df['model_size'].unique()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    x = np.arange(len(model_sizes))
    width = 0.18
    n_configs = len(configs_to_show)

    # ===== Panel A: T2 TTFT by model size =====
    ax1 = axes[0]

    for i, (config, label) in enumerate(zip(configs_to_show, config_labels)):
        config_data = model_df[model_df['config'] == config]
        values = []

        for size in model_sizes:
            size_data = config_data[config_data['model_size'] == size]
            if len(size_data) > 0:
                values.append(size_data.iloc[0]['t2_ttft'])
            else:
                values.append(np.nan)

        offset = (i - (n_configs - 1) / 2) * width
        bars = ax1.bar(x + offset, values, width,
                      label=config, color=config_colors.get(config, '#333'),
                      edgecolor='black', linewidth=0.5, alpha=0.9)

        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val) and val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xlabel('Model Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('T2 TTFT (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) T2 TTFT by Model Size\n(lower is better)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_sizes)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_axisbelow(True)

    # ===== Panel B: TPOT by model size =====
    ax2 = axes[1]

    for i, (config, label) in enumerate(zip(configs_to_show, config_labels)):
        config_data = model_df[model_df['config'] == config]
        values = []

        for size in model_sizes:
            size_data = config_data[config_data['model_size'] == size]
            if len(size_data) > 0:
                # Use average TPOT
                val = size_data.iloc[0].get('avg_tpot')
                if val is None or np.isnan(val):
                    val = size_data.iloc[0].get('t2_tpot', np.nan)
                values.append(val)
            else:
                values.append(np.nan)

        offset = (i - (n_configs - 1) / 2) * width
        bars = ax2.bar(x + offset, values, width,
                      label=config, color=config_colors.get(config, '#333'),
                      edgecolor='black', linewidth=0.5, alpha=0.9)

        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val) and val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_xlabel('Model Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg TPOT (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Average TPOT by Model Size\n(lower is better)', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_sizes)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_axisbelow(True)

    # ===== Panel C: E2E Latency by model size =====
    ax3 = axes[2]

    for i, (config, label) in enumerate(zip(configs_to_show, config_labels)):
        config_data = model_df[model_df['config'] == config]
        values = []

        for size in model_sizes:
            size_data = config_data[config_data['model_size'] == size]
            if len(size_data) > 0:
                # Use average E2E
                val = size_data.iloc[0].get('avg_e2e')
                if val is None or np.isnan(val):
                    val = size_data.iloc[0].get('t2_e2e', np.nan)
                values.append(val)
            else:
                values.append(np.nan)

        offset = (i - (n_configs - 1) / 2) * width
        bars = ax3.bar(x + offset, values, width,
                      label=config, color=config_colors.get(config, '#333'),
                      edgecolor='black', linewidth=0.5, alpha=0.9)

        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val) and val > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax3.set_xlabel('Model Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Avg E2E Latency (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Average E2E Latency by Model Size\n(lower is better)', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_sizes)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_axisbelow(True)

    # Suptitle
    plt.suptitle('Model Scaling Analysis: PPD Advantage is Model-Independent\n'
                 'PPD (2P_2pD) shows 70%+ T2 TTFT improvement over PD (2P_2D) across model sizes',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'Part2_Fig4_model_scaling.png', facecolor='white', dpi=300)
    plt.close()
    print("  Saved Part2_Fig4_model_scaling.png")


# =============================================================================
# Legacy Figures (retained for backward compatibility)
# =============================================================================

def plot_Fig3_failure_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    Fig3: Failure Region Heatmap

    Shows success rate for each (config, QPS) combination.
    Visualizes Finding 3 (Single-Prefill Bottleneck) and Finding 10 (Reliability Hotspots).

    Key insight: 1P configurations show clear failure bands at high QPS.
    """
    print("Plotting Fig3: Failure Heatmap...")

    # Get all configs and QPS levels
    all_configs = sorted(df['config'].unique())
    all_qps = sorted(df['qps'].unique())

    # Order configs by category for better visualization
    config_order = ['4R',  # Replica first
                    '1P_3D', '2P_2D', '3P_1D',  # PD
                    '1P_3pD', '2P_2pD', '3P_1pD',  # PPD
                    '1P_2D_1pD', '1P_1D_2pD', '2P_1D_1pD',  # Mixed
                    '1R_1P_2D', '1R_1P_2pD', '1R_1P_1D_1pD',  # Hybrid
                    '1R_2P_1D', '1R_2P_1pD', '2R_1P_1D', '2R_1P_1pD']
    config_order = [c for c in config_order if c in all_configs]

    # Compute average success rate for each (config, QPS)
    success_matrix = np.full((len(config_order), len(all_qps)), np.nan)

    for i, config in enumerate(config_order):
        for j, qps in enumerate(all_qps):
            subset = df[(df['config'] == config) & (df['qps'] == qps)]
            if len(subset) > 0:
                success_matrix[i, j] = subset['success_rate'].mean()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Custom colormap: Red (low) -> Yellow (medium) -> Green (high)
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('success', colors)

    # Plot heatmap
    im = ax.imshow(success_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    # Add text annotations
    for i in range(len(config_order)):
        for j in range(len(all_qps)):
            val = success_matrix[i, j]
            if not np.isnan(val):
                # Choose text color based on value
                text_color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=7, color=text_color, fontweight='bold')

    # Labels
    ax.set_xticks(range(len(all_qps)))
    ax.set_xticklabels([str(q) for q in all_qps], fontsize=10)
    ax.set_yticks(range(len(config_order)))
    ax.set_yticklabels(config_order, fontsize=9)

    ax.set_xlabel('QPS', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Success Rate (%)', fontsize=11, fontweight='bold')

    # Add category separators
    category_boundaries = [1, 4, 7, 10]  # After 4R, PD, PPD, Mixed
    for boundary in category_boundaries:
        if boundary < len(config_order):
            ax.axhline(y=boundary - 0.5, color='black', linewidth=2)

    # Add category labels on the right
    category_labels = [
        (0, 'Replica'),
        (2.5, 'PD'),
        (5.5, 'PPD'),
        (8.5, 'Mixed'),
        (13.5, 'Hybrid'),
    ]
    for y, label in category_labels:
        if y < len(config_order):
            ax.text(len(all_qps) + 0.5, y, label, ha='left', va='center',
                   fontsize=10, fontweight='bold', style='italic')

    # Title with key insight
    plt.title('Configuration Reliability by QPS Level\n'
              'Single-Prefill (1P) configurations fail at high QPS; 2:2 ratio is most stable',
              fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'Fig3_failure_heatmap.png', facecolor='white', dpi=300)
    plt.close()
    print("  Saved Fig3_failure_heatmap.png")


def plot_Fig5_qps_scalability(df: pd.DataFrame, output_dir: Path):
    """
    Fig5: QPS Scalability Curves

    Shows how T2 TTFT scales with QPS for key configurations.
    Visualizes Finding 5 (QPS Scalability Patterns).

    Key insight: Different configs have very different degradation rates.
    """
    print("Plotting Fig5: QPS Scalability...")

    valid_df = df[df['success_rate'] >= 80].copy()  # Use 80% threshold to show degradation

    # Select representative configs across categories
    configs_to_show = ['4R', '2P_2D', '2P_2pD', '1P_3D', '1R_1P_2pD']

    # Use a consistent workload
    workload = 'small_mid_bal'
    wl_data = valid_df[valid_df['workload'] == workload]

    # Enhanced colors
    config_colors = {
        '4R': '#1a73e8',         # Blue - Replica
        '2P_2D': '#ea8600',      # Orange - PD
        '2P_2pD': '#34a853',     # Green - PPD
        '1P_3D': '#ea4335',      # Red - Single-P PD
        '1R_1P_2pD': '#9334e6',  # Purple - Hybrid
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Raw T2 TTFT values
    ax1 = axes[0]
    for config in configs_to_show:
        cfg_data = wl_data[wl_data['config'] == config].sort_values('qps')
        if len(cfg_data) > 0:
            ax1.plot(cfg_data['qps'], cfg_data['t2_avg_ttft'], 'o-',
                    label=config, color=config_colors.get(config, '#333'),
                    linewidth=2.5, markersize=7, alpha=0.9)

    ax1.set_xlabel('QPS', fontsize=12, fontweight='bold')
    ax1.set_ylabel('T2 TTFT (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency vs QPS (Absolute Values)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add annotation for 1P_3D collapse
    cfg_1p3d = wl_data[wl_data['config'] == '1P_3D'].sort_values('qps')
    if len(cfg_1p3d) > 3:
        max_idx = cfg_1p3d['t2_avg_ttft'].idxmax()
        max_qps = cfg_1p3d.loc[max_idx, 'qps']
        max_val = cfg_1p3d.loc[max_idx, 't2_avg_ttft']
        ax1.annotate('1P Bottleneck\n(9.5x degradation)',
                    xy=(max_qps, max_val), xytext=(max_qps - 3, max_val * 0.7),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', edgecolor='red'))

    # Right plot: Normalized degradation (relative to low QPS baseline)
    ax2 = axes[1]
    for config in configs_to_show:
        cfg_data = wl_data[wl_data['config'] == config].sort_values('qps')
        if len(cfg_data) > 0:
            # Use lowest QPS as baseline
            baseline = cfg_data['t2_avg_ttft'].iloc[0]
            if baseline > 0:
                normalized = cfg_data['t2_avg_ttft'] / baseline
                ax2.plot(cfg_data['qps'], normalized, 'o-',
                        label=config, color=config_colors.get(config, '#333'),
                        linewidth=2.5, markersize=7, alpha=0.9)

    ax2.set_xlabel('QPS', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Degradation Factor (× baseline)', fontsize=12, fontweight='bold')
    ax2.set_title('Latency Degradation (Normalized)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')

    # Add degradation factor annotations
    degradation_info = {
        '4R': '1.4x',
        '2P_2pD': '2.2x',
        '2P_2D': '2.5x',
        '1P_3D': '9.5x',
    }

    # Title
    plt.suptitle(f'QPS Scalability Analysis (Workload: {workload})\n'
                 'Replica shows best scalability (1.4x); Single-P shows worst (9.5x)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'Fig5_qps_scalability.png', facecolor='white', dpi=300)
    plt.close()
    print("  Saved Fig5_qps_scalability.png")


# =============================================================================
# F-Series: Decision Guide (for internal use)
# =============================================================================

def plot_F1_decision_table(df: pd.DataFrame, output_dir: Path):
    """F1: Decision Table for internal analysis."""
    print("Plotting F1: Decision Table (internal)...")

    valid_df = df[df['success_rate'] >= 95].copy()

    qps_levels = ['Low (0.5-2)', 'Medium (4-8)', 'High (10-20)']
    qps_ranges = [(0.5, 2), (4, 8), (10, 20)]

    workload_cats = ['Decode-heavy', 'Balanced', 'Prefill-heavy']

    results = []
    for wl_cat in workload_cats:
        for qps_label, (qps_min, qps_max) in zip(qps_levels, qps_ranges):
            subset = valid_df[
                (valid_df['workload_category'] == wl_cat) &
                (valid_df['qps'] >= qps_min) &
                (valid_df['qps'] <= qps_max)
            ]

            if len(subset) == 0:
                results.append({
                    'Workload': wl_cat,
                    'QPS Level': qps_label,
                    'Best for TTFT': 'N/A',
                    'Best for TPOT': 'N/A',
                    'Best for Throughput': 'N/A',
                })
                continue

            ttft_winner = subset.loc[subset['t2_avg_ttft'].idxmin(), 'config']
            tpot_winner = subset.loc[subset['avg_tpot'].idxmin(), 'config']
            tp_winner = subset.loc[subset['throughput_tps'].idxmax(), 'config']

            results.append({
                'Workload': wl_cat,
                'QPS Level': qps_label,
                'Best for TTFT': ttft_winner,
                'Best for TPOT': tpot_winner,
                'Best for Throughput': tp_winner,
            })

    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(results_df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for i in range(len(results_df.columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.title('Configuration Recommendation Guide', fontsize=14, pad=20)
    plt.savefig(output_dir / 'F1_decision_table.png', bbox_inches='tight')
    plt.close()
    print("  Saved F1_decision_table.png")


def plot_F2_slo_selection(df: pd.DataFrame, output_dir: Path):
    """F2: SLO-based Selection for internal analysis."""
    print("Plotting F2: SLO Selection (internal)...")

    valid_df = df[df['success_rate'] >= 95].copy()

    ttft_slos = [50, 100, 200, 500]
    tpot_slos = [20, 30, 50, 100]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    compliance_ttft = {}
    for slo in ttft_slos:
        configs_meeting = valid_df[valid_df['t2_avg_ttft'] <= slo]['config'].unique()
        compliance_ttft[slo] = len(configs_meeting)

    ax1.bar(range(len(ttft_slos)), list(compliance_ttft.values()), color='#3498db', edgecolor='black')
    ax1.set_xticks(range(len(ttft_slos)))
    ax1.set_xticklabels([f'<{slo}ms' for slo in ttft_slos])
    ax1.set_xlabel('T2 TTFT SLO')
    ax1.set_ylabel('Number of Configs Meeting SLO')
    ax1.set_title('T2 TTFT SLO Compliance')

    ax2 = axes[1]
    compliance_tpot = {}
    for slo in tpot_slos:
        configs_meeting = valid_df[valid_df['avg_tpot'] <= slo]['config'].unique()
        compliance_tpot[slo] = len(configs_meeting)

    ax2.bar(range(len(tpot_slos)), list(compliance_tpot.values()), color='#e74c3c', edgecolor='black')
    ax2.set_xticks(range(len(tpot_slos)))
    ax2.set_xticklabels([f'<{slo}ms' for slo in tpot_slos])
    ax2.set_xlabel('Avg TPOT SLO')
    ax2.set_ylabel('Number of Configs Meeting SLO')
    ax2.set_title('Avg TPOT SLO Compliance')

    plt.suptitle('SLO-based Configuration Selection', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / 'F2_slo_selection.png')
    plt.close()
    print("  Saved F2_slo_selection.png")


# =============================================================================
# H-Series: Supplementary Figures
# =============================================================================

def plot_H1_full_workload_matrix(df: pd.DataFrame, output_dir: Path):
    """H1: Full Workload Matrix."""
    print("Plotting H1: Full Workload Matrix...")

    valid_df = df[df['success_rate'] >= 95].copy()

    qps = 4
    subset = valid_df[valid_df['qps'] == qps]

    metrics = [
        ('t2_avg_ttft', 'T2 TTFT (ms)', 'YlOrRd'),
        ('avg_tpot', 'Avg TPOT (ms)', 'YlOrRd'),
        ('throughput_tps', 'Throughput (tps)', 'YlGn'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for ax, (metric, title, cmap) in zip(axes, metrics):
        pivot = subset.pivot_table(
            values=metric,
            index='config',
            columns='workload',
            aggfunc='mean'
        )

        config_order = ['4R'] + sorted([c for c in pivot.index if c != '4R'])
        pivot = pivot.reindex(config_order)

        sns.heatmap(pivot, annot=True, fmt='.0f', cmap=cmap, ax=ax,
                   cbar_kws={'label': title}, annot_kws={'fontsize': 7})
        ax.set_title(f'{title} @ QPS={qps}')
        ax.set_xlabel('Workload')
        ax.set_ylabel('Config')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle('Full Workload Performance Matrix', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / 'H1_full_workload_matrix.png')
    plt.close()
    print("  Saved H1_full_workload_matrix.png")


def plot_H2_percentile_analysis(df: pd.DataFrame, output_dir: Path):
    """H2: Percentile Analysis."""
    print("Plotting H2: Percentile Analysis...")

    valid_df = df[df['success_rate'] >= 95].copy()

    configs = REPRESENTATIVE_CONFIGS
    workload = 'small_mid_bal'

    subset = valid_df[(valid_df['workload'] == workload) & (valid_df['config'].isin(configs))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        (('t2_p50_ttft', 't2_p99_ttft'), 'T2 TTFT'),
        (('t1_p50_tpot', 't1_p99_tpot'), 'T1 TPOT'),
    ]

    for row, ((p50_col, p99_col), title) in enumerate(metrics):
        ax1 = axes[row, 0]
        for config in configs:
            cfg_data = subset[subset['config'] == config]
            if len(cfg_data) > 0:
                ax1.scatter(cfg_data[p50_col], cfg_data[p99_col],
                           color=CONFIG_COLORS.get(config, '#333'),
                           label=config, s=50, alpha=0.7)

        max_val = max(subset[p50_col].max(), subset[p99_col].max()) * 1.1
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='p50=p99')
        ax1.set_xlabel(f'{title} p50 (ms)')
        ax1.set_ylabel(f'{title} p99 (ms)')
        ax1.set_title(f'{title}: p50 vs p99')
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[row, 1]
        subset_copy = subset.copy()
        subset_copy['ratio'] = subset_copy[p99_col] / subset_copy[p50_col].replace(0, np.nan)

        for config in configs:
            cfg_data = subset_copy[subset_copy['config'] == config].sort_values('qps')
            if len(cfg_data) > 0:
                ax2.plot(cfg_data['qps'], cfg_data['ratio'], 'o-',
                        color=CONFIG_COLORS.get(config, '#333'),
                        label=config, linewidth=1.5, markersize=4)

        ax2.set_xlabel('QPS')
        ax2.set_ylabel('p99/p50 Ratio')
        ax2.set_title(f'{title}: Tail Latency Ratio vs QPS')
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.3)
        ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Percentile Analysis (Workload: {workload})', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / 'H2_percentile_analysis.png')
    plt.close()
    print("  Saved H2_percentile_analysis.png")


def plot_H3_all_config_comparison(df: pd.DataFrame, output_dir: Path):
    """H3: All Config Comparison."""
    print("Plotting H3: All Config Comparison...")

    valid_df = df[df['success_rate'] >= 95].copy()

    avg_by_config = valid_df.groupby('config').agg({
        't2_avg_ttft': 'mean',
        'avg_tpot': 'mean',
        'throughput_tps': 'mean',
    }).reset_index()

    avg_by_config = avg_by_config.sort_values('t2_avg_ttft')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ('t2_avg_ttft', 'T2 TTFT (ms)', True),
        ('avg_tpot', 'Avg TPOT (ms)', True),
        ('throughput_tps', 'Throughput (tps)', False),
    ]

    for ax, (metric, ylabel, ascending) in zip(axes, metrics):
        sorted_df = avg_by_config.sort_values(metric, ascending=ascending)
        colors = [CONFIG_COLORS.get(c, '#333') for c in sorted_df['config']]

        bars = ax.barh(range(len(sorted_df)), sorted_df[metric].values, color=colors, edgecolor='black')
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['config'])
        ax.set_xlabel(ylabel)
        ax.set_title(f'All Configs by {ylabel}')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('All Configuration Comparison (Averaged)', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_dir / 'H3_all_config_comparison.png')
    plt.close()
    print("  Saved H3_all_config_comparison.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PPD TRADE-OFF VISUALIZATION (v2 - 4 Parts Structure)")
    print("=" * 70)

    # Create output directories
    paper_dir = OUTPUT_DIR / "paper"
    internal_dir = OUTPUT_DIR / "internal"
    supp_dir = OUTPUT_DIR / "supplementary"
    paper_dir.mkdir(parents=True, exist_ok=True)
    internal_dir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()

    print("\n" + "=" * 70)
    print("Part 1: Objective-Oriented Configuration Selection")
    print("=" * 70)
    plot_Fig1_objective_oriented(df, paper_dir)  # Part1_Fig1 (will be renamed)

    print("\n" + "=" * 70)
    print("Part 2: PPD vs PD Analysis")
    print("=" * 70)
    plot_C2_d_vs_pd_comparison(df, paper_dir)    # Part2_Fig1
    plot_C1_pd_ratio_heatmap(df, paper_dir)      # Part2_Fig2
    plot_Part2_Fig3_turn_scaling(paper_dir)       # Part2_Fig3 (NEW - Turn Scaling)
    plot_Part2_Fig4_model_scaling(paper_dir)      # Part2_Fig4 (NEW - Model Scaling)

    print("\n" + "=" * 70)
    print("Part 3: Mode Trade-offs and Throughput")
    print("=" * 70)
    plot_C3_hybrid_vs_pure(df, paper_dir)              # Part3_Fig1
    plot_Part3_Fig2_throughput_comparison(df, paper_dir)  # Part3_Fig2 (NEW)

    print("\n" + "=" * 70)
    print("Part 4: Scalability and Reliability")
    print("=" * 70)
    plot_Part4_Fig1_1p_scalability(df, paper_dir)  # Part4_Fig1 (MODIFIED - single panel)

    print("\n" + "=" * 70)
    print("Generating Legacy Figures (for backward compatibility)")
    print("=" * 70)
    # These are kept for reference but may be archived
    plot_Fig3_failure_heatmap(df, paper_dir)
    plot_A3_improved_radar(df, paper_dir)

    print("\n" + "=" * 70)
    print("Generating F-Series: Decision Guide (Internal)")
    print("=" * 70)
    plot_F1_decision_table(df, internal_dir)
    plot_F2_slo_selection(df, internal_dir)

    print("\n" + "=" * 70)
    print("Generating H-Series: Supplementary Figures")
    print("=" * 70)
    plot_H1_full_workload_matrix(df, supp_dir)
    plot_H2_percentile_analysis(df, supp_dir)
    plot_H3_all_config_comparison(df, supp_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nPaper figures saved to: {paper_dir}")
    print(f"Internal figures saved to: {internal_dir}")
    print(f"Supplementary figures saved to: {supp_dir}")

    # Summary
    paper_files = list(paper_dir.glob("*.png"))
    internal_files = list(internal_dir.glob("*.png"))
    supp_files = list(supp_dir.glob("*.png"))
    print(f"\nGenerated {len(paper_files)} paper figures, {len(internal_files)} internal, {len(supp_files)} supplementary")

    print("\n" + "=" * 70)
    print("Figure Naming for 4-Part Structure:")
    print("=" * 70)
    print("Part 1: Part1_Fig1_objective_oriented.png")
    print("Part 2: Part2_Fig1_d_vs_pd_comparison.png, Part2_Fig2_pd_ratio_heatmap.png")
    print("        Part2_Fig3_turn_scaling.png (NEW), Part2_Fig4_model_scaling.png (NEW)")
    print("Part 3: Part3_Fig1_hybrid_vs_pure.png, Part3_Fig2_throughput_comparison.png")
    print("Part 4: Part4_Fig1_1p_scalability_comparison.png")
    print("\nArchived: Fig3_failure_heatmap.png, A3_improved_radar.png, H2_percentile_analysis.png")


if __name__ == "__main__":
    main()
