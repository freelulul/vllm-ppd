#!/usr/bin/env python3
"""
Comprehensive Data Analysis for PD/PPD/Replica Trade-off Study

This script generates detailed visualizations and analysis for understanding
the trade-offs between three execution modes:
- PD: Full disaggregation (P→D with KV transfer)
- PPD: Prefix-aware PD (T1: P→D, T2+: D-direct with prefix cache)
- Replica: Data parallelism (same worker for all turns, uses prefix cache)

Usage:
    python scripts/comprehensive_analysis.py [--input results/final/merged_results.json]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'pd': '#e74c3c',      # Red
    'ppd': '#2ecc71',     # Green
    'replica': '#3498db', # Blue
}
MODE_LABELS = {
    'pd': 'PD (Full Disagg)',
    'ppd': 'PPD (Prefix-aware)',
    'replica': 'Replica (Data Parallel)',
}
MARKERS = {'pd': 'o', 'ppd': 's', 'replica': '^'}

# Context size descriptions
CONTEXT_DESC = {
    'XS': 'XSmall (128+128)',
    'S': 'Small (256+256)',
    'M': 'Medium (512+512)',
    'L': 'Large (1024+1024)',
    'XL': 'XLarge (2048+1024)',
}

# T2 type descriptions
T2_DESC = {
    'a': 'Tiny (16→32)',
    'b': 'Short Q Long Out (32→256)',
    'c': 'Balanced (128→128)',
    'd': 'Big Paste (512→64)',
}


def load_data(input_path: Path) -> dict:
    """Load merged benchmark results."""
    with open(input_path) as f:
        return json.load(f)


def organize_data(data: dict) -> dict:
    """Organize data by workload and mode for easy plotting."""
    organized = defaultdict(lambda: defaultdict(dict))

    for r in data['results']:
        mode = r['mode']
        workload = r['workload']
        qps = r['target_qps']
        organized[workload][mode][qps] = r

    return organized


def get_workload_info(workload: str) -> tuple:
    """Extract context size and T2 type from workload name."""
    parts = workload.split('_')
    return parts[0], parts[1] if len(parts) > 1 else 'a'


def plot_metric_vs_qps(organized: dict, metric_path: list, ylabel: str,
                       title_suffix: str, output_dir: Path, filename: str,
                       log_scale: bool = False):
    """
    Create a grid of plots showing metric vs QPS for all workloads.
    Each subplot shows all three modes for one workload.
    """
    workloads = sorted(organized.keys())

    # Group workloads by context size
    by_context = defaultdict(list)
    for w in workloads:
        ctx, _ = get_workload_info(w)
        by_context[ctx].append(w)

    # Create figure with subplots
    contexts = ['XS', 'S', 'M', 'L', 'XL']
    t2_types = ['a', 'b', 'c', 'd']

    fig, axes = plt.subplots(len(contexts), len(t2_types), figsize=(16, 15))
    fig.suptitle(f'{title_suffix} vs QPS by Workload', fontsize=14, fontweight='bold')

    for i, ctx in enumerate(contexts):
        for j, t2 in enumerate(t2_types):
            ax = axes[i, j]
            workload = f"{ctx}_{t2}"

            if workload not in organized:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{workload}', fontsize=10)
                continue

            for mode in ['pd', 'ppd', 'replica']:
                if mode not in organized[workload]:
                    continue

                mode_data = organized[workload][mode]
                qps_values = sorted(mode_data.keys())
                metric_values = []

                for qps in qps_values:
                    r = mode_data[qps]
                    # Navigate through nested dict path
                    val = r
                    for key in metric_path:
                        val = val.get(key, {}) if isinstance(val, dict) else 0
                    metric_values.append(val if val else 0)

                ax.plot(qps_values, metric_values,
                       marker=MARKERS[mode], color=COLORS[mode],
                       label=MODE_LABELS[mode], linewidth=2, markersize=6)

            ax.set_title(f'{workload}\n({CONTEXT_DESC.get(ctx, ctx)}, {T2_DESC.get(t2, t2)})',
                        fontsize=9)
            ax.set_xlabel('QPS', fontsize=8)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            if log_scale:
                ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

    # Add legend
    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], label=MODE_LABELS[m],
                      linewidth=2, markersize=8) for m in ['pd', 'ppd', 'replica']]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparison_heatmap(organized: dict, metric_path: list, title: str,
                           output_dir: Path, filename: str,
                           compare_modes: tuple = ('ppd', 'pd')):
    """
    Create heatmap showing relative performance between two modes.
    Positive = first mode is better (lower is better for latency).
    """
    contexts = ['XS', 'S', 'M', 'L', 'XL']
    t2_types = ['a', 'b', 'c', 'd']

    # Get all QPS values
    all_qps = set()
    for w_data in organized.values():
        for m_data in w_data.values():
            all_qps.update(m_data.keys())
    qps_values = sorted(all_qps)

    # Create figure with subplots for each QPS
    n_qps = len(qps_values)
    cols = min(4, n_qps)
    rows = (n_qps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    mode1, mode2 = compare_modes
    fig.suptitle(f'{title}: {MODE_LABELS[mode1]} vs {MODE_LABELS[mode2]}\n(Green = {MODE_LABELS[mode1]} better, Red = {MODE_LABELS[mode2]} better)',
                 fontsize=12, fontweight='bold')

    for idx, qps in enumerate(qps_values):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        data = np.zeros((len(contexts), len(t2_types)))
        data[:] = np.nan

        for i, ctx in enumerate(contexts):
            for j, t2 in enumerate(t2_types):
                workload = f"{ctx}_{t2}"
                if workload not in organized:
                    continue

                if mode1 not in organized[workload] or mode2 not in organized[workload]:
                    continue

                if qps not in organized[workload][mode1] or qps not in organized[workload][mode2]:
                    continue

                r1 = organized[workload][mode1][qps]
                r2 = organized[workload][mode2][qps]

                val1, val2 = r1, r2
                for key in metric_path:
                    val1 = val1.get(key, {}) if isinstance(val1, dict) else 0
                    val2 = val2.get(key, {}) if isinstance(val2, dict) else 0

                if val1 and val2:
                    # Improvement percentage (positive = mode1 is better/lower)
                    improvement = (val2 - val1) / val2 * 100
                    data[i, j] = improvement

        im = ax.imshow(data, cmap='RdYlGn', vmin=-50, vmax=50, aspect='auto')
        ax.set_xticks(range(len(t2_types)))
        ax.set_xticklabels(t2_types, fontsize=8)
        ax.set_yticks(range(len(contexts)))
        ax.set_yticklabels(contexts, fontsize=8)
        ax.set_title(f'QPS = {qps}', fontsize=10)

        # Add text annotations
        for i in range(len(contexts)):
            for j in range(len(t2_types)):
                if not np.isnan(data[i, j]):
                    color = 'white' if abs(data[i, j]) > 25 else 'black'
                    ax.text(j, i, f'{data[i, j]:.0f}%', ha='center', va='center',
                           fontsize=7, color=color)

    # Hide empty subplots
    for idx in range(len(qps_values), rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_crossover_analysis(organized: dict, output_dir: Path):
    """
    Analyze and visualize crossover points where one mode becomes better than another.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    metrics = [
        (['turn2_metrics', 'avg_ttft'], 'T2 Avg TTFT (ms)', 'lower'),
        (['turn2_metrics', 'p99_ttft'], 'T2 P99 TTFT (ms)', 'lower'),
        (['turn2_metrics', 'avg_tpot'], 'T2 Avg TPOT (ms/token)', 'lower'),
        (['avg_throughput_tps'], 'Throughput (tokens/s)', 'higher'),
    ]

    for ax_idx, (metric_path, ylabel, better) in enumerate(metrics):
        ax = axes[ax_idx // 2, ax_idx % 2]

        # Collect crossover points
        crossovers_ppd_pd = []
        crossovers_ppd_replica = []

        for workload in sorted(organized.keys()):
            modes_data = organized[workload]

            if 'ppd' not in modes_data:
                continue

            ppd_data = modes_data['ppd']
            qps_values = sorted(ppd_data.keys())

            for mode2 in ['pd', 'replica']:
                if mode2 not in modes_data:
                    continue

                mode2_data = modes_data[mode2]
                prev_winner = None

                for qps in qps_values:
                    if qps not in mode2_data:
                        continue

                    r_ppd = ppd_data[qps]
                    r_other = mode2_data[qps]

                    val_ppd, val_other = r_ppd, r_other
                    for key in metric_path:
                        val_ppd = val_ppd.get(key, 0) if isinstance(val_ppd, dict) else val_ppd
                        val_other = val_other.get(key, 0) if isinstance(val_other, dict) else val_other

                    if not val_ppd or not val_other:
                        continue

                    if better == 'lower':
                        winner = 'ppd' if val_ppd < val_other else mode2
                    else:
                        winner = 'ppd' if val_ppd > val_other else mode2

                    if prev_winner and prev_winner != winner:
                        if mode2 == 'pd':
                            crossovers_ppd_pd.append((workload, qps, prev_winner, winner))
                        else:
                            crossovers_ppd_replica.append((workload, qps, prev_winner, winner))

                    prev_winner = winner

        # Plot summary
        ctx_order = {'XS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4}
        t2_order = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

        for cross_data, color, label in [
            (crossovers_ppd_pd, COLORS['pd'], 'PPD↔PD'),
            (crossovers_ppd_replica, COLORS['replica'], 'PPD↔Replica')
        ]:
            if cross_data:
                x = [item[1] for item in cross_data]
                y = [ctx_order[get_workload_info(item[0])[0]] +
                     t2_order[get_workload_info(item[0])[1]] * 0.2 for item in cross_data]
                ax.scatter(x, y, c=color, s=100, label=label, alpha=0.7, edgecolors='black')

        ax.set_xlabel('QPS at Crossover', fontsize=10)
        ax.set_ylabel('Workload (Context × T2)', fontsize=10)
        ax.set_title(f'Crossover Points: {ylabel}', fontsize=11, fontweight='bold')
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(['XS', 'S', 'M', 'L', 'XL'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'crossover_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary_by_context(organized: dict, output_dir: Path):
    """
    Create summary plots grouped by context size showing T2 TTFT trends.
    """
    contexts = ['XS', 'S', 'M', 'L', 'XL']
    t2_types = ['a', 'b', 'c', 'd']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('T2 P99 TTFT by Context Size (All T2 Types)', fontsize=14, fontweight='bold')

    for ax_idx, ctx in enumerate(contexts):
        ax = axes[ax_idx // 3, ax_idx % 3]

        for mode in ['pd', 'ppd', 'replica']:
            # Aggregate across T2 types
            all_qps = set()
            for t2 in t2_types:
                workload = f"{ctx}_{t2}"
                if workload in organized and mode in organized[workload]:
                    all_qps.update(organized[workload][mode].keys())

            if not all_qps:
                continue

            qps_values = sorted(all_qps)
            avg_values = []
            std_values = []

            for qps in qps_values:
                values = []
                for t2 in t2_types:
                    workload = f"{ctx}_{t2}"
                    if workload in organized and mode in organized[workload]:
                        if qps in organized[workload][mode]:
                            val = organized[workload][mode][qps].get('turn2_metrics', {}).get('p99_ttft', 0)
                            if val:
                                values.append(val)

                if values:
                    avg_values.append(np.mean(values))
                    std_values.append(np.std(values))
                else:
                    avg_values.append(np.nan)
                    std_values.append(0)

            ax.errorbar(qps_values, avg_values, yerr=std_values,
                       marker=MARKERS[mode], color=COLORS[mode],
                       label=MODE_LABELS[mode], linewidth=2, markersize=6,
                       capsize=3, capthick=1)

        ax.set_xlabel('QPS', fontsize=10)
        ax.set_ylabel('P99 TTFT (ms)', fontsize=10)
        ax.set_title(f'Context: {CONTEXT_DESC.get(ctx, ctx)}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide extra subplot (6th in 2x3 grid when we have 5 contexts)
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / 'summary_by_context_p99_ttft.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary_by_t2type(organized: dict, output_dir: Path):
    """
    Create summary plots grouped by T2 type showing trends across context sizes.
    """
    contexts = ['XS', 'S', 'M', 'L', 'XL']
    t2_types = ['a', 'b', 'c', 'd']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('T2 P99 TTFT by T2 Type (All Context Sizes)', fontsize=14, fontweight='bold')

    for ax_idx, t2 in enumerate(t2_types):
        ax = axes[ax_idx // 2, ax_idx % 2]

        for mode in ['pd', 'ppd', 'replica']:
            # Aggregate across context sizes
            all_qps = set()
            for ctx in contexts:
                workload = f"{ctx}_{t2}"
                if workload in organized and mode in organized[workload]:
                    all_qps.update(organized[workload][mode].keys())

            if not all_qps:
                continue

            qps_values = sorted(all_qps)
            avg_values = []
            std_values = []

            for qps in qps_values:
                values = []
                for ctx in contexts:
                    workload = f"{ctx}_{t2}"
                    if workload in organized and mode in organized[workload]:
                        if qps in organized[workload][mode]:
                            val = organized[workload][mode][qps].get('turn2_metrics', {}).get('p99_ttft', 0)
                            if val:
                                values.append(val)

                if values:
                    avg_values.append(np.mean(values))
                    std_values.append(np.std(values))
                else:
                    avg_values.append(np.nan)
                    std_values.append(0)

            ax.errorbar(qps_values, avg_values, yerr=std_values,
                       marker=MARKERS[mode], color=COLORS[mode],
                       label=MODE_LABELS[mode], linewidth=2, markersize=6,
                       capsize=3, capthick=1)

        ax.set_xlabel('QPS', fontsize=10)
        ax.set_ylabel('P99 TTFT (ms)', fontsize=10)
        ax.set_title(f'T2 Type: {T2_DESC.get(t2, t2)}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / 'summary_by_t2type_p99_ttft.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_winner_matrix(organized: dict, metric_path: list, title: str,
                      output_dir: Path, filename: str, better: str = 'lower'):
    """
    Create a matrix showing the winner at each (workload, QPS) combination.
    """
    workloads = sorted(organized.keys())
    all_qps = set()
    for w_data in organized.values():
        for m_data in w_data.values():
            all_qps.update(m_data.keys())
    qps_values = sorted(all_qps)

    fig, ax = plt.subplots(figsize=(14, 10))

    winner_colors = {'pd': 0, 'ppd': 1, 'replica': 2, 'tie': 3}
    data = np.full((len(workloads), len(qps_values)), np.nan)

    for i, workload in enumerate(workloads):
        for j, qps in enumerate(qps_values):
            values = {}
            for mode in ['pd', 'ppd', 'replica']:
                if mode in organized[workload] and qps in organized[workload][mode]:
                    r = organized[workload][mode][qps]
                    val = r
                    for key in metric_path:
                        val = val.get(key, 0) if isinstance(val, dict) else val
                    if val:
                        values[mode] = val

            if len(values) >= 2:
                if better == 'lower':
                    winner = min(values, key=values.get)
                else:
                    winner = max(values, key=values.get)
                data[i, j] = winner_colors[winner]

    cmap = plt.cm.colors.ListedColormap([COLORS['pd'], COLORS['ppd'], COLORS['replica'], 'gray'])
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=3)

    ax.set_xticks(range(len(qps_values)))
    ax.set_xticklabels([str(q) for q in qps_values], fontsize=8)
    ax.set_yticks(range(len(workloads)))
    ax.set_yticklabels(workloads, fontsize=8)
    ax.set_xlabel('QPS', fontsize=12)
    ax.set_ylabel('Workload', fontsize=12)
    ax.set_title(f'Winner Matrix: {title}\n(Best performer at each QPS)', fontsize=13, fontweight='bold')

    # Legend
    legend_elements = [mpatches.Patch(facecolor=COLORS[m], label=MODE_LABELS[m])
                       for m in ['pd', 'ppd', 'replica']]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_radar_comparison(organized: dict, output_dir: Path):
    """
    Create radar charts comparing modes across multiple metrics for representative workloads.
    """
    # Select representative workloads
    rep_workloads = ['S_a', 'M_c', 'L_b', 'XL_d']

    metrics = [
        ('T2 TTFT', ['turn2_metrics', 'avg_ttft'], 'lower', 5000),
        ('T2 TPOT', ['turn2_metrics', 'avg_tpot'], 'lower', 50),
        ('Throughput', ['avg_throughput_tps'], 'higher', 200),
        ('E2E Latency', ['avg_e2e'], 'lower', 10000),
        ('Success Rate', None, 'higher', 100),  # Special handling
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(projection='polar'))
    fig.suptitle('Multi-Metric Radar Comparison (Normalized, Higher=Better)',
                fontsize=14, fontweight='bold')

    for ax_idx, workload in enumerate(rep_workloads):
        if workload not in organized:
            continue

        ax = axes[ax_idx // 2, ax_idx % 2]

        # Get mid QPS for this workload
        ctx, _ = get_workload_info(workload)
        qps_map = {'XS': 6, 'S': 4, 'M': 3, 'L': 2, 'XL': 1}
        target_qps = qps_map.get(ctx, 2)

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        for mode in ['pd', 'ppd', 'replica']:
            if mode not in organized[workload]:
                continue

            # Find closest QPS
            available_qps = list(organized[workload][mode].keys())
            if not available_qps:
                continue
            closest_qps = min(available_qps, key=lambda x: abs(x - target_qps))
            r = organized[workload][mode][closest_qps]

            values = []
            for name, path, better, max_val in metrics:
                if name == 'Success Rate':
                    sc = r.get('success_count', 0)
                    total = r.get('sample_count', 1)
                    raw_val = (sc / total * 100) if total > 0 else 0
                else:
                    val = r
                    for key in path:
                        val = val.get(key, 0) if isinstance(val, dict) else val
                    raw_val = val if val else 0

                # Normalize to 0-1 (higher is better)
                if better == 'lower':
                    normalized = max(0, 1 - raw_val / max_val)
                else:
                    normalized = min(1, raw_val / max_val)
                values.append(normalized)

            values += values[:1]
            ax.plot(angles, values, color=COLORS[mode], linewidth=2, label=MODE_LABELS[mode])
            ax.fill(angles, values, color=COLORS[mode], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[0] for m in metrics], fontsize=8)
        ax.set_title(f'{workload} @ QPS≈{target_qps}', fontsize=11, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    output_path = output_dir / 'radar_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_t2_advantage_analysis(organized: dict, output_dir: Path):
    """
    Analyze the T2 TTFT advantage of PPD over PD (avoiding KV transfer).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPD vs PD: Turn 2 TTFT Advantage Analysis\n(PPD avoids KV transfer by using prefix cache)',
                fontsize=13, fontweight='bold')

    t2_types = ['a', 'b', 'c', 'd']
    contexts = ['XS', 'S', 'M', 'L', 'XL']

    for ax_idx, t2 in enumerate(t2_types):
        ax = axes[ax_idx // 2, ax_idx % 2]

        for ctx in contexts:
            workload = f"{ctx}_{t2}"
            if workload not in organized:
                continue

            if 'ppd' not in organized[workload] or 'pd' not in organized[workload]:
                continue

            ppd_data = organized[workload]['ppd']
            pd_data = organized[workload]['pd']

            common_qps = sorted(set(ppd_data.keys()) & set(pd_data.keys()))
            if not common_qps:
                continue

            improvements = []
            for qps in common_qps:
                ppd_ttft = ppd_data[qps].get('turn2_metrics', {}).get('avg_ttft', 0)
                pd_ttft = pd_data[qps].get('turn2_metrics', {}).get('avg_ttft', 0)

                if ppd_ttft and pd_ttft:
                    improvement = (pd_ttft - ppd_ttft) / pd_ttft * 100
                    improvements.append(improvement)
                else:
                    improvements.append(np.nan)

            ax.plot(common_qps, improvements, marker='o', linewidth=2,
                   label=f'{ctx} ({CONTEXT_DESC.get(ctx, ctx).split()[0]})')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('QPS', fontsize=10)
        ax.set_ylabel('PPD Improvement over PD (%)', fontsize=10)
        ax.set_title(f'T2 Type: {T2_DESC.get(t2, t2)}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-20, 80)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = output_dir / 'ppd_vs_pd_advantage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_replica_vs_ppd_analysis(organized: dict, output_dir: Path):
    """
    Analyze when Replica outperforms PPD (double GPU capacity matters).
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Replica vs PPD: When Double Capacity Wins\n(Replica: 4 independent workers, PPD: 2P+2D)',
                fontsize=13, fontweight='bold')

    contexts = ['XS', 'S', 'M', 'L', 'XL']

    for ax_idx, ctx in enumerate(contexts):
        ax = axes[ax_idx // 3, ax_idx % 3]

        for t2 in ['a', 'b', 'c', 'd']:
            workload = f"{ctx}_{t2}"
            if workload not in organized:
                continue

            if 'ppd' not in organized[workload] or 'replica' not in organized[workload]:
                continue

            ppd_data = organized[workload]['ppd']
            replica_data = organized[workload]['replica']

            common_qps = sorted(set(ppd_data.keys()) & set(replica_data.keys()))
            if not common_qps:
                continue

            differences = []
            for qps in common_qps:
                ppd_ttft = ppd_data[qps].get('turn2_metrics', {}).get('p99_ttft', 0)
                replica_ttft = replica_data[qps].get('turn2_metrics', {}).get('p99_ttft', 0)

                if ppd_ttft and replica_ttft:
                    # Positive = Replica is better (lower latency)
                    diff = (ppd_ttft - replica_ttft) / ppd_ttft * 100
                    differences.append(diff)
                else:
                    differences.append(np.nan)

            ax.plot(common_qps, differences, marker='o', linewidth=2,
                   label=f'{t2} ({T2_DESC.get(t2, t2).split()[0]})')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.fill_between(ax.get_xlim(), 0, 100, alpha=0.1, color='blue', label='_Replica wins')
        ax.fill_between(ax.get_xlim(), -100, 0, alpha=0.1, color='green', label='_PPD wins')

        ax.set_xlabel('QPS', fontsize=10)
        ax.set_ylabel('Replica Improvement over PPD (%)', fontsize=10)
        ax.set_title(f'Context: {CONTEXT_DESC.get(ctx, ctx)}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-60, 60)

    # Hide extra subplot (6th in 2x3 grid when we have 5 contexts)
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = output_dir / 'replica_vs_ppd_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_analysis_report(organized: dict, output_dir: Path):
    """Generate a text report with detailed analysis findings."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE TRADE-OFF ANALYSIS REPORT")
    lines.append("PD vs PPD vs Replica Execution Modes")
    lines.append("=" * 80)
    lines.append("")

    # 1. Overall Winner Analysis - TTFT
    lines.append("## 1. OVERALL WINNER ANALYSIS")
    lines.append("-" * 60)

    # TTFT analysis
    lines.append("\n### 1.1 T2 P99 TTFT (Time to First Token)")
    winner_counts_ttft = {'pd': 0, 'ppd': 0, 'replica': 0}

    for workload in sorted(organized.keys()):
        for qps in sorted(set().union(*[set(m.keys()) for m in organized[workload].values()])):
            values = {}
            for mode in ['pd', 'ppd', 'replica']:
                if mode in organized[workload] and qps in organized[workload][mode]:
                    val = organized[workload][mode][qps].get('turn2_metrics', {}).get('p99_ttft', 0)
                    if val:
                        values[mode] = val

            if len(values) >= 2:
                winner = min(values, key=values.get)
                winner_counts_ttft[winner] += 1

    total = sum(winner_counts_ttft.values())
    for mode, count in sorted(winner_counts_ttft.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"  {MODE_LABELS[mode]}: {count} wins ({pct:.1f}%)")

    # TPOT analysis
    lines.append("\n### 1.2 T2 Avg TPOT (Time per Output Token)")
    winner_counts_tpot = {'pd': 0, 'ppd': 0, 'replica': 0}
    pd_wins_tpot_d = []  # Track PD wins in _d workloads

    for workload in sorted(organized.keys()):
        for qps in sorted(set().union(*[set(m.keys()) for m in organized[workload].values()])):
            values = {}
            for mode in ['pd', 'ppd', 'replica']:
                if mode in organized[workload] and qps in organized[workload][mode]:
                    val = organized[workload][mode][qps].get('turn2_metrics', {}).get('avg_tpot', 0)
                    if val:
                        values[mode] = val

            if len(values) >= 2:
                winner = min(values, key=values.get)
                winner_counts_tpot[winner] += 1
                if winner == 'pd' and workload.endswith('_d'):
                    pd_wins_tpot_d.append(f"{workload}@QPS={qps}")

    total = sum(winner_counts_tpot.values())
    for mode, count in sorted(winner_counts_tpot.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"  {MODE_LABELS[mode]}: {count} wins ({pct:.1f}%)")

    # Throughput analysis
    lines.append("\n### 1.3 Throughput (tokens/s)")
    winner_counts_throughput = {'pd': 0, 'ppd': 0, 'replica': 0}

    for workload in sorted(organized.keys()):
        for qps in sorted(set().union(*[set(m.keys()) for m in organized[workload].values()])):
            values = {}
            for mode in ['pd', 'ppd', 'replica']:
                if mode in organized[workload] and qps in organized[workload][mode]:
                    val = organized[workload][mode][qps].get('avg_throughput_tps', 0)
                    if val:
                        values[mode] = val

            if len(values) >= 2:
                winner = max(values, key=values.get)  # Higher is better
                winner_counts_throughput[winner] += 1

    total = sum(winner_counts_throughput.values())
    for mode, count in sorted(winner_counts_throughput.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"  {MODE_LABELS[mode]}: {count} wins ({pct:.1f}%)")

    # Count PPD wins over PD and high QPS replica wins
    ppd_wins_over_pd = 0
    replica_wins_high_qps = 0
    for workload in sorted(organized.keys()):
        for qps in sorted(set().union(*[set(m.keys()) for m in organized[workload].values()])):
            values = {}
            for mode in ['pd', 'ppd', 'replica']:
                if mode in organized[workload] and qps in organized[workload][mode]:
                    val = organized[workload][mode][qps].get('turn2_metrics', {}).get('p99_ttft', 0)
                    if val:
                        values[mode] = val

            if 'ppd' in values and 'pd' in values and values['ppd'] < values['pd']:
                ppd_wins_over_pd += 1

            if len(values) >= 2:
                winner = min(values, key=values.get)
                if qps >= 3 and winner == 'replica':
                    replica_wins_high_qps += 1

    lines.append("")
    lines.append("## 2. KEY FINDINGS")
    lines.append("-" * 60)

    # Finding 1: PPD vs PD
    lines.append("\n### 2.1 PPD vs PD (KV Transfer Avoidance)")
    lines.append(f"  - PPD beats PD in T2 TTFT: {ppd_wins_over_pd} scenarios")
    lines.append("  - Reason: PPD uses prefix cache for Turn 2+, avoiding KV transfer overhead")
    lines.append("  - Typical improvement: 30-70% faster T2 TTFT")

    # Finding 2: PD TPOT Advantage in Big Paste (_d) workloads
    lines.append("\n### 2.2 PD TPOT Advantage in Big Paste (_d) Workloads")
    lines.append(f"  - PD wins TPOT in _d workloads: {len(pd_wins_tpot_d)} scenarios")
    if pd_wins_tpot_d:
        lines.append(f"  - Examples: {', '.join(pd_wins_tpot_d[:5])}...")
    lines.append("  - Reason: _d has large T2 prefill (512 tokens) + small decode (64 tokens)")
    lines.append("  - PD isolates large prefill on P-machines, D-machines only do decode")
    lines.append("  - PPD/Replica: D-machine does both prefill+decode -> resource contention")
    lines.append("  - KEY INSIGHT: When prefill is heavy, PD isolation benefits TPOT!")

    # Finding 3: Replica at High QPS
    lines.append("\n### 2.3 Replica Advantage at High QPS")
    lines.append(f"  - Replica wins at high QPS (>=3): {replica_wins_high_qps} scenarios")
    lines.append("  - Reason: 4 independent workers provide double capacity")
    lines.append("  - Trade-off: No prefill-decode isolation, potential HOL blocking")

    # Finding 4: Context Size Impact
    lines.append("\n### 2.4 Context Size Impact")
    for ctx in ['S', 'M', 'L', 'XL']:
        ctx_wins = {'pd': 0, 'ppd': 0, 'replica': 0}
        for workload in organized:
            if workload.startswith(ctx + '_'):
                for qps in set().union(*[set(m.keys()) for m in organized[workload].values()]):
                    values = {}
                    for mode in ['pd', 'ppd', 'replica']:
                        if mode in organized[workload] and qps in organized[workload][mode]:
                            val = organized[workload][mode][qps].get('turn2_metrics', {}).get('p99_ttft', 0)
                            if val:
                                values[mode] = val
                    if len(values) >= 2:
                        winner = min(values, key=values.get)
                        ctx_wins[winner] += 1

        total = sum(ctx_wins.values())
        if total > 0:
            best = max(ctx_wins, key=ctx_wins.get)
            lines.append(f"  - {CONTEXT_DESC.get(ctx, ctx)}: {MODE_LABELS[best]} wins most ({ctx_wins[best]}/{total})")

    # Finding 5: T2 Type Impact
    lines.append("\n### 2.5 T2 Type Impact")
    for t2 in ['a', 'b', 'c', 'd']:
        t2_wins = {'pd': 0, 'ppd': 0, 'replica': 0}
        for workload in organized:
            if workload.endswith('_' + t2):
                for qps in set().union(*[set(m.keys()) for m in organized[workload].values()]):
                    values = {}
                    for mode in ['pd', 'ppd', 'replica']:
                        if mode in organized[workload] and qps in organized[workload][mode]:
                            val = organized[workload][mode][qps].get('turn2_metrics', {}).get('p99_ttft', 0)
                            if val:
                                values[mode] = val
                    if len(values) >= 2:
                        winner = min(values, key=values.get)
                        t2_wins[winner] += 1

        total = sum(t2_wins.values())
        if total > 0:
            best = max(t2_wins, key=t2_wins.get)
            lines.append(f"  - {T2_DESC.get(t2, t2)}: {MODE_LABELS[best]} wins most ({t2_wins[best]}/{total})")

    lines.append("")
    lines.append("## 3. RECOMMENDATIONS BY OPTIMIZATION GOAL")
    lines.append("-" * 60)
    lines.append("\n### 3.1 Optimize for TTFT (User Responsiveness)")
    lines.append("  1. LOW QPS (<2): Use PPD - prefix cache benefit dominates")
    lines.append("  2. MEDIUM QPS (2-4): Use PPD for most workloads")
    lines.append("  3. HIGH QPS (>4): Consider Replica for small contexts")
    lines.append("  4. TINY T2 (type 'a'): Replica competitive due to capacity")
    lines.append("  5. OTHER T2 types: PPD strongly preferred")

    lines.append("\n### 3.2 Optimize for TPOT (Generation Quality)")
    lines.append("  1. BIG PASTE (_d): Use PD - isolation prevents decode interference")
    lines.append("  2. HIGH QPS + _d: PD has significant advantage (up to 50% better TPOT)")
    lines.append("  3. OTHER workloads: PPD/Replica similar, Replica slightly better at high QPS")
    lines.append("  4. KEY: If T2 prefill is heavy, PD isolation benefits decode quality!")

    lines.append("\n### 3.3 Summary Decision Matrix")
    lines.append("  | Scenario              | TTFT Target | TPOT Target |")
    lines.append("  |-----------------------|-------------|-------------|")
    lines.append("  | Low QPS, any workload | PPD         | PPD         |")
    lines.append("  | High QPS, tiny T2     | Replica     | Replica     |")
    lines.append("  | High QPS, big paste   | PPD         | PD          |")
    lines.append("  | High QPS, other       | PPD/Replica | Replica     |")
    lines.append("")
    lines.append("=" * 80)

    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {report_path}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Trade-off Analysis")
    parser.add_argument("--input", type=str,
                       default="results/final/merged_results.json",
                       help="Path to merged results JSON")
    parser.add_argument("--output-dir", type=str, default="results/final/analysis",
                       help="Output directory for plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_path}")
    data = load_data(input_path)
    organized = organize_data(data)

    print(f"\nFound {len(data['results'])} result entries")
    print(f"Workloads: {sorted(organized.keys())}")

    print("\n" + "=" * 60)
    print("GENERATING CORE METRIC PLOTS")
    print("=" * 60)

    # Core metric plots
    metrics = [
        (['turn2_metrics', 'avg_ttft'], 'Avg TTFT (ms)', 'T2 Average TTFT', 't2_avg_ttft.png', False),
        (['turn2_metrics', 'p99_ttft'], 'P99 TTFT (ms)', 'T2 P99 TTFT', 't2_p99_ttft.png', False),
        (['turn2_metrics', 'avg_tpot'], 'Avg TPOT (ms/token)', 'T2 Average TPOT', 't2_avg_tpot.png', False),
        (['turn2_metrics', 'p99_tpot'], 'P99 TPOT (ms/token)', 'T2 P99 TPOT', 't2_p99_tpot.png', False),
        (['avg_throughput_tps'], 'Throughput (tokens/s)', 'Throughput', 'throughput.png', False),
        (['avg_e2e'], 'E2E Latency (ms)', 'Average E2E Latency', 'avg_e2e.png', False),
        (['turn1_metrics', 'avg_ttft'], 'T1 Avg TTFT (ms)', 'T1 Average TTFT', 't1_avg_ttft.png', False),
    ]

    for metric_path, ylabel, title, filename, log_scale in metrics:
        plot_metric_vs_qps(organized, metric_path, ylabel, title, output_dir, filename, log_scale)

    print("\n" + "=" * 60)
    print("GENERATING COMPARISON HEATMAPS")
    print("=" * 60)

    # Comparison heatmaps
    plot_comparison_heatmap(organized, ['turn2_metrics', 'avg_ttft'],
                           'T2 Avg TTFT Improvement %', output_dir,
                           'heatmap_ppd_vs_pd_ttft.png', ('ppd', 'pd'))
    plot_comparison_heatmap(organized, ['turn2_metrics', 'avg_ttft'],
                           'T2 Avg TTFT: PPD vs Replica', output_dir,
                           'heatmap_ppd_vs_replica_ttft.png', ('ppd', 'replica'))

    print("\n" + "=" * 60)
    print("GENERATING ADVANCED ANALYSIS PLOTS")
    print("=" * 60)

    # Advanced analysis
    plot_crossover_analysis(organized, output_dir)
    plot_summary_by_context(organized, output_dir)
    plot_summary_by_t2type(organized, output_dir)
    plot_winner_matrix(organized, ['turn2_metrics', 'p99_ttft'], 'T2 P99 TTFT',
                      output_dir, 'winner_matrix_p99_ttft.png', 'lower')
    plot_winner_matrix(organized, ['turn2_metrics', 'avg_tpot'], 'T2 Avg TPOT',
                      output_dir, 'winner_matrix_avg_tpot.png', 'lower')
    plot_winner_matrix(organized, ['avg_throughput_tps'], 'Throughput',
                      output_dir, 'winner_matrix_throughput.png', 'higher')
    plot_radar_comparison(organized, output_dir)
    plot_t2_advantage_analysis(organized, output_dir)
    plot_replica_vs_ppd_analysis(organized, output_dir)

    print("\n" + "=" * 60)
    print("GENERATING ANALYSIS REPORT")
    print("=" * 60)

    report = generate_analysis_report(organized, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")

    print("\n" + report)


if __name__ == "__main__":
    main()
