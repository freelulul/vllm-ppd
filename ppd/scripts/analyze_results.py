#!/usr/bin/env python3
"""
Comprehensive Benchmark Analysis and Visualization (V3)

Usage: python scripts/analyze_results.py [RUN_NUM]
       RUN_NUM: 1, 2, 3, ... (analyzes results from results/run{N}/)

Generates detailed plots comparing PD, PPD, and Replication modes across:
- Turn 2 P99 TTFT vs QPS (primary)
- Throughput (TPS) vs QPS (primary)
- E2E Latency vs QPS
- Success Rate vs QPS
- Turn 1 metrics comparison
- Cross-workload summaries
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Setup - will be configured in main()
PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = None  # Set in main()
PLOT_DIR = None     # Set in main()

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'pd': '#e74c3c',       # Red
    'ppd': '#3498db',      # Blue
    'replication': '#2ecc71'  # Green
}
MARKERS = {'pd': 'o', 'ppd': 's', 'replication': '^'}
LABELS = {'pd': 'PD', 'ppd': 'PPD', 'replication': 'Replication'}

# Workload descriptions
T1_DESC = {'S': 'C1=512', 'M': 'C1=1024', 'L': 'C1=2048', 'XL': 'C1=4096'}
T2_DESC = {
    'a': '32→64 (tiny)',
    'b': '32→512 (long out)',
    'c': '256→256 (balanced)',
    'd': '1024→64 (big in)'
}


def load_data():
    """Load merged benchmark results."""
    # Find the most recent merged files
    qps_files = sorted(RESULTS_DIR.glob("qps_benchmark_v3_merged_*.json"))
    rep_files = sorted(RESULTS_DIR.glob("replication_benchmark_v3_merged_*.json"))

    if not qps_files or not rep_files:
        raise FileNotFoundError("Merged result files not found. Run merge_benchmark_results.py first.")

    qps_data = json.load(open(qps_files[-1]))
    rep_data = json.load(open(rep_files[-1]))

    print(f"Loaded QPS data: {len(qps_data['results'])} experiments")
    print(f"Loaded Replication data: {len(rep_data['results'])} experiments")

    return qps_data, rep_data


def organize_data(qps_data, rep_data):
    """Organize data by workload and mode for easy plotting."""
    # Structure: data[workload][mode][qps] = metrics
    data = defaultdict(lambda: defaultdict(dict))

    # Process QPS results (PD and PPD)
    for r in qps_data['results']:
        wk = r['workload']
        mode = r['mode']
        qps = r['target_qps']

        t2 = r.get('turn2_metrics', {}) or {}
        t1 = r.get('turn1_metrics', {}) or {}

        data[wk][mode][qps] = {
            'p99_ttft': t2.get('p99_ttft', 0),
            'avg_ttft': t2.get('avg_ttft', 0),
            'p50_ttft': t2.get('p50_ttft', 0),
            'avg_e2e': t2.get('avg_e2e', 0),
            'p99_e2e': t2.get('p99_e2e', 0),
            'avg_tpot': t2.get('avg_tpot', 0),
            'p99_tpot': t2.get('p99_tpot', 0),
            'p50_tpot': t2.get('p50_tpot', 0),
            'avg_throughput': t2.get('avg_throughput_tps', 0),
            'success_count': r.get('success_count', 0),
            'sample_count': r.get('sample_count', 0),
            'success_rate': r.get('success_count', 0) / max(r.get('sample_count', 1), 1),
            't1_avg_ttft': t1.get('avg_ttft', 0),
            't1_p99_ttft': t1.get('p99_ttft', 0),
            't1_avg_e2e': t1.get('avg_e2e', 0),
            'total_avg_e2e': r.get('avg_total_e2e', 0),
            'total_p99_e2e': r.get('p99_total_e2e', 0),
        }

    # Process Replication results
    for r in rep_data['results']:
        wk = r['workload']
        qps = r['target_qps']

        t2 = r.get('turn2_metrics', {}) or {}
        t1 = r.get('turn1_metrics', {}) or {}

        data[wk]['replication'][qps] = {
            'p99_ttft': t2.get('p99_ttft', 0),
            'avg_ttft': t2.get('avg_ttft', 0),
            'p50_ttft': t2.get('p50_ttft', 0),
            'avg_e2e': t2.get('avg_e2e', 0),
            'p99_e2e': t2.get('p99_e2e', 0),
            'avg_tpot': t2.get('avg_tpot', 0),
            'p99_tpot': t2.get('p99_tpot', 0),
            'p50_tpot': t2.get('p50_tpot', 0),
            'avg_throughput': t2.get('avg_throughput_tps', 0),
            'success_count': r.get('success_count', 0),
            'sample_count': r.get('sample_count', 0),
            'success_rate': r.get('success_count', 0) / max(r.get('sample_count', 1), 1),
            't1_avg_ttft': t1.get('avg_ttft', 0),
            't1_p99_ttft': t1.get('p99_ttft', 0),
            't1_avg_e2e': t1.get('avg_e2e', 0),
            'total_avg_e2e': r.get('avg_total_e2e', 0),
            'total_p99_e2e': r.get('p99_total_e2e', 0),
        }

    return data


def get_workload_title(workload):
    """Generate descriptive title for workload."""
    parts = workload.split('_')
    if len(parts) == 2:
        t1, t2 = parts
        return f"{workload}: {T1_DESC.get(t1, t1)}, T2={T2_DESC.get(t2, t2)}"
    return workload


def plot_metric_vs_qps(data, workload, metric_key, ylabel, title_suffix, filename_suffix,
                       log_y=False, ylim=None, subdir=None):
    """Generic function to plot a metric vs QPS for all three modes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for mode in ['pd', 'ppd', 'replication']:
        if mode not in data[workload]:
            continue

        mode_data = data[workload][mode]
        qps_vals = sorted(mode_data.keys())
        metric_vals = [mode_data[q].get(metric_key, 0) for q in qps_vals]

        ax.plot(qps_vals, metric_vals,
                color=COLORS[mode], marker=MARKERS[mode],
                label=LABELS[mode], linewidth=2, markersize=8)

    ax.set_xlabel('QPS (Requests/Second)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{get_workload_title(workload)}\n{title_suffix}', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    if log_y:
        ax.set_yscale('log')
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    output_dir = PLOT_DIR / subdir if subdir else PLOT_DIR
    plt.savefig(output_dir / f'{workload}_{filename_suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_workloads_p99_ttft(data):
    """Plot Turn 2 P99 TTFT vs QPS for all workloads (PRIMARY PLOT 1)."""
    print("\nGenerating P99 TTFT plots...")
    for wk in sorted(data.keys()):
        plot_metric_vs_qps(
            data, wk,
            metric_key='p99_ttft',
            ylabel='Turn 2 P99 TTFT (ms)',
            title_suffix='Turn 2 P99 TTFT vs QPS',
            filename_suffix='p99_ttft',
            subdir='01_p99_ttft'
        )
        print(f"  - 01_p99_ttft/{wk}_p99_ttft.png")


def plot_all_workloads_throughput(data):
    """Plot Throughput (TPS) vs QPS for all workloads (PRIMARY PLOT 2)."""
    print("\nGenerating Throughput plots...")
    for wk in sorted(data.keys()):
        plot_metric_vs_qps(
            data, wk,
            metric_key='avg_throughput',
            ylabel='Throughput (tokens/sec)',
            title_suffix='Turn 2 Throughput vs QPS',
            filename_suffix='throughput',
            subdir='03_throughput'
        )
        print(f"  - 03_throughput/{wk}_throughput.png")


def plot_all_workloads_avg_tpot(data):
    """Plot Average TPOT vs QPS for all workloads (TPOT PLOT 1)."""
    print("\nGenerating Average TPOT plots...")
    for wk in sorted(data.keys()):
        plot_metric_vs_qps(
            data, wk,
            metric_key='avg_tpot',
            ylabel='Turn 2 Avg TPOT (ms/token)',
            title_suffix='Turn 2 Average Time Per Output Token vs QPS',
            filename_suffix='avg_tpot',
            subdir='02_tpot/avg_tpot'
        )
        print(f"  - 02_tpot/avg_tpot/{wk}_avg_tpot.png")


def plot_all_workloads_p99_tpot(data):
    """Plot P99 TPOT vs QPS for all workloads (TPOT PLOT 2)."""
    print("\nGenerating P99 TPOT plots...")
    for wk in sorted(data.keys()):
        plot_metric_vs_qps(
            data, wk,
            metric_key='p99_tpot',
            ylabel='Turn 2 P99 TPOT (ms/token)',
            title_suffix='Turn 2 P99 Time Per Output Token vs QPS',
            filename_suffix='p99_tpot',
            subdir='02_tpot/p99_tpot'
        )
        print(f"  - 02_tpot/p99_tpot/{wk}_p99_tpot.png")


def plot_tpot_summary_by_t2(data):
    """Create TPOT summary plots grouped by T2 (request type) - shows PD TPOT advantage."""
    print("\nGenerating T2 TPOT summary plots (PD advantage focus)...")

    t2_groups = {'a': [], 'b': [], 'c': [], 'd': []}
    for wk in data.keys():
        parts = wk.split('_')
        if len(parts) == 2:
            t2 = parts[1]
            if t2 in t2_groups:
                t2_groups[t2].append(wk)

    for t2, workloads in t2_groups.items():
        if not workloads:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, wk in enumerate(sorted(workloads)):
            ax = axes[idx]
            t1 = wk.split('_')[0]

            for mode in ['pd', 'ppd', 'replication']:
                if mode not in data[wk]:
                    continue
                mode_data = data[wk][mode]
                qps_vals = sorted(mode_data.keys())
                p99_vals = [mode_data[q].get('p99_tpot', 0) for q in qps_vals]

                ax.plot(qps_vals, p99_vals, color=COLORS[mode],
                       marker=MARKERS[mode], label=LABELS[mode], linewidth=2)

            ax.set_xlabel('QPS')
            ax.set_ylabel('P99 TPOT (ms/token)')
            ax.set_title(f'{T1_DESC.get(t1, t1)}')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Request Type: T2={T2_DESC[t2]} - P99 TPOT Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        subdir = PLOT_DIR / '07_summary_by_request'
        plt.savefig(subdir / f'summary_T2_{t2}_p99_tpot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - 07_summary_by_request/summary_T2_{t2}_p99_tpot.png")


def plot_pd_tpot_advantage(data):
    """Create focused analysis of PD TPOT advantage scenarios."""
    print("\nGenerating PD TPOT advantage analysis...")

    # Focus on _d workloads where PD should shine
    d_workloads = [wk for wk in sorted(data.keys()) if wk.endswith('_d')]

    if not d_workloads:
        print("  No _d workloads found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, wk in enumerate(d_workloads[:4]):
        ax = axes[idx]
        t1 = wk.split('_')[0]

        # Plot P99 TPOT for all modes
        for mode in ['pd', 'ppd', 'replication']:
            if mode not in data[wk]:
                continue
            mode_data = data[wk][mode]
            qps_vals = sorted(mode_data.keys())
            tpot_vals = [mode_data[q].get('p99_tpot', 0) for q in qps_vals]

            ax.plot(qps_vals, tpot_vals, color=COLORS[mode],
                   marker=MARKERS[mode], label=LABELS[mode], linewidth=2, markersize=8)

        ax.set_xlabel('QPS', fontsize=11)
        ax.set_ylabel('P99 TPOT (ms/token)', fontsize=11)
        ax.set_title(f'{wk}: {T1_DESC.get(t1, t1)}, T2=1024→64 (prefill-heavy)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('PD TPOT Advantage: Prefill-Heavy Workloads (_d)\nPD decode isolation → stable TPOT under load',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    subdir = PLOT_DIR / '09_pd_advantage'
    plt.savefig(subdir / 'pd_tpot_advantage_d_workloads.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - 09_pd_advantage/pd_tpot_advantage_d_workloads.png")

    # Also create a heatmap showing PD TPOT advantage at high QPS
    plot_pd_tpot_advantage_heatmap(data)


def plot_pd_tpot_advantage_heatmap(data):
    """Create heatmap showing PD P99 TPOT advantage over PPD at high QPS."""
    t1_types = ['S', 'M', 'L', 'XL']
    t2_types = ['a', 'b', 'c', 'd']

    # Calculate PD advantage at high QPS (use QPS=8 or max available)
    target_qps = 8.0
    improvements = np.zeros((len(t1_types), len(t2_types)))

    for i, t1 in enumerate(t1_types):
        for j, t2 in enumerate(t2_types):
            wk = f"{t1}_{t2}"
            if wk in data and 'pd' in data[wk] and 'ppd' in data[wk]:
                pd_data = data[wk]['pd']
                ppd_data = data[wk]['ppd']

                # Find highest available QPS
                common_qps = set(pd_data.keys()) & set(ppd_data.keys())
                if not common_qps:
                    continue

                qps = min(common_qps, key=lambda x: abs(x - target_qps))

                pd_tpot = pd_data[qps].get('p99_tpot', 0)
                ppd_tpot = ppd_data[qps].get('p99_tpot', 0)

                if ppd_tpot > 0:
                    # Positive = PD better (lower TPOT)
                    improvements[i, j] = (ppd_tpot - pd_tpot) / ppd_tpot * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)

    ax.set_xticks(range(len(t2_types)))
    ax.set_xticklabels([T2_DESC[t] for t in t2_types], fontsize=10)
    ax.set_yticks(range(len(t1_types)))
    ax.set_yticklabels([T1_DESC[t] for t in t1_types], fontsize=10)

    # Add text annotations
    for i in range(len(t1_types)):
        for j in range(len(t2_types)):
            color = 'white' if abs(improvements[i, j]) > 30 else 'black'
            text = ax.text(j, i, f'{improvements[i, j]:+.1f}%',
                          ha='center', va='center', fontsize=11, fontweight='bold', color=color)

    ax.set_xlabel('T2 (Request Type)', fontsize=12)
    ax.set_ylabel('T1 (Context Size)', fontsize=12)
    ax.set_title(f'PD P99 TPOT Advantage over PPD (%) at QPS≈{target_qps}\nGreen=PD better, Red=PPD better',
                 fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PD Improvement (%)', fontsize=11)

    plt.tight_layout()
    subdir = PLOT_DIR / '09_pd_advantage'
    plt.savefig(subdir / 'heatmap_pd_tpot_advantage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - 09_pd_advantage/heatmap_pd_tpot_advantage.png")


def plot_all_workloads_e2e(data):
    """Plot E2E Latency vs QPS for all workloads."""
    print("\nGenerating E2E Latency plots...")
    for wk in sorted(data.keys()):
        plot_metric_vs_qps(
            data, wk,
            metric_key='avg_e2e',
            ylabel='Turn 2 Avg E2E Latency (ms)',
            title_suffix='Turn 2 E2E Latency vs QPS',
            filename_suffix='e2e_latency',
            subdir='04_e2e_latency/avg'
        )
        print(f"  - 04_e2e_latency/avg/{wk}_e2e_latency.png")


def plot_all_workloads_total_e2e(data):
    """Plot Total E2E (Turn1+Turn2) vs QPS for all workloads."""
    print("\nGenerating Total E2E plots...")
    for wk in sorted(data.keys()):
        plot_metric_vs_qps(
            data, wk,
            metric_key='total_avg_e2e',
            ylabel='Total E2E (Turn1+Turn2) (ms)',
            title_suffix='Total E2E Latency vs QPS',
            filename_suffix='total_e2e',
            subdir='04_e2e_latency/total'
        )
        print(f"  - 04_e2e_latency/total/{wk}_total_e2e.png")


def plot_all_workloads_success_rate(data):
    """Plot Success Rate vs QPS for all workloads."""
    print("\nGenerating Success Rate plots...")
    for wk in sorted(data.keys()):
        plot_metric_vs_qps(
            data, wk,
            metric_key='success_rate',
            ylabel='Success Rate',
            title_suffix='Success Rate vs QPS',
            filename_suffix='success_rate',
            ylim=(0, 1.05),
            subdir='05_success_rate'
        )
        print(f"  - 05_success_rate/{wk}_success_rate.png")


def plot_summary_by_t1(data):
    """Create summary plots grouped by T1 (context size)."""
    print("\nGenerating T1 summary plots...")

    t1_groups = {'S': [], 'M': [], 'L': [], 'XL': []}
    for wk in data.keys():
        t1 = wk.split('_')[0]
        if t1 in t1_groups:
            t1_groups[t1].append(wk)

    for t1, workloads in t1_groups.items():
        if not workloads:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, wk in enumerate(sorted(workloads)):
            ax = axes[idx]
            t2 = wk.split('_')[1]

            for mode in ['pd', 'ppd', 'replication']:
                if mode not in data[wk]:
                    continue
                mode_data = data[wk][mode]
                qps_vals = sorted(mode_data.keys())
                p99_vals = [mode_data[q].get('p99_ttft', 0) for q in qps_vals]

                ax.plot(qps_vals, p99_vals, color=COLORS[mode],
                       marker=MARKERS[mode], label=LABELS[mode], linewidth=2)

            ax.set_xlabel('QPS')
            ax.set_ylabel('P99 TTFT (ms)')
            ax.set_title(f'T2={T2_DESC.get(t2, t2)}')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Context Size: {T1_DESC[t1]} - P99 TTFT Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        subdir = PLOT_DIR / '06_summary_by_context'
        plt.savefig(subdir / f'summary_T1_{t1}_p99_ttft.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - 06_summary_by_context/summary_T1_{t1}_p99_ttft.png")


def plot_summary_by_t2(data):
    """Create summary plots grouped by T2 (request type)."""
    print("\nGenerating T2 summary plots...")

    t2_groups = {'a': [], 'b': [], 'c': [], 'd': []}
    for wk in data.keys():
        parts = wk.split('_')
        if len(parts) == 2:
            t2 = parts[1]
            if t2 in t2_groups:
                t2_groups[t2].append(wk)

    for t2, workloads in t2_groups.items():
        if not workloads:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, wk in enumerate(sorted(workloads)):
            ax = axes[idx]
            t1 = wk.split('_')[0]

            for mode in ['pd', 'ppd', 'replication']:
                if mode not in data[wk]:
                    continue
                mode_data = data[wk][mode]
                qps_vals = sorted(mode_data.keys())
                p99_vals = [mode_data[q].get('p99_ttft', 0) for q in qps_vals]

                ax.plot(qps_vals, p99_vals, color=COLORS[mode],
                       marker=MARKERS[mode], label=LABELS[mode], linewidth=2)

            ax.set_xlabel('QPS')
            ax.set_ylabel('P99 TTFT (ms)')
            ax.set_title(f'{T1_DESC.get(t1, t1)}')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Request Type: T2={T2_DESC[t2]} - P99 TTFT Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        subdir = PLOT_DIR / '07_summary_by_request'
        plt.savefig(subdir / f'summary_T2_{t2}_p99_ttft.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - 07_summary_by_request/summary_T2_{t2}_p99_ttft.png")


def plot_ppd_improvement_heatmap(data):
    """Create heatmap showing PPD improvement over PD at different QPS levels."""
    print("\nGenerating PPD improvement heatmap...")

    # Organize by T1 and T2
    t1_types = ['S', 'M', 'L', 'XL']
    t2_types = ['a', 'b', 'c', 'd']

    # Calculate average improvement at medium QPS (2.0)
    target_qps = 2.0
    improvements = np.zeros((len(t1_types), len(t2_types)))

    for i, t1 in enumerate(t1_types):
        for j, t2 in enumerate(t2_types):
            wk = f"{t1}_{t2}"
            if wk in data and 'pd' in data[wk] and 'ppd' in data[wk]:
                pd_data = data[wk]['pd']
                ppd_data = data[wk]['ppd']

                # Find closest QPS
                pd_qps = min(pd_data.keys(), key=lambda x: abs(x - target_qps))
                ppd_qps = min(ppd_data.keys(), key=lambda x: abs(x - target_qps))

                pd_ttft = pd_data[pd_qps].get('p99_ttft', 0)
                ppd_ttft = ppd_data[ppd_qps].get('p99_ttft', 0)

                if pd_ttft > 0:
                    improvements[i, j] = (pd_ttft - ppd_ttft) / pd_ttft * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=80)

    ax.set_xticks(range(len(t2_types)))
    ax.set_xticklabels([T2_DESC[t] for t in t2_types], fontsize=10)
    ax.set_yticks(range(len(t1_types)))
    ax.set_yticklabels([T1_DESC[t] for t in t1_types], fontsize=10)

    # Add text annotations
    for i in range(len(t1_types)):
        for j in range(len(t2_types)):
            text = ax.text(j, i, f'{improvements[i, j]:.1f}%',
                          ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('T2 (Request Type)', fontsize=12)
    ax.set_ylabel('T1 (Context Size)', fontsize=12)
    ax.set_title(f'PPD P99 TTFT Improvement over PD (%) at QPS={target_qps}', fontsize=14)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', fontsize=11)

    plt.tight_layout()
    subdir = PLOT_DIR / '08_comparative_heatmaps'
    plt.savefig(subdir / 'heatmap_ppd_improvement.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - 08_comparative_heatmaps/heatmap_ppd_improvement.png")


def plot_crossover_analysis(data):
    """Analyze and plot QPS crossover points where PPD beats Replication."""
    print("\nGenerating crossover analysis...")

    crossover_data = []

    for wk in sorted(data.keys()):
        if 'ppd' not in data[wk] or 'replication' not in data[wk]:
            continue

        ppd_data = data[wk]['ppd']
        rep_data = data[wk]['replication']

        common_qps = sorted(set(ppd_data.keys()) & set(rep_data.keys()))

        crossover_qps = None
        for qps in common_qps:
            ppd_ttft = ppd_data[qps].get('p99_ttft', float('inf'))
            rep_ttft = rep_data[qps].get('p99_ttft', float('inf'))

            if ppd_ttft < rep_ttft:
                crossover_qps = qps
                break

        t1, t2 = wk.split('_')
        crossover_data.append({
            'workload': wk,
            't1': t1,
            't2': t2,
            'crossover_qps': crossover_qps if crossover_qps else '>max'
        })

    # Create crossover summary plot
    fig, ax = plt.subplots(figsize=(12, 6))

    workloads = [d['workload'] for d in crossover_data]
    crossovers = [d['crossover_qps'] if isinstance(d['crossover_qps'], (int, float)) else -1
                  for d in crossover_data]

    colors = ['#2ecc71' if c > 0 else '#95a5a6' for c in crossovers]
    bars = ax.bar(workloads, [c if c > 0 else 0 for c in crossovers], color=colors)

    ax.set_xlabel('Workload', fontsize=12)
    ax.set_ylabel('Crossover QPS (PPD < Replication)', fontsize=12)
    ax.set_title('QPS Threshold Where PPD Becomes Better Than Replication', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    # Add annotations
    for i, (bar, c) in enumerate(zip(bars, crossovers)):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{c}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                   'N/A', ha='center', va='bottom', fontsize=9, color='gray')

    plt.tight_layout()
    subdir = PLOT_DIR / '08_comparative_heatmaps'
    plt.savefig(subdir / 'crossover_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - 08_comparative_heatmaps/crossover_analysis.png")

    return crossover_data


def plot_turn1_comparison(data):
    """Compare Turn 1 metrics across modes."""
    print("\nGenerating Turn 1 comparison...")

    workloads = sorted(data.keys())

    # Collect Turn 1 data at low QPS (0.1)
    t1_data = {'pd': [], 'ppd': [], 'replication': []}

    for wk in workloads:
        for mode in ['pd', 'ppd', 'replication']:
            if mode in data[wk]:
                mode_data = data[wk][mode]
                # Get low QPS data
                low_qps = min(mode_data.keys())
                t1_ttft = mode_data[low_qps].get('t1_avg_ttft', 0)
                t1_data[mode].append(t1_ttft)
            else:
                t1_data[mode].append(0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(workloads))
    width = 0.25

    ax.bar(x - width, t1_data['pd'], width, label='PD', color=COLORS['pd'])
    ax.bar(x, t1_data['ppd'], width, label='PPD', color=COLORS['ppd'])
    ax.bar(x + width, t1_data['replication'], width, label='Replication', color=COLORS['replication'])

    ax.set_xlabel('Workload', fontsize=12)
    ax.set_ylabel('Turn 1 Avg TTFT (ms)', fontsize=12)
    ax.set_title('Turn 1 TTFT Comparison Across Modes', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    subdir = PLOT_DIR / '12_turn1'
    plt.savefig(subdir / 'turn1_ttft_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - 12_turn1/turn1_ttft_comparison.png")


def plot_high_qps_degradation(data):
    """Analyze performance degradation at high QPS."""
    print("\nGenerating high QPS degradation analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Group by T1 for clarity
    t1_groups = {'S': 'upper left', 'M': 'upper right', 'L': 'lower left', 'XL': 'lower right'}
    ax_map = {'S': axes[0,0], 'M': axes[0,1], 'L': axes[1,0], 'XL': axes[1,1]}

    for wk in sorted(data.keys()):
        t1, t2 = wk.split('_')
        ax = ax_map[t1]

        for mode in ['pd', 'ppd', 'replication']:
            if mode not in data[wk]:
                continue

            mode_data = data[wk][mode]
            qps_vals = sorted(mode_data.keys())

            # Normalize TTFT relative to lowest QPS
            base_ttft = mode_data[qps_vals[0]].get('p99_ttft', 1)
            if base_ttft == 0:
                base_ttft = 1

            norm_ttft = [mode_data[q].get('p99_ttft', 0) / base_ttft for q in qps_vals]

            linestyle = '-' if mode == 'ppd' else ('--' if mode == 'pd' else ':')
            ax.plot(qps_vals, norm_ttft, color=COLORS[mode],
                   linestyle=linestyle, alpha=0.7, linewidth=1.5,
                   label=f'{t2}-{LABELS[mode]}' if t2 == 'a' else '')

    for t1, ax in ax_map.items():
        ax.set_xlabel('QPS')
        ax.set_ylabel('Normalized P99 TTFT')
        ax.set_title(f'{T1_DESC[t1]}')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        if t1 == 'S':
            ax.legend(loc='best', fontsize=8)

    fig.suptitle('Performance Degradation at High QPS (Normalized to Low QPS Baseline)', fontsize=14)
    plt.tight_layout()
    subdir = PLOT_DIR / '10_degradation'
    plt.savefig(subdir / 'high_qps_degradation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  - 10_degradation/high_qps_degradation.png")


def plot_mode_comparison_radar(data):
    """Create radar/spider charts comparing modes across metrics."""
    print("\nGenerating radar comparison charts...")

    # Select representative workloads
    representative = ['S_a', 'M_b', 'L_c', 'XL_d']

    for wk in representative:
        if wk not in data:
            continue

        # Get data at medium QPS
        target_qps = 2.0

        metrics = ['P99 TTFT', 'Avg E2E', 'Throughput', 'Success Rate']

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for mode in ['pd', 'ppd', 'replication']:
            if mode not in data[wk]:
                continue

            mode_data = data[wk][mode]
            qps = min(mode_data.keys(), key=lambda x: abs(x - target_qps))

            values = [
                mode_data[qps].get('p99_ttft', 0),
                mode_data[qps].get('avg_e2e', 0),
                mode_data[qps].get('avg_throughput', 0),
                mode_data[qps].get('success_rate', 0) * 100
            ]
            values += values[:1]

            ax.plot(angles, values, color=COLORS[mode], linewidth=2, label=LABELS[mode])
            ax.fill(angles, values, color=COLORS[mode], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(f'{get_workload_title(wk)}\nMetrics Comparison at QPS={target_qps}', fontsize=12)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        subdir = PLOT_DIR / '11_radar'
        plt.savefig(subdir / f'radar_{wk}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - 11_radar/radar_{wk}.png")


def generate_summary_report(data, crossover_data):
    """Generate a text summary report of key findings."""
    print("\nGenerating summary report...")

    report = []
    report.append("=" * 80)
    report.append("BENCHMARK ANALYSIS SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)

    # Overall statistics
    report.append("\n## OVERALL STATISTICS\n")

    total_experiments = sum(len(data[wk].get(m, {}))
                           for wk in data for m in ['pd', 'ppd', 'replication'])
    report.append(f"Total experiments: {total_experiments}")
    report.append(f"Workloads: {len(data)}")

    # PPD vs PD improvement
    report.append("\n## PPD vs PD IMPROVEMENT (Turn 2 P99 TTFT)\n")

    improvements = []
    for wk in sorted(data.keys()):
        if 'pd' in data[wk] and 'ppd' in data[wk]:
            # At QPS 2.0
            pd_data = data[wk]['pd']
            ppd_data = data[wk]['ppd']

            pd_qps = min(pd_data.keys(), key=lambda x: abs(x - 2.0))
            ppd_qps = min(ppd_data.keys(), key=lambda x: abs(x - 2.0))

            pd_ttft = pd_data[pd_qps].get('p99_ttft', 0)
            ppd_ttft = ppd_data[ppd_qps].get('p99_ttft', 0)

            if pd_ttft > 0:
                imp = (pd_ttft - ppd_ttft) / pd_ttft * 100
                improvements.append((wk, imp, pd_ttft, ppd_ttft))

    for wk, imp, pd, ppd in sorted(improvements, key=lambda x: -x[1]):
        report.append(f"  {wk}: {imp:+.1f}% (PD={pd:.1f}ms → PPD={ppd:.1f}ms)")

    if improvements:
        avg_imp = np.mean([x[1] for x in improvements])
        report.append(f"\n  Average improvement: {avg_imp:.1f}%")

    # Crossover analysis
    report.append("\n## PPD vs REPLICATION CROSSOVER POINTS\n")

    for d in crossover_data:
        report.append(f"  {d['workload']}: {d['crossover_qps']}")

    # Best mode recommendations
    report.append("\n## RECOMMENDATIONS BY SCENARIO\n")
    report.append("  - Low QPS (<1.0): Replication (lowest overhead)")
    report.append("  - Medium QPS (1.0-4.0): PPD (good balance)")
    report.append("  - High QPS (>4.0): PPD (best TTFT under load)")
    report.append("  - Large context (L/XL): PPD strongly preferred")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_text = "\n".join(report)

    with open(PLOT_DIR / 'analysis_summary.txt', 'w') as f:
        f.write(report_text)

    print("  - analysis_summary.txt")
    print("\n" + report_text)


def main():
    global RESULTS_DIR, PLOT_DIR

    # Get run identifier from command line (number or "final")
    run_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if run_arg == "final":
        RESULTS_DIR = PROJECT_DIR / "results" / "final"
        PLOT_DIR = RESULTS_DIR / "analysis_plots"
        run_label = "FINAL (averaged)"
    elif run_arg:
        run_num = int(run_arg)
        RESULTS_DIR = PROJECT_DIR / "results" / f"run{run_num}"
        PLOT_DIR = RESULTS_DIR / "analysis_plots"
        run_label = f"Run {run_num}"
    else:
        RESULTS_DIR = PROJECT_DIR / "results"
        PLOT_DIR = RESULTS_DIR / "analysis_plots"
        run_label = None

    PLOT_DIR.mkdir(exist_ok=True)

    # Create subdirectory structure
    subdirs = [
        '01_p99_ttft',
        '02_tpot/avg_tpot',
        '02_tpot/p99_tpot',
        '03_throughput',
        '04_e2e_latency/avg',
        '04_e2e_latency/total',
        '05_success_rate',
        '06_summary_by_context',
        '07_summary_by_request',
        '08_comparative_heatmaps',
        '09_pd_advantage',
        '10_degradation',
        '11_radar',
        '12_turn1',
    ]
    for subdir in subdirs:
        (PLOT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Benchmark Analysis and Visualization V3")
    if run_label:
        print(f"Analyzing: {run_label}")
    print(f"Input:  {RESULTS_DIR}")
    print(f"Output: {PLOT_DIR}")
    print("=" * 60)

    # Load data
    qps_data, rep_data = load_data()
    data = organize_data(qps_data, rep_data)

    print(f"\nOrganized data for {len(data)} workloads")

    # Generate all plots
    plot_all_workloads_p99_ttft(data)      # PRIMARY 1
    plot_all_workloads_throughput(data)     # PRIMARY 2
    plot_all_workloads_avg_tpot(data)       # TPOT 1
    plot_all_workloads_p99_tpot(data)       # TPOT 2
    plot_all_workloads_e2e(data)
    plot_all_workloads_total_e2e(data)
    plot_all_workloads_success_rate(data)

    plot_summary_by_t1(data)
    plot_summary_by_t2(data)
    plot_tpot_summary_by_t2(data)           # TPOT summary by T2

    plot_ppd_improvement_heatmap(data)
    plot_pd_tpot_advantage(data)            # PD TPOT advantage focus
    crossover_data = plot_crossover_analysis(data)

    plot_turn1_comparison(data)
    plot_high_qps_degradation(data)
    plot_mode_comparison_radar(data)

    generate_summary_report(data, crossover_data)

    print("\n" + "=" * 60)
    print(f"Analysis complete! {len(list(PLOT_DIR.glob('*.png')))} plots generated.")
    print(f"Output: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
