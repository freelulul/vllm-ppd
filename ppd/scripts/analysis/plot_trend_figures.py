#!/usr/bin/env python3
"""
Plot Trend Figures for Paper

Generates publication-quality figures showing:
1. QPS Scaling Trend
2. E2E Ratio Trend
3. SLO Strictness Trend
4. Input Length Trend
5. Optimizer Advantage Heatmap
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# Style configuration for publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'pd': '#e74c3c',       # Red
    'ppd': '#f39c12',      # Orange
    'replica': '#3498db',  # Blue
    'optimizer': '#27ae60', # Green
}

MARKERS = {
    'pd': 's',      # Square
    'ppd': '^',     # Triangle
    'replica': 'o', # Circle
    'optimizer': 'D', # Diamond
}

LABELS = {
    'pd': 'PD',
    'ppd': 'PPD',
    'replica': 'Replica',
    'optimizer': 'Optimizer',
}


def load_trend_data(data_path: str):
    """Load trend data from JSON file"""
    with open(data_path, 'r') as f:
        return json.load(f)


def plot_qps_trend(data: dict, output_dir: str):
    """
    Plot 1: SLO Attainment vs QPS/Concurrency

    Shows how each mode performs as load increases.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    qps = data["qps"]
    modes = ['pd', 'ppd', 'replica', 'optimizer']

    for mode in modes:
        ax.plot(qps, data[mode], marker=MARKERS[mode], color=COLORS[mode],
                label=LABELS[mode], linewidth=2, markersize=8)

    ax.set_xlabel('QPS (Concurrent Requests)', fontweight='bold')
    ax.set_ylabel('SLO Attainment (%)', fontweight='bold')
    ax.set_title('SLO Attainment vs Request Rate', fontweight='bold', fontsize=16)

    ax.set_ylim(0, 105)
    ax.set_xlim(min(qps) - 1, max(qps) + 1)
    ax.legend(loc='lower left', framealpha=0.9)

    # Add annotation for optimizer advantage region
    ax.axvspan(6, max(qps) + 1, alpha=0.1, color='green',
               label='Optimizer Advantage Zone')

    # Highlight the crossover point
    ax.axvline(x=6, color='gray', linestyle='--', alpha=0.5)
    ax.text(6.5, 95, 'Optimizer leads\nat higher QPS',
            fontsize=10, style='italic', color='darkgreen')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_qps_trend.png'))
    plt.savefig(os.path.join(output_dir, 'fig1_qps_trend.pdf'))
    print(f"Saved: fig1_qps_trend.png/pdf")
    plt.close()


def plot_e2e_ratio_trend(data: dict, output_dir: str):
    """
    Plot 2: SLO Attainment vs E2E Task Ratio

    Shows how workload composition affects performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data["e2e_ratio"]
    modes = ['pd', 'ppd', 'replica', 'optimizer']

    for mode in modes:
        ax.plot(x, data[mode], marker=MARKERS[mode], color=COLORS[mode],
                label=LABELS[mode], linewidth=2, markersize=8)

    ax.set_xlabel('E2E Task Ratio (%)', fontweight='bold')
    ax.set_ylabel('SLO Attainment (%)', fontweight='bold')
    ax.set_title('SLO Attainment vs Workload Composition', fontweight='bold', fontsize=16)

    ax.set_ylim(0, 105)
    ax.set_xlim(-5, 105)
    ax.legend(loc='lower left', framealpha=0.9)

    # Add annotation
    ax.annotate('Optimizer maintains high\nattainment across all mixes',
                xy=(50, data['optimizer'][5]), xytext=(60, 75),
                fontsize=10, style='italic',
                arrowprops=dict(arrowstyle='->', color='darkgreen'),
                color='darkgreen')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_e2e_ratio_trend.png'))
    plt.savefig(os.path.join(output_dir, 'fig2_e2e_ratio_trend.pdf'))
    print(f"Saved: fig2_e2e_ratio_trend.png/pdf")
    plt.close()


def plot_slo_strictness_trend(data: dict, output_dir: str):
    """
    Plot 3: SLO Attainment vs SLO Strictness

    Shows how performance degrades with stricter SLOs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data["strictness"]
    modes = ['pd', 'ppd', 'replica', 'optimizer']

    for mode in modes:
        ax.plot(x, data[mode], marker=MARKERS[mode], color=COLORS[mode],
                label=LABELS[mode], linewidth=2, markersize=8)

    ax.set_xlabel('SLO Strictness Factor (lower = stricter)', fontweight='bold')
    ax.set_ylabel('SLO Attainment (%)', fontweight='bold')
    ax.set_title('SLO Attainment vs SLO Strictness', fontweight='bold', fontsize=16)

    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', framealpha=0.9)

    # Add strictness labels
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}x' for s in x], rotation=45)

    # Highlight strict region
    ax.axvspan(0.4, 0.8, alpha=0.1, color='red')
    ax.text(0.55, 15, 'Strict\nSLO', fontsize=9, ha='center', color='darkred')

    ax.axvspan(1.4, 2.1, alpha=0.1, color='blue')
    ax.text(1.7, 15, 'Relaxed\nSLO', fontsize=9, ha='center', color='darkblue')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_slo_strictness_trend.png'))
    plt.savefig(os.path.join(output_dir, 'fig3_slo_strictness_trend.pdf'))
    print(f"Saved: fig3_slo_strictness_trend.png/pdf")
    plt.close()


def plot_input_length_trend(data: dict, output_dir: str):
    """
    Plot 4: SLO Attainment vs Input Length

    Shows how input complexity affects performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data["input_length"]
    modes = ['pd', 'ppd', 'replica', 'optimizer']

    for mode in modes:
        ax.plot(x, data[mode], marker=MARKERS[mode], color=COLORS[mode],
                label=LABELS[mode], linewidth=2, markersize=8)

    ax.set_xlabel('Average Input Length (tokens)', fontweight='bold')
    ax.set_ylabel('SLO Attainment (%)', fontweight='bold')
    ax.set_title('SLO Attainment vs Input Complexity', fontweight='bold', fontsize=16)

    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', framealpha=0.9)

    # Format x-axis with K notation for thousands
    ax.set_xticklabels([f'{int(l/1000)}K' if l >= 1000 else str(l) for l in x])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_input_length_trend.png'))
    plt.savefig(os.path.join(output_dir, 'fig4_input_length_trend.pdf'))
    print(f"Saved: fig4_input_length_trend.png/pdf")
    plt.close()


def plot_optimizer_advantage(all_data: dict, output_dir: str):
    """
    Plot 5: Optimizer Advantage Summary

    Bar chart showing optimizer advantage across different scenarios.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = []
    opt_values = []
    best_fixed_values = []
    advantages = []

    # Extract key points from each trend
    trend_configs = [
        ("qps_trend", "QPS=16", 7),
        ("qps_trend", "QPS=24", 8),
        ("e2e_ratio_trend", "E2E=50%", 5),
        ("e2e_ratio_trend", "E2E=80%", 8),
        ("slo_strictness_trend", "Strict (0.6x)", 1),
        ("slo_strictness_trend", "Normal (1.0x)", 5),
        ("input_length_trend", "Input=2K", 5),
        ("input_length_trend", "Input=4K", 7),
    ]

    for trend_name, label, idx in trend_configs:
        data = all_data[trend_name]
        opt = data["optimizer"][idx]
        replica = data["replica"][idx]
        ppd = data["ppd"][idx]
        pd = data["pd"][idx]
        best_fixed = max(replica, ppd, pd)

        scenarios.append(label)
        opt_values.append(opt)
        best_fixed_values.append(best_fixed)
        advantages.append(opt - best_fixed)

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax.bar(x - width/2, best_fixed_values, width, label='Best Fixed Mode',
                   color='#95a5a6', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, opt_values, width, label='Optimizer',
                   color=COLORS['optimizer'], edgecolor='black', linewidth=0.5)

    # Add advantage labels on top
    for i, (opt, best, adv) in enumerate(zip(opt_values, best_fixed_values, advantages)):
        if adv > 0:
            ax.text(i + width/2, opt + 2, f'+{adv:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   color='darkgreen')

    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_ylabel('SLO Attainment (%)', fontweight='bold')
    ax.set_title('Optimizer Advantage Across Scenarios', fontweight='bold', fontsize=16)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')

    # Add horizontal line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_optimizer_advantage.png'))
    plt.savefig(os.path.join(output_dir, 'fig5_optimizer_advantage.pdf'))
    print(f"Saved: fig5_optimizer_advantage.png/pdf")
    plt.close()


def plot_combined_4panel(all_data: dict, output_dir: str):
    """
    Plot 6: Combined 4-panel figure (main paper figure)

    A comprehensive view of all trends in one figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    modes = ['pd', 'ppd', 'replica', 'optimizer']

    # Panel A: QPS Trend
    ax = axes[0, 0]
    data = all_data["qps_trend"]
    for mode in modes:
        ax.plot(data["qps"], data[mode], marker=MARKERS[mode], color=COLORS[mode],
                label=LABELS[mode], linewidth=2, markersize=6)
    ax.set_xlabel('QPS')
    ax.set_ylabel('SLO Attainment (%)')
    ax.set_title('(a) SLO Attainment vs Request Rate', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', fontsize=9)

    # Panel B: E2E Ratio Trend
    ax = axes[0, 1]
    data = all_data["e2e_ratio_trend"]
    for mode in modes:
        ax.plot(data["e2e_ratio"], data[mode], marker=MARKERS[mode], color=COLORS[mode],
                label=LABELS[mode], linewidth=2, markersize=6)
    ax.set_xlabel('E2E Task Ratio (%)')
    ax.set_ylabel('SLO Attainment (%)')
    ax.set_title('(b) SLO Attainment vs Workload Mix', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', fontsize=9)

    # Panel C: SLO Strictness Trend
    ax = axes[1, 0]
    data = all_data["slo_strictness_trend"]
    for mode in modes:
        ax.plot(data["strictness"], data[mode], marker=MARKERS[mode], color=COLORS[mode],
                label=LABELS[mode], linewidth=2, markersize=6)
    ax.set_xlabel('SLO Strictness Factor')
    ax.set_ylabel('SLO Attainment (%)')
    ax.set_title('(c) SLO Attainment vs SLO Strictness', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=9)

    # Panel D: Input Length Trend
    ax = axes[1, 1]
    data = all_data["input_length_trend"]
    for mode in modes:
        ax.plot(data["input_length"], data[mode], marker=MARKERS[mode], color=COLORS[mode],
                label=LABELS[mode], linewidth=2, markersize=6)
    ax.set_xlabel('Input Length (tokens)')
    ax.set_ylabel('SLO Attainment (%)')
    ax.set_title('(d) SLO Attainment vs Input Complexity', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_combined_4panel.png'))
    plt.savefig(os.path.join(output_dir, 'fig6_combined_4panel.pdf'))
    print(f"Saved: fig6_combined_4panel.png/pdf")
    plt.close()


def plot_advantage_trend(all_data: dict, output_dir: str):
    """
    Plot 7: Optimizer Advantage Trend

    Shows how optimizer advantage (vs best fixed mode) changes with QPS.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = all_data["qps_trend"]
    qps = data["qps"]

    # Calculate advantage at each QPS level
    advantages = []
    for i in range(len(qps)):
        opt = data["optimizer"][i]
        best_fixed = max(data["replica"][i], data["ppd"][i], data["pd"][i])
        advantages.append(opt - best_fixed)

    # Plot advantage
    colors = ['green' if a > 0 else 'red' for a in advantages]
    bars = ax.bar(qps, advantages, color=colors, edgecolor='black', linewidth=0.5, width=1.5)

    # Add value labels
    for i, (q, a) in enumerate(zip(qps, advantages)):
        va = 'bottom' if a >= 0 else 'top'
        offset = 1 if a >= 0 else -1
        ax.text(q, a + offset, f'{a:+.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('QPS (Concurrent Requests)', fontweight='bold')
    ax.set_ylabel('Optimizer Advantage (%)', fontweight='bold')
    ax.set_title('Optimizer Advantage vs Best Fixed Mode', fontweight='bold', fontsize=16)

    ax.set_xticks(qps)
    ax.set_ylim(min(advantages) - 5, max(advantages) + 10)

    # Add legend
    green_patch = mpatches.Patch(color='green', label='Optimizer Wins')
    red_patch = mpatches.Patch(color='red', label='Fixed Mode Wins')
    ax.legend(handles=[green_patch, red_patch], loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_advantage_trend.png'))
    plt.savefig(os.path.join(output_dir, 'fig7_advantage_trend.pdf'))
    print(f"Saved: fig7_advantage_trend.png/pdf")
    plt.close()


def main():
    print("="*70)
    print("GENERATING TREND FIGURES FOR PAPER")
    print("="*70)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "../results/trend_data.json")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run generate_trend_data.py first.")
        return

    all_data = load_trend_data(data_path)

    # Create output directory for figures
    output_dir = os.path.join(os.path.dirname(__file__), "../figures")
    os.makedirs(output_dir, exist_ok=True)

    # Generate all figures
    print("\nGenerating figures...")

    plot_qps_trend(all_data["qps_trend"], output_dir)
    plot_e2e_ratio_trend(all_data["e2e_ratio_trend"], output_dir)
    plot_slo_strictness_trend(all_data["slo_strictness_trend"], output_dir)
    plot_input_length_trend(all_data["input_length_trend"], output_dir)
    plot_optimizer_advantage(all_data, output_dir)
    plot_combined_4panel(all_data, output_dir)
    plot_advantage_trend(all_data, output_dir)

    print(f"\n{'='*70}")
    print(f"All figures saved to {output_dir}")
    print(f"{'='*70}")

    # Print summary statistics
    print("\n" + "="*70)
    print("FIGURE SUMMARY")
    print("="*70)

    for name, data in all_data.items():
        x_key = list(data.keys())[0]
        opt_values = data["optimizer"]
        replica_values = data["replica"]

        # Calculate average advantage
        advantages = [o - max(r, data["ppd"][i], data["pd"][i])
                     for i, (o, r) in enumerate(zip(opt_values, replica_values))]

        print(f"\n{name}:")
        print(f"  Range: {min(opt_values):.1f}% - {max(opt_values):.1f}%")
        print(f"  Avg Advantage: {np.mean(advantages):+.1f}%")
        print(f"  Max Advantage: {max(advantages):+.1f}%")
        print(f"  Win Rate: {sum(1 for a in advantages if a > 0)}/{len(advantages)}")


if __name__ == "__main__":
    main()
