#!/usr/bin/env python3
"""
Interference Benchmark Visualization

Generates publication-quality figures for PPD paper:
- Figure 1: Core TPOT comparison (decoding_only vs full-prefill vs append-prefill)
- Figure 2: Slowdown percentage comparison (6 lines)
- Figure 3: Sensitivity analysis (append-prefill input length impact)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'baseline': '#1f77b4',  # Blue
    '1_full': '#d62728',    # Red
    '2_full': '#ff7f0e',    # Orange
    '4_full': '#ffbb78',    # Light orange
    '1_append': '#2ca02c',  # Green
    '2_append': '#98df8a',  # Light green
    '4_append': '#9467bd',  # Purple (for visibility)
}

MARKERS = {
    'baseline': 'o',
    '1_full': '^',
    '2_full': 's',
    '4_full': 'D',
    '1_append': 'v',
    '2_append': 'p',
    '4_append': '*',
}


def load_data():
    """Load experiment data from JSON files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(script_dir, 'core_experiment.json')) as f:
        core_data = json.load(f)

    with open(os.path.join(script_dir, 'sensitivity_experiment.json')) as f:
        sens_data = json.load(f)

    return core_data, sens_data


def process_core_data(core_data):
    """Process core experiment data, computing mean and std."""
    results_by_scenario_batch = defaultdict(list)

    for r in core_data['results']:
        decode_count = r.get('decode_count', 0)
        scenario = r['scenario']
        tpot = r.get('decode_avg_tpot_ms', 0)
        if tpot > 0:  # Only include valid data
            results_by_scenario_batch[(scenario, decode_count)].append(tpot)

    # Get unique batch sizes (sorted)
    batch_sizes = sorted(set(dc for (_, dc) in results_by_scenario_batch.keys()))

    # Compute stats for each scenario
    scenarios = ['decoding_only',
                 'decoding_with_1_full_prefill', 'decoding_with_1_append_prefill',
                 'decoding_with_2_full_prefill', 'decoding_with_2_append_prefill',
                 'decoding_with_4_full_prefill', 'decoding_with_4_append_prefill']

    stats = {}
    for scenario in scenarios:
        means = []
        stds = []
        valid_batches = []
        for bs in batch_sizes:
            data = results_by_scenario_batch.get((scenario, bs), [])
            if data:
                means.append(np.mean(data))
                stds.append(np.std(data))
                valid_batches.append(bs)
        stats[scenario] = {
            'batch_sizes': valid_batches,
            'means': np.array(means),
            'stds': np.array(stds)
        }

    return stats, batch_sizes


def process_sensitivity_data(sens_data):
    """Process sensitivity experiment data."""
    # Get baseline
    baseline_results = [r for r in sens_data['results']
                       if r['scenario'] == 'decoding_only_baseline']
    baseline_tpots = [r['decode_avg_tpot_ms'] for r in baseline_results if r['decode_avg_tpot_ms'] > 0]
    baseline_mean = np.mean(baseline_tpots)
    baseline_std = np.std(baseline_tpots)

    # Group by input length
    results_by_length = defaultdict(list)
    for r in sens_data['results']:
        if 'append_input_length' in r and r['decode_avg_tpot_ms'] > 0:
            results_by_length[r['append_input_length']].append(r['decode_avg_tpot_ms'])

    input_lengths = sorted(results_by_length.keys())
    means = []
    stds = []
    slowdowns = []
    slowdown_stds = []

    for length in input_lengths:
        data = results_by_length[length]
        mean = np.mean(data)
        std = np.std(data)
        means.append(mean)
        stds.append(std)
        slowdowns.append((mean - baseline_mean) / baseline_mean * 100)
        # Propagate error for slowdown
        slowdown_stds.append(std / baseline_mean * 100)

    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'input_lengths': input_lengths,
        'means': np.array(means),
        'stds': np.array(stds),
        'slowdowns': np.array(slowdowns),
        'slowdown_stds': np.array(slowdown_stds)
    }


def plot_figure1(stats, output_dir):
    """
    Figure 1: Core TPOT comparison
    (a) 1 prefill inserted
    (b) 4 prefills inserted
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) 1 prefill
    ax = axes[0]

    # Baseline
    s = stats['decoding_only']
    ax.plot(s['batch_sizes'], s['means'],
            color=COLORS['baseline'], marker=MARKERS['baseline'],
            label='decoding-only', linewidth=1.5, markersize=4)
    ax.fill_between(s['batch_sizes'], s['means'] - s['stds'], s['means'] + s['stds'],
                    color=COLORS['baseline'], alpha=0.2)

    # 1 full-prefill
    s = stats['decoding_with_1_full_prefill']
    ax.plot(s['batch_sizes'], s['means'],
            color=COLORS['1_full'], marker=MARKERS['1_full'],
            label='decoding + 1 full-prefill', linewidth=1.5, markersize=4, linestyle='--')
    ax.fill_between(s['batch_sizes'], s['means'] - s['stds'], s['means'] + s['stds'],
                    color=COLORS['1_full'], alpha=0.2)

    # 1 append-prefill
    s = stats['decoding_with_1_append_prefill']
    ax.plot(s['batch_sizes'], s['means'],
            color=COLORS['1_append'], marker=MARKERS['1_append'],
            label='decoding + 1 append-prefill', linewidth=1.5, markersize=4)
    ax.fill_between(s['batch_sizes'], s['means'] - s['stds'], s['means'] + s['stds'],
                    color=COLORS['1_append'], alpha=0.2)

    ax.set_xlabel('Decode Batch Size')
    ax.set_ylabel('Average TPOT (ms)')
    ax.set_title('(a) Interference with 1 Prefill')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 260)
    ax.set_ylim(0, None)

    # Add annotation for slowdown at batch=200
    baseline_200 = stats['decoding_only']['means'][stats['decoding_only']['batch_sizes'].index(200)] if 200 in stats['decoding_only']['batch_sizes'] else None
    full_200 = stats['decoding_with_1_full_prefill']['means'][stats['decoding_with_1_full_prefill']['batch_sizes'].index(200)] if 200 in stats['decoding_with_1_full_prefill']['batch_sizes'] else None
    append_200 = stats['decoding_with_1_append_prefill']['means'][stats['decoding_with_1_append_prefill']['batch_sizes'].index(200)] if 200 in stats['decoding_with_1_append_prefill']['batch_sizes'] else None

    if baseline_200 and full_200 and append_200:
        full_slowdown = (full_200 - baseline_200) / baseline_200 * 100
        append_slowdown = (append_200 - baseline_200) / baseline_200 * 100
        ax.annotate(f'+{full_slowdown:.0f}%', xy=(200, full_200), xytext=(210, full_200 + 10),
                   fontsize=8, color=COLORS['1_full'])
        ax.annotate(f'+{append_slowdown:.0f}%', xy=(200, append_200), xytext=(210, append_200 + 5),
                   fontsize=8, color=COLORS['1_append'])

    # (b) 4 prefills
    ax = axes[1]

    # Baseline
    s = stats['decoding_only']
    ax.plot(s['batch_sizes'], s['means'],
            color=COLORS['baseline'], marker=MARKERS['baseline'],
            label='decoding-only', linewidth=1.5, markersize=4)
    ax.fill_between(s['batch_sizes'], s['means'] - s['stds'], s['means'] + s['stds'],
                    color=COLORS['baseline'], alpha=0.2)

    # 4 full-prefills
    s = stats['decoding_with_4_full_prefill']
    ax.plot(s['batch_sizes'], s['means'],
            color=COLORS['4_full'], marker=MARKERS['4_full'],
            label='decoding + 4 full-prefills', linewidth=1.5, markersize=4, linestyle='--')
    ax.fill_between(s['batch_sizes'], s['means'] - s['stds'], s['means'] + s['stds'],
                    color=COLORS['4_full'], alpha=0.2)

    # 4 append-prefills
    s = stats['decoding_with_4_append_prefill']
    ax.plot(s['batch_sizes'], s['means'],
            color=COLORS['4_append'], marker=MARKERS['4_append'],
            label='decoding + 4 append-prefills', linewidth=1.5, markersize=4)
    ax.fill_between(s['batch_sizes'], s['means'] - s['stds'], s['means'] + s['stds'],
                    color=COLORS['4_append'], alpha=0.2)

    ax.set_xlabel('Decode Batch Size')
    ax.set_ylabel('Average TPOT (ms)')
    ax.set_title('(b) Interference with 4 Prefills')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 260)
    ax.set_ylim(0, None)

    # Add annotation for slowdown at batch=200
    baseline_200 = stats['decoding_only']['means'][stats['decoding_only']['batch_sizes'].index(200)] if 200 in stats['decoding_only']['batch_sizes'] else None
    full_200 = stats['decoding_with_4_full_prefill']['means'][stats['decoding_with_4_full_prefill']['batch_sizes'].index(200)] if 200 in stats['decoding_with_4_full_prefill']['batch_sizes'] else None
    append_200 = stats['decoding_with_4_append_prefill']['means'][stats['decoding_with_4_append_prefill']['batch_sizes'].index(200)] if 200 in stats['decoding_with_4_append_prefill']['batch_sizes'] else None

    if baseline_200 and full_200 and append_200:
        full_slowdown = (full_200 - baseline_200) / baseline_200 * 100
        append_slowdown = (append_200 - baseline_200) / baseline_200 * 100
        ax.annotate(f'+{full_slowdown:.0f}%', xy=(200, full_200), xytext=(210, full_200 + 10),
                   fontsize=8, color=COLORS['4_full'])
        ax.annotate(f'+{append_slowdown:.0f}%', xy=(200, append_200), xytext=(210, append_200 + 5),
                   fontsize=8, color=COLORS['4_append'])

    plt.tight_layout()

    # Save
    fig.savefig(os.path.join(output_dir, 'fig1_core_tpot_comparison.pdf'))
    fig.savefig(os.path.join(output_dir, 'fig1_core_tpot_comparison.png'))
    plt.close(fig)
    print(f"Saved Figure 1 to {output_dir}")


def plot_figure2(stats, output_dir):
    """
    Figure 2: Slowdown percentage comparison
    4 prefill scenarios (1 and 4 only, removed 2 for clarity)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    baseline = stats['decoding_only']

    # Only keep 1 and 4 prefill scenarios (removed 2 for cleaner visualization)
    scenarios_to_plot = [
        ('decoding_with_1_full_prefill', '1 full-prefill', COLORS['1_full'], MARKERS['1_full'], '--'),
        ('decoding_with_4_full_prefill', '4 full-prefills', COLORS['4_full'], MARKERS['4_full'], '--'),
        ('decoding_with_1_append_prefill', '1 append-prefill', COLORS['1_append'], MARKERS['1_append'], '-'),
        ('decoding_with_4_append_prefill', '4 append-prefills', COLORS['4_append'], MARKERS['4_append'], '-'),
    ]

    # Draw baseline at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='baseline (0%)')

    for scenario, label, color, marker, linestyle in scenarios_to_plot:
        s = stats[scenario]

        # Compute slowdown for each batch size
        slowdowns = []
        slowdown_stds = []
        valid_batches = []

        for i, bs in enumerate(s['batch_sizes']):
            if bs in baseline['batch_sizes']:
                idx = baseline['batch_sizes'].index(bs)
                base_mean = baseline['means'][idx]
                if base_mean > 0:
                    slowdown = (s['means'][i] - base_mean) / base_mean * 100
                    slowdowns.append(slowdown)
                    # Propagate error
                    slowdown_std = s['stds'][i] / base_mean * 100
                    slowdown_stds.append(slowdown_std)
                    valid_batches.append(bs)

        slowdowns = np.array(slowdowns)
        slowdown_stds = np.array(slowdown_stds)

        ax.plot(valid_batches, slowdowns, color=color, marker=marker,
                label=label, linewidth=1.5, markersize=4, linestyle=linestyle)
        ax.fill_between(valid_batches, slowdowns - slowdown_stds, slowdowns + slowdown_stds,
                        color=color, alpha=0.15)

    ax.set_xlabel('Decode Batch Size')
    ax.set_ylabel('Decode Slowdown (%)')
    ax.set_title('Prefill-Decode Interference: Slowdown Comparison')
    ax.legend(loc='upper left', ncol=2)
    ax.set_xlim(0, 260)

    # Add horizontal regions for context
    ax.axhspan(-20, 20, alpha=0.1, color='green', label='_nolegend_')
    ax.axhspan(80, 160, alpha=0.1, color='red', label='_nolegend_')

    # Add text annotations
    ax.text(250, 10, 'Append-prefill\n(PPD mode)', fontsize=8, ha='right', va='center', color='green')
    ax.text(250, 120, 'Full-prefill\n(PD mode)', fontsize=8, ha='right', va='center', color='red')

    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, 'fig2_slowdown_percentage.pdf'))
    fig.savefig(os.path.join(output_dir, 'fig2_slowdown_percentage.png'))
    plt.close(fig)
    print(f"Saved Figure 2 to {output_dir}")


def plot_figure3(sens_stats, output_dir):
    """
    Figure 3: Sensitivity analysis - append-prefill input length impact
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = sens_stats['input_lengths']
    y = sens_stats['slowdowns']
    yerr = sens_stats['slowdown_stds']

    # Draw baseline at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='baseline (0%)')

    # Plot slowdown curve
    ax.errorbar(x, y, yerr=yerr, color=COLORS['1_append'], marker='o',
                linewidth=2, markersize=6, capsize=3, label='Decode slowdown')
    ax.fill_between(x, y - yerr, y + yerr, color=COLORS['1_append'], alpha=0.2)

    # Add reference line for full-prefill typical slowdown
    ax.axhline(y=100, color=COLORS['1_full'], linestyle=':', linewidth=1.5, alpha=0.7,
               label='Full-prefill typical (~100%)')

    ax.set_xlabel('Append-prefill Input Length (tokens)')
    ax.set_ylabel('Decode Slowdown (%)')
    ax.set_title('Append-prefill Input Length Sensitivity (batch=64)')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1100)
    ax.set_ylim(-10, 120)

    # Add annotations for key points
    key_points = [64, 512, 1024]
    for kp in key_points:
        if kp in x:
            idx = list(x).index(kp)
            slowdown = y[idx]
            ax.annotate(f'{kp}: {slowdown:+.0f}%',
                       xy=(kp, slowdown),
                       xytext=(kp + 50, slowdown + 15),
                       fontsize=8,
                       arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    # Add text box with conclusion
    textstr = 'Even at 1024 tokens,\nslowdown << full-prefill'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.65, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, 'fig3_sensitivity_input_length.pdf'))
    fig.savefig(os.path.join(output_dir, 'fig3_sensitivity_input_length.png'))
    plt.close(fig)
    print(f"Saved Figure 3 to {output_dir}")


def main():
    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    core_data, sens_data = load_data()

    print("Processing core experiment data...")
    core_stats, batch_sizes = process_core_data(core_data)

    print("Processing sensitivity experiment data...")
    sens_stats = process_sensitivity_data(sens_data)

    print("\nGenerating figures...")

    print("  Figure 1: Core TPOT comparison...")
    plot_figure1(core_stats, output_dir)

    print("  Figure 2: Slowdown percentage comparison...")
    plot_figure2(core_stats, output_dir)

    print("  Figure 3: Sensitivity analysis...")
    plot_figure3(sens_stats, output_dir)

    print(f"\nAll figures saved to: {output_dir}")
    print("Files generated:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
