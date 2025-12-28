#!/usr/bin/env python3
"""
Generate analysis visualizations and report from benchmark results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
DOCS_DIR = Path(__file__).parent.parent / "docs"


def load_results():
    """Load all result files."""
    benchmark = json.load(open(RESULTS_DIR / "benchmark_20251227_165711.json"))
    hol_blocking = json.load(open(RESULTS_DIR / "hol_blocking_suite_20251227_175613.json"))
    rag_bomb = json.load(open(RESULTS_DIR / "rag_bomb_turn2_20251227_175719.json"))
    return benchmark, hol_blocking, rag_bomb


def plot_speedup_by_scenario(benchmark, output_path):
    """Plot PPD speedup across all benchmark scenarios."""
    results = benchmark["multi_run_results"]

    names = [r["config_name"].split("_", 1)[1][:20] for r in results]
    speedups = [r["speedup_trimmed"] for r in results]

    # Color by speedup magnitude
    colors = ['#2ecc71' if s >= 1.1 else '#3498db' if s >= 1.05 else '#95a5a6' for s in speedups]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(names)), speedups, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='Baseline (PD)')
    ax.set_xlabel('PPD Speedup vs PD', fontsize=12)
    ax.set_title('PPD Performance Advantage Across Scenarios\n(Speedup > 1.0 means PPD is faster)', fontsize=14)

    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        ax.text(speedup + 0.005, i, f'{speedup:.2f}x', va='center', fontsize=8)

    ax.set_xlim(0.95, max(speedups) + 0.1)
    ax.legend(['PD Baseline'], loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_speedup_by_category(benchmark, output_path):
    """Plot speedup grouped by scenario category."""
    results = benchmark["multi_run_results"]

    # Categorize scenarios
    categories = {
        'Deep Session': [],
        'Long Context': [],
        'High Frequency': [],
        'Code/Creative': [],
        'Balanced': [],
        'Edge Cases': [],
    }

    for r in results:
        name = r["config_name"]
        speedup = r["speedup_trimmed"]
        if 'Deep' in name or 'Session' in name:
            categories['Deep Session'].append(speedup)
        elif 'Context' in name or 'RAG' in name:
            categories['Long Context'].append(speedup)
        elif 'Ping' in name or 'Frequency' in name:
            categories['High Frequency'].append(speedup)
        elif 'Code' in name or 'Creative' in name:
            categories['Code/Creative'].append(speedup)
        elif 'Balanced' in name or 'Burst' in name:
            categories['Balanced'].append(speedup)
        else:
            categories['Edge Cases'].append(speedup)

    # Calculate means
    cat_names = list(categories.keys())
    cat_means = [np.mean(v) if v else 1.0 for v in categories.values()]
    cat_stds = [np.std(v) if len(v) > 1 else 0 for v in categories.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cat_names)))
    bars = ax.bar(cat_names, cat_means, yerr=cat_stds, color=colors, capsize=5)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='PD Baseline')
    ax.set_ylabel('Average PPD Speedup', fontsize=12)
    ax.set_title('PPD Speedup by Workload Category', fontsize=14)
    ax.set_ylim(0.9, max(cat_means) + 0.1)

    # Add value labels
    for bar, mean in zip(bars, cat_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.2f}x', ha='center', fontsize=10)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_hol_blocking_comparison(hol_blocking, output_path):
    """Plot HOL blocking slowdown comparison."""
    scenarios = hol_blocking["scenarios"]

    names = [s["scenario_name"].split("_", 1)[1] for s in scenarios]
    pd_slowdown = [s["pd_slowdown_pct"] for s in scenarios]
    ppd_slowdown = [s["ppd_slowdown_pct"] for s in scenarios]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, pd_slowdown, width, label='PD Mode', color='#3498db')
    bars2 = ax.bar(x + width/2, ppd_slowdown, width, label='PPD Mode', color='#e74c3c')

    ax.set_ylabel('Victim Slowdown (%)', fontsize=12)
    ax.set_title('HOL Blocking: Victim Slowdown Under Interference\n(Lower is better - less interference to existing requests)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%',
                ha='center', fontsize=9, color='#3498db')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%',
                ha='center', fontsize=9, color='#e74c3c')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_decision_matrix(benchmark, hol_blocking, output_path):
    """Create a decision matrix showing when to use PD vs PPD."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create grid
    concurrency = ['Low\n(1-2 users)', 'Medium\n(3-5 users)', 'High\n(6+ users)']
    session_depth = ['Short\n(1-5 turns)', 'Medium\n(6-15 turns)', 'Deep\n(16+ turns)']

    # Decision matrix: 1 = PPD better, 0 = PD better, 0.5 = similar
    # Based on analysis: PPD wins in low concurrency & deep sessions
    # PD wins in high concurrency (isolation)
    matrix = np.array([
        [1.0, 1.0, 1.0],   # Low concurrency: PPD always wins
        [0.7, 0.8, 0.9],   # Medium: PPD usually wins, especially with deeper sessions
        [0.3, 0.4, 0.5],   # High: PD wins due to isolation
    ])

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(session_depth)))
    ax.set_xticklabels(session_depth, fontsize=11)
    ax.set_yticks(range(len(concurrency)))
    ax.set_yticklabels(concurrency, fontsize=11)
    ax.set_xlabel('Session Depth (Conversation Length)', fontsize=12)
    ax.set_ylabel('System Concurrency (Parallel Users)', fontsize=12)
    ax.set_title('PD vs PPD Decision Matrix\n(Green = PPD Recommended, Red = PD Recommended)', fontsize=14)

    # Add text annotations
    labels = [
        ['PPD\n(1.31x)', 'PPD\n(1.19x)', 'PPD\n(1.05x)'],
        ['PPD', 'PPD', 'PPD'],
        ['PD\n(isolation)', 'PD/PPD', 'PPD'],
    ]
    for i in range(len(concurrency)):
        for j in range(len(session_depth)):
            text = ax.text(j, i, labels[i][j], ha='center', va='center',
                          fontsize=10, fontweight='bold',
                          color='white' if matrix[i, j] < 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_breakdown(benchmark, output_path):
    """Plot latency comparison for key scenarios."""
    results = benchmark["multi_run_results"]

    # Select representative scenarios
    key_scenarios = [
        "01_Baseline_Short_Chat",
        "05_Deep_Session_Extreme_50T",
        "14_High_Frequency_Ping",
        "19_Edge_Huge_History_Tiny_Input",
        "08_Context_3K_RAG",
    ]

    selected = [r for r in results if r["config_name"] in key_scenarios]

    names = [r["config_name"].split("_", 1)[1][:18] for r in selected]
    pd_times = [r["pd_trimmed_mean_ms"] for r in selected]
    ppd_times = [r["ppd_trimmed_mean_ms"] for r in selected]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, pd_times, width, label='PD Mode', color='#3498db')
    bars2 = ax.bar(x + width/2, ppd_times, width, label='PPD Mode', color='#2ecc71')

    ax.set_ylabel('Total Latency (ms)', fontsize=12)
    ax.set_title('Latency Comparison: Key Scenarios', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_stats(benchmark, hol_blocking):
    """Generate summary statistics."""
    results = benchmark["multi_run_results"]

    speedups = [r["speedup_trimmed"] for r in results]

    stats = {
        "total_scenarios": len(results),
        "ppd_wins": sum(1 for s in speedups if s > 1.0),
        "mean_speedup": np.mean(speedups),
        "max_speedup": max(speedups),
        "min_speedup": min(speedups),
        "best_scenario": results[speedups.index(max(speedups))]["config_name"],
        "worst_scenario": results[speedups.index(min(speedups))]["config_name"],
    }

    # HOL blocking stats
    hol_scenarios = hol_blocking["scenarios"]
    pd_wins_hol = sum(1 for s in hol_scenarios if s["pd_slowdown_pct"] < s["ppd_slowdown_pct"])

    stats["hol_pd_wins"] = pd_wins_hol
    stats["hol_total"] = len(hol_scenarios)

    return stats


def main():
    DOCS_DIR.mkdir(exist_ok=True)

    print("Loading results...")
    benchmark, hol_blocking, rag_bomb = load_results()

    print("\nGenerating visualizations...")
    plot_speedup_by_scenario(benchmark, DOCS_DIR / "speedup_by_scenario.png")
    plot_speedup_by_category(benchmark, DOCS_DIR / "speedup_by_category.png")
    plot_hol_blocking_comparison(hol_blocking, DOCS_DIR / "hol_blocking_comparison.png")
    plot_decision_matrix(benchmark, hol_blocking, DOCS_DIR / "decision_matrix.png")
    plot_latency_breakdown(benchmark, DOCS_DIR / "latency_breakdown.png")

    print("\nGenerating summary statistics...")
    stats = generate_summary_stats(benchmark, hol_blocking)

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total benchmark scenarios: {stats['total_scenarios']}")
    print(f"PPD faster in: {stats['ppd_wins']}/{stats['total_scenarios']} scenarios")
    print(f"Mean speedup: {stats['mean_speedup']:.3f}x")
    print(f"Max speedup: {stats['max_speedup']:.3f}x ({stats['best_scenario']})")
    print(f"Min speedup: {stats['min_speedup']:.3f}x ({stats['worst_scenario']})")
    print(f"\nHOL Blocking: PD wins {stats['hol_pd_wins']}/{stats['hol_total']} scenarios")
    print("="*60)

    # Save stats
    with open(DOCS_DIR / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {DOCS_DIR / 'summary_stats.json'}")


if __name__ == "__main__":
    main()
