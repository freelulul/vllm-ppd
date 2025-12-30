#!/usr/bin/env python3
"""
Generate QPS vs Latency curves from benchmark results.

This script creates the standard figures for system papers:
- Figure 1: QPS vs P99 TTFT for each workload
- Figure 2: QPS vs Avg E2E Latency for each workload
- Figure 3: Combined comparison showing crossover points
- Figure 4: Success rate vs QPS
- Figure 5: Throughput vs QPS
- Figure 6: Throughput comparison bar chart

Supports both 2-mode (PD/PPD) and 3-mode (Replication/PPD/PD) comparisons.
Automatically detects available modes from the data.

Usage:
    python scripts/plot_qps_curves.py results/qps_benchmark_*.json
    python scripts/plot_qps_curves.py results/merged_3mode_*.json  # For 3-mode comparison
    python scripts/plot_qps_curves.py --latest  # Use most recent result
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
DOCS_DIR = Path(__file__).parent.parent / "docs"


def load_results(filepath: str = None) -> dict:
    """Load QPS benchmark results."""
    if filepath:
        with open(filepath) as f:
            return json.load(f)

    # Find latest result
    result_files = sorted(RESULTS_DIR.glob("qps_benchmark_*.json"))
    if not result_files:
        raise FileNotFoundError("No QPS benchmark results found")

    with open(result_files[-1]) as f:
        return json.load(f)


def extract_data(results: dict) -> tuple[dict, list]:
    """Extract data organized by workload and mode. Returns (data, available_modes)."""
    data = {}
    all_modes = set()

    for r in results["results"]:
        wk = r["workload"]
        mode = r["mode"]
        qps = r["target_qps"]
        all_modes.add(mode)

        if wk not in data:
            data[wk] = {}

        if mode not in data[wk]:
            data[wk][mode] = {}

        data[wk][mode][qps] = {
            "p50_ttft": r["p50_ttft"],
            "p90_ttft": r["p90_ttft"],
            "p99_ttft": r["p99_ttft"],
            "avg_e2e": r["avg_e2e"],
            "p99_e2e": r["p99_e2e"],
            "success_rate": r["success_count"] / r["sample_count"] if r["sample_count"] > 0 else 0,
            "real_qps": r["real_qps"],
            "avg_throughput_tps": r.get("avg_throughput_tps", 0),
            "total_tokens": r.get("total_tokens", 0),
        }

    # Determine mode order (replication first if present, then ppd, then pd)
    available_modes = []
    for m in ["replication", "ppd", "pd"]:
        if m in all_modes:
            available_modes.append(m)

    return data, available_modes


def plot_qps_curves(data: dict, output_dir: Path, available_modes: list):
    """Generate QPS vs Latency curves."""
    workloads = list(data.keys())
    n_workloads = len(workloads)

    if n_workloads == 0:
        print("No data to plot")
        return

    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Colors and markers for up to 3 modes
    colors = {"replication": "#e74c3c", "ppd": "#2ecc71", "pd": "#3498db"}
    markers = {"replication": "^", "ppd": "o", "pd": "s"}
    mode_labels = {"replication": "Replication", "ppd": "PPD", "pd": "PD"}

    # Determine title based on available modes
    has_replication = "replication" in available_modes
    if has_replication:
        title_suffix = "Replication vs PPD vs PD"
    else:
        title_suffix = "PD vs PPD Mode"

    # ================================================================
    # Figure 1: P99 TTFT vs QPS (2x2 subplots)
    # ================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    axes1 = axes1.flatten()

    for idx, wk in enumerate(workloads[:4]):
        ax = axes1[idx]
        wk_data = data[wk]

        for mode in available_modes:
            if mode not in wk_data or not wk_data[mode]:
                continue

            qps_vals = sorted(wk_data[mode].keys())
            p99_vals = [wk_data[mode][q]["p99_ttft"] for q in qps_vals]

            ax.plot(qps_vals, p99_vals,
                    marker=markers[mode], color=colors[mode],
                    linewidth=2, markersize=8, label=mode_labels[mode])

        ax.set_xlabel("Request Rate (QPS)", fontsize=12)
        ax.set_ylabel("P99 TTFT (ms)", fontsize=12)
        ax.set_title(wk.replace("_", " "), fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Log scale for y-axis if range is large
        y_vals = [wk_data[m][q]["p99_ttft"] for m in wk_data for q in wk_data[m] if wk_data[m]]
        if y_vals and max(y_vals) / (min(y_vals) + 0.1) > 10:
            ax.set_yscale('log')

    # Hide unused subplots
    for idx in range(len(workloads), 4):
        axes1[idx].set_visible(False)

    fig1.suptitle(f"QPS vs P99 TTFT: {title_suffix}\n(Lower is better)", fontsize=16)
    plt.tight_layout()
    fig1.savefig(output_dir / "qps_p99_ttft.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {output_dir / 'qps_p99_ttft.png'}")

    # ================================================================
    # Figure 2: Avg E2E Latency vs QPS (2x2 subplots)
    # ================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    axes2 = axes2.flatten()

    for idx, wk in enumerate(workloads[:4]):
        ax = axes2[idx]
        wk_data = data[wk]

        for mode in available_modes:
            if mode not in wk_data or not wk_data[mode]:
                continue

            qps_vals = sorted(wk_data[mode].keys())
            e2e_vals = [wk_data[mode][q]["avg_e2e"] for q in qps_vals]

            ax.plot(qps_vals, e2e_vals,
                    marker=markers[mode], color=colors[mode],
                    linewidth=2, markersize=8, label=mode_labels[mode])

        ax.set_xlabel("Request Rate (QPS)", fontsize=12)
        ax.set_ylabel("Avg E2E Latency (ms)", fontsize=12)
        ax.set_title(wk.replace("_", " "), fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    for idx in range(len(workloads), 4):
        axes2[idx].set_visible(False)

    fig2.suptitle(f"QPS vs Average E2E Latency: {title_suffix}\n(Lower is better)", fontsize=16)
    plt.tight_layout()
    fig2.savefig(output_dir / "qps_avg_e2e.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {output_dir / 'qps_avg_e2e.png'}")

    # ================================================================
    # Figure 3: Latency Ratio vs Baseline (shows crossover point)
    # ================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
    axes3 = axes3.flatten()

    # For 3-mode: use replication as baseline; for 2-mode: use PD as baseline
    if has_replication:
        baseline_mode = "replication"
        compare_modes = [m for m in available_modes if m != "replication"]
        ratio_title = "P99 TTFT Ratio vs Replication Baseline\n(Below 1.0 = Disaggregation wins)"
    else:
        baseline_mode = "pd"
        compare_modes = ["ppd"]
        ratio_title = "PPD vs PD Performance Ratio\n(Below 1.0 = PPD wins, Above 1.0 = PD wins)"

    for idx, wk in enumerate(workloads[:4]):
        ax = axes3[idx]
        wk_data = data[wk]

        if baseline_mode not in wk_data or not wk_data[baseline_mode]:
            continue

        baseline_qps = set(wk_data[baseline_mode].keys())

        for mode in compare_modes:
            if mode not in wk_data or not wk_data[mode]:
                continue

            common_qps = sorted(baseline_qps & set(wk_data[mode].keys()))
            if not common_qps:
                continue

            ratios = []
            for q in common_qps:
                baseline_ttft = wk_data[baseline_mode][q]["p99_ttft"]
                mode_ttft = wk_data[mode][q]["p99_ttft"]
                if baseline_ttft > 0:
                    ratios.append(mode_ttft / baseline_ttft)
                else:
                    ratios.append(1.0)

            ax.plot(common_qps, ratios,
                    marker=markers[mode], color=colors[mode],
                    linewidth=2, markersize=8, label=mode_labels[mode])

        ax.axhline(y=1.0, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel("Request Rate (QPS)", fontsize=12)
        ax.set_ylabel(f"P99 TTFT Ratio (vs {mode_labels[baseline_mode]})", fontsize=12)
        ax.set_title(wk.replace("_", " "), fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add shaded regions
        ax.axhspan(0, 1.0, alpha=0.1, color='green')
        ax.axhspan(1.0, 2.0, alpha=0.1, color='red')

    for idx in range(len(workloads), 4):
        axes3[idx].set_visible(False)

    fig3.suptitle(ratio_title, fontsize=14)
    plt.tight_layout()
    fig3.savefig(output_dir / "qps_ratio_crossover.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {output_dir / 'qps_ratio_crossover.png'}")

    # ================================================================
    # Figure 4: Success Rate vs QPS (system stability)
    # ================================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))
    axes4 = axes4.flatten()

    for idx, wk in enumerate(workloads[:4]):
        ax = axes4[idx]
        wk_data = data[wk]

        for mode in available_modes:
            if mode not in wk_data or not wk_data[mode]:
                continue

            qps_vals = sorted(wk_data[mode].keys())
            success_vals = [wk_data[mode][q]["success_rate"] * 100 for q in qps_vals]

            ax.plot(qps_vals, success_vals,
                    marker=markers[mode], color=colors[mode],
                    linewidth=2, markersize=8, label=mode_labels[mode])

        ax.set_xlabel("Request Rate (QPS)", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(wk.replace("_", " "), fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    for idx in range(len(workloads), 4):
        axes4[idx].set_visible(False)

    fig4.suptitle(f"QPS vs Success Rate: {title_suffix}\n(Higher is better)", fontsize=16)
    plt.tight_layout()
    fig4.savefig(output_dir / "qps_success_rate.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {output_dir / 'qps_success_rate.png'}")

    # ================================================================
    # Figure 5: Throughput vs QPS (tokens per second)
    # ================================================================
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 12))
    axes5 = axes5.flatten()

    for idx, wk in enumerate(workloads[:4]):
        ax = axes5[idx]
        wk_data = data[wk]

        for mode in available_modes:
            if mode not in wk_data or not wk_data[mode]:
                continue

            qps_vals = sorted(wk_data[mode].keys())
            tps_vals = [wk_data[mode][q]["avg_throughput_tps"] for q in qps_vals]

            ax.plot(qps_vals, tps_vals,
                    marker=markers[mode], color=colors[mode],
                    linewidth=2, markersize=8, label=mode_labels[mode])

        ax.set_xlabel("Request Rate (QPS)", fontsize=12)
        ax.set_ylabel("Avg Throughput (tokens/sec)", fontsize=12)
        ax.set_title(wk.replace("_", " "), fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    for idx in range(len(workloads), 4):
        axes5[idx].set_visible(False)

    fig5.suptitle(f"QPS vs Throughput: {title_suffix}\n(Higher is better)", fontsize=16)
    plt.tight_layout()
    fig5.savefig(output_dir / "qps_throughput.png", dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"Saved: {output_dir / 'qps_throughput.png'}")

    # ================================================================
    # Figure 6: Aggregate Throughput Comparison Bar Chart
    # ================================================================
    fig6, ax6 = plt.subplots(figsize=(14, 8))

    n_modes = len(available_modes)
    bar_width = 0.8 / n_modes
    x_positions = np.arange(len(workloads))

    # Get throughput at highest QPS for each workload
    max_qps = max(qps_vals) if qps_vals else 8.0

    mode_tps = {mode: [] for mode in available_modes}
    for wk in workloads:
        wk_data = data[wk]
        for mode in available_modes:
            val = wk_data.get(mode, {}).get(max_qps, {}).get("avg_throughput_tps", 0)
            mode_tps[mode].append(val)

    bars_all = []
    for i, mode in enumerate(available_modes):
        offset = (i - (n_modes - 1) / 2) * bar_width
        bars = ax6.bar(x_positions + offset, mode_tps[mode], bar_width,
                       label=mode_labels[mode], color=colors[mode])
        bars_all.append((bars, mode_tps[mode]))

    ax6.set_xlabel("Workload", fontsize=12)
    ax6.set_ylabel(f"Avg Throughput @ QPS={max_qps} (tokens/sec)", fontsize=12)
    ax6.set_title(f"Throughput Comparison at Peak Load (QPS={max_qps})\n(Higher is better)", fontsize=14)
    ax6.set_xticks(x_positions)
    ax6.set_xticklabels([wk.replace("_", "\n") for wk in workloads], fontsize=10)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars, vals in bars_all:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax6.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig6.savefig(output_dir / "qps_throughput_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print(f"Saved: {output_dir / 'qps_throughput_comparison.png'}")


def generate_crossover_analysis(data: dict, available_modes: list) -> dict:
    """Analyze crossover points and mode advantages."""
    analysis = {}
    has_replication = "replication" in available_modes

    for wk, wk_data in data.items():
        wk_analysis = {"winners_by_qps": {}}

        # Get common QPS across all available modes
        all_qps_sets = [set(wk_data[m].keys()) for m in available_modes if m in wk_data and wk_data[m]]
        if len(all_qps_sets) < 2:
            continue

        common_qps = sorted(set.intersection(*all_qps_sets))
        if not common_qps:
            continue

        # Find winner at each QPS
        for q in common_qps:
            modes_ttft = {}
            for mode in available_modes:
                if mode in wk_data and q in wk_data[mode]:
                    modes_ttft[mode] = wk_data[mode][q]["p99_ttft"]
            if modes_ttft:
                winner = min(modes_ttft, key=modes_ttft.get)
                wk_analysis["winners_by_qps"][q] = winner

        # If replication is present, analyze PPD/PD vs replication
        if has_replication and "replication" in wk_data:
            for mode in ["ppd", "pd"]:
                if mode not in wk_data:
                    continue

                # Find where mode beats replication
                beats_repl_qps = None
                max_advantage = 0
                max_advantage_qps = None

                for q in common_qps:
                    repl_ttft = wk_data["replication"].get(q, {}).get("p99_ttft", 0)
                    mode_ttft = wk_data[mode].get(q, {}).get("p99_ttft", 0)
                    if repl_ttft > 0 and mode_ttft > 0:
                        if mode_ttft < repl_ttft and beats_repl_qps is None:
                            beats_repl_qps = q
                        advantage = (repl_ttft - mode_ttft) / repl_ttft
                        if advantage > max_advantage:
                            max_advantage = advantage
                            max_advantage_qps = q

                wk_analysis[f"{mode}_beats_repl_qps"] = beats_repl_qps
                wk_analysis[f"max_{mode}_vs_repl_advantage_pct"] = max_advantage * 100
                wk_analysis[f"max_{mode}_vs_repl_qps"] = max_advantage_qps
        else:
            # 2-mode analysis: PPD vs PD
            if "ppd" in wk_data and "pd" in wk_data:
                crossover_qps = None
                max_advantage = 0
                max_advantage_qps = None

                for q in common_qps:
                    ppd_ttft = wk_data["ppd"][q]["p99_ttft"]
                    pd_ttft = wk_data["pd"][q]["p99_ttft"]
                    if ppd_ttft > pd_ttft and crossover_qps is None:
                        crossover_qps = q
                    if pd_ttft > 0:
                        advantage = (pd_ttft - ppd_ttft) / pd_ttft
                        if advantage > max_advantage:
                            max_advantage = advantage
                            max_advantage_qps = q

                wk_analysis["crossover_qps"] = crossover_qps
                wk_analysis["max_ppd_advantage_qps"] = max_advantage_qps
                wk_analysis["max_ppd_advantage_pct"] = max_advantage * 100

        analysis[wk] = wk_analysis

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Plot QPS vs Latency curves")
    parser.add_argument("result_file", nargs="?", help="Path to QPS benchmark result JSON")
    parser.add_argument("--latest", action="store_true", help="Use latest result file")
    parser.add_argument("--output-dir", type=str, default=str(DOCS_DIR),
                        help="Output directory for figures")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        if args.result_file:
            results = load_results(args.result_file)
        else:
            results = load_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run qps_benchmark.py first to generate results.")
        return

    print(f"Loaded results from: {results.get('timestamp', 'unknown')}")
    print(f"Workloads: {len(results['config']['workloads'])}")
    print(f"QPS sweep: {results['config']['qps_sweep']}")

    # Extract and plot
    data, available_modes = extract_data(results)
    print(f"Available modes: {available_modes}")

    plot_qps_curves(data, output_dir, available_modes)

    # Generate crossover analysis
    analysis = generate_crossover_analysis(data, available_modes)

    has_replication = "replication" in available_modes

    print("\n" + "=" * 70)
    if has_replication:
        print("CROSSOVER ANALYSIS (3-Mode Comparison)")
    else:
        print("CROSSOVER ANALYSIS")
    print("=" * 70)

    for wk, a in analysis.items():
        print(f"\n{wk}:")

        # Show winners by QPS
        winners = a.get("winners_by_qps", {})
        if winners:
            winner_str = ", ".join([f"QPS {q}: {w.upper()}" for q, w in sorted(winners.items())])
            print(f"  Winners: {winner_str}")

        if has_replication:
            # 3-mode analysis
            if a.get("ppd_beats_repl_qps"):
                print(f"  PPD beats Replication starting at QPS = {a['ppd_beats_repl_qps']}")
            if a.get("max_ppd_vs_repl_qps"):
                print(f"  Max PPD advantage vs Replication: {a.get('max_ppd_vs_repl_advantage_pct', 0):.1f}% at QPS = {a['max_ppd_vs_repl_qps']}")
            if a.get("pd_beats_repl_qps"):
                print(f"  PD beats Replication starting at QPS = {a['pd_beats_repl_qps']}")
            if a.get("max_pd_vs_repl_qps"):
                print(f"  Max PD advantage vs Replication: {a.get('max_pd_vs_repl_advantage_pct', 0):.1f}% at QPS = {a['max_pd_vs_repl_qps']}")
        else:
            # 2-mode analysis
            if a.get("crossover_qps"):
                print(f"  Crossover point: QPS = {a['crossover_qps']} (PD starts winning)")
            else:
                print(f"  No crossover: PPD wins at all tested QPS levels")
            if a.get("max_ppd_advantage_qps"):
                print(f"  Max PPD advantage: {a.get('max_ppd_advantage_pct', 0):.1f}% at QPS = {a['max_ppd_advantage_qps']}")

    # Save analysis
    analysis_file = output_dir / "qps_crossover_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
