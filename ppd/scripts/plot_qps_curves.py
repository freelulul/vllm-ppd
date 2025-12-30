#!/usr/bin/env python3
"""
Generate QPS vs Latency curves from benchmark results.

This script creates the standard figures for system papers:
- Figure 1: QPS vs P99 TTFT for each workload
- Figure 2: QPS vs Avg E2E Latency for each workload
- Figure 3: Combined comparison showing crossover points

Usage:
    python scripts/plot_qps_curves.py results/qps_benchmark_*.json
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


def extract_data(results: dict) -> dict:
    """Extract data organized by workload and mode."""
    data = {}

    for r in results["results"]:
        wk = r["workload"]
        mode = r["mode"]
        qps = r["target_qps"]

        if wk not in data:
            data[wk] = {"ppd": {}, "pd": {}}

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

    return data


def plot_qps_curves(data: dict, output_dir: Path):
    """Generate QPS vs Latency curves."""
    workloads = list(data.keys())
    n_workloads = len(workloads)

    if n_workloads == 0:
        print("No data to plot")
        return

    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {"ppd": "#2ecc71", "pd": "#3498db"}
    markers = {"ppd": "o", "pd": "s"}

    # ================================================================
    # Figure 1: P99 TTFT vs QPS (2x2 subplots)
    # ================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    axes1 = axes1.flatten()

    for idx, wk in enumerate(workloads[:4]):
        ax = axes1[idx]
        wk_data = data[wk]

        for mode in ["ppd", "pd"]:
            if mode not in wk_data:
                continue

            qps_vals = sorted(wk_data[mode].keys())
            p99_vals = [wk_data[mode][q]["p99_ttft"] for q in qps_vals]

            ax.plot(qps_vals, p99_vals,
                    marker=markers[mode], color=colors[mode],
                    linewidth=2, markersize=8, label=mode.upper())

        ax.set_xlabel("Request Rate (QPS)", fontsize=12)
        ax.set_ylabel("P99 TTFT (ms)", fontsize=12)
        ax.set_title(wk.replace("_", " "), fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Log scale for y-axis if range is large
        y_vals = [wk_data[m][q]["p99_ttft"] for m in wk_data for q in wk_data[m]]
        if max(y_vals) / (min(y_vals) + 0.1) > 10:
            ax.set_yscale('log')

    # Hide unused subplots
    for idx in range(len(workloads), 4):
        axes1[idx].set_visible(False)

    fig1.suptitle("QPS vs P99 TTFT: PD vs PPD Mode Comparison\n(Lower is better)", fontsize=16)
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

        for mode in ["ppd", "pd"]:
            if mode not in wk_data:
                continue

            qps_vals = sorted(wk_data[mode].keys())
            e2e_vals = [wk_data[mode][q]["avg_e2e"] for q in qps_vals]

            ax.plot(qps_vals, e2e_vals,
                    marker=markers[mode], color=colors[mode],
                    linewidth=2, markersize=8, label=mode.upper())

        ax.set_xlabel("Request Rate (QPS)", fontsize=12)
        ax.set_ylabel("Avg E2E Latency (ms)", fontsize=12)
        ax.set_title(wk.replace("_", " "), fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    for idx in range(len(workloads), 4):
        axes2[idx].set_visible(False)

    fig2.suptitle("QPS vs Average End-to-End Latency: PD vs PPD Mode\n(Lower is better)", fontsize=16)
    plt.tight_layout()
    fig2.savefig(output_dir / "qps_avg_e2e.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {output_dir / 'qps_avg_e2e.png'}")

    # ================================================================
    # Figure 3: PPD/PD Latency Ratio vs QPS (shows crossover point)
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 8))

    colors_wk = plt.cm.viridis(np.linspace(0.2, 0.8, len(workloads)))

    for idx, wk in enumerate(workloads):
        wk_data = data[wk]

        if "ppd" not in wk_data or "pd" not in wk_data:
            continue

        common_qps = sorted(set(wk_data["ppd"].keys()) & set(wk_data["pd"].keys()))
        if not common_qps:
            continue

        ratios = []
        for q in common_qps:
            ppd_ttft = wk_data["ppd"][q]["p99_ttft"]
            pd_ttft = wk_data["pd"][q]["p99_ttft"]
            if pd_ttft > 0:
                ratios.append(ppd_ttft / pd_ttft)
            else:
                ratios.append(1.0)

        ax3.plot(common_qps, ratios,
                marker='o', color=colors_wk[idx],
                linewidth=2, markersize=8, label=wk.replace("_", " "))

    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Crossover (PPD = PD)')
    ax3.set_xlabel("Request Rate (QPS)", fontsize=12)
    ax3.set_ylabel("P99 TTFT Ratio (PPD / PD)", fontsize=12)
    ax3.set_title("PPD vs PD Performance Ratio by QPS\n(Below 1.0 = PPD wins, Above 1.0 = PD wins)", fontsize=14)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Add shaded regions
    ax3.axhspan(0, 1.0, alpha=0.1, color='green', label='_nolegend_')
    ax3.axhspan(1.0, ax3.get_ylim()[1], alpha=0.1, color='blue', label='_nolegend_')

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

        for mode in ["ppd", "pd"]:
            if mode not in wk_data:
                continue

            qps_vals = sorted(wk_data[mode].keys())
            success_vals = [wk_data[mode][q]["success_rate"] * 100 for q in qps_vals]

            ax.plot(qps_vals, success_vals,
                    marker=markers[mode], color=colors[mode],
                    linewidth=2, markersize=8, label=mode.upper())

        ax.set_xlabel("Request Rate (QPS)", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(wk.replace("_", " "), fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    for idx in range(len(workloads), 4):
        axes4[idx].set_visible(False)

    fig4.suptitle("QPS vs Success Rate: System Stability Under Load\n(Higher is better)", fontsize=16)
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

        for mode in ["ppd", "pd"]:
            if mode not in wk_data:
                continue

            qps_vals = sorted(wk_data[mode].keys())
            tps_vals = [wk_data[mode][q]["avg_throughput_tps"] for q in qps_vals]

            ax.plot(qps_vals, tps_vals,
                    marker=markers[mode], color=colors[mode],
                    linewidth=2, markersize=8, label=mode.upper())

        ax.set_xlabel("Request Rate (QPS)", fontsize=12)
        ax.set_ylabel("Avg Throughput (tokens/sec)", fontsize=12)
        ax.set_title(wk.replace("_", " "), fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    for idx in range(len(workloads), 4):
        axes5[idx].set_visible(False)

    fig5.suptitle("QPS vs Throughput: Token Generation Speed\n(Higher is better)", fontsize=16)
    plt.tight_layout()
    fig5.savefig(output_dir / "qps_throughput.png", dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print(f"Saved: {output_dir / 'qps_throughput.png'}")

    # ================================================================
    # Figure 6: Aggregate Throughput vs QPS (total tokens generated)
    # ================================================================
    fig6, ax6 = plt.subplots(figsize=(12, 8))

    bar_width = 0.35
    x_positions = np.arange(len(workloads))

    # Get throughput at highest QPS for each workload
    max_qps = max(qps_vals) if qps_vals else 8.0

    ppd_tps = []
    pd_tps = []
    for wk in workloads:
        wk_data = data[wk]
        ppd_val = wk_data.get("ppd", {}).get(max_qps, {}).get("avg_throughput_tps", 0)
        pd_val = wk_data.get("pd", {}).get(max_qps, {}).get("avg_throughput_tps", 0)
        ppd_tps.append(ppd_val)
        pd_tps.append(pd_val)

    bars1 = ax6.bar(x_positions - bar_width/2, ppd_tps, bar_width, label='PPD', color='#2ecc71')
    bars2 = ax6.bar(x_positions + bar_width/2, pd_tps, bar_width, label='PD', color='#3498db')

    ax6.set_xlabel("Workload", fontsize=12)
    ax6.set_ylabel(f"Avg Throughput @ QPS={max_qps} (tokens/sec)", fontsize=12)
    ax6.set_title(f"Throughput Comparison at Peak Load (QPS={max_qps})\n(Higher is better)", fontsize=14)
    ax6.set_xticks(x_positions)
    ax6.set_xticklabels([wk.replace("_", "\n") for wk in workloads], fontsize=10)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars1, ppd_tps):
        if val > 0:
            ax6.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, pd_tps):
        if val > 0:
            ax6.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig6.savefig(output_dir / "qps_throughput_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print(f"Saved: {output_dir / 'qps_throughput_comparison.png'}")


def generate_crossover_analysis(data: dict) -> dict:
    """Analyze crossover points where PD becomes better than PPD."""
    analysis = {}

    for wk, wk_data in data.items():
        if "ppd" not in wk_data or "pd" not in wk_data:
            continue

        common_qps = sorted(set(wk_data["ppd"].keys()) & set(wk_data["pd"].keys()))

        crossover_qps = None
        for q in common_qps:
            ppd_ttft = wk_data["ppd"][q]["p99_ttft"]
            pd_ttft = wk_data["pd"][q]["p99_ttft"]
            if ppd_ttft > pd_ttft:
                crossover_qps = q
                break

        # Find the QPS where PPD advantage is maximum
        max_advantage_qps = None
        max_advantage = 0
        for q in common_qps:
            ppd_ttft = wk_data["ppd"][q]["p99_ttft"]
            pd_ttft = wk_data["pd"][q]["p99_ttft"]
            if pd_ttft > 0:
                advantage = (pd_ttft - ppd_ttft) / pd_ttft
                if advantage > max_advantage:
                    max_advantage = advantage
                    max_advantage_qps = q

        analysis[wk] = {
            "crossover_qps": crossover_qps,
            "max_ppd_advantage_qps": max_advantage_qps,
            "max_ppd_advantage_pct": max_advantage * 100 if max_advantage else 0,
        }

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
    data = extract_data(results)
    plot_qps_curves(data, output_dir)

    # Generate crossover analysis
    analysis = generate_crossover_analysis(data)

    print("\n" + "=" * 60)
    print("CROSSOVER ANALYSIS")
    print("=" * 60)
    for wk, a in analysis.items():
        print(f"\n{wk}:")
        if a["crossover_qps"]:
            print(f"  Crossover point: QPS = {a['crossover_qps']} (PD starts winning)")
        else:
            print(f"  No crossover: PPD wins at all tested QPS levels")
        if a["max_ppd_advantage_qps"]:
            print(f"  Max PPD advantage: {a['max_ppd_advantage_pct']:.1f}% at QPS = {a['max_ppd_advantage_qps']}")

    # Save analysis
    analysis_file = output_dir / "qps_crossover_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
