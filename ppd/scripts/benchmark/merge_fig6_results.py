#!/usr/bin/env python3
"""
Merge Fig6 Benchmark Results and Generate Report

Reads individual panel results from results/fig6/ and generates:
1. merged_results.json - Combined JSON file
2. fig6_report_TIMESTAMP.md - Formatted markdown report

Usage:
    python scripts/benchmark/merge_fig6_results.py
    python scripts/benchmark/merge_fig6_results.py --output-dir results/fig6
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import PANEL_CONFIGS for panel metadata
from scripts.tests.fig6_benchmark import PANEL_CONFIGS, SLO_THRESHOLDS


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all JSON result files from directory."""
    all_results = {}

    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('panel_'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    panel = data.get('panel', filename.split('_')[1].upper())
                    if panel not in all_results:
                        all_results[panel] = []
                    all_results[panel].append(data)
                    print(f"  Loaded: {filename} (Panel {panel})")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")

    return all_results


def compute_panel_stats(panel_data: List[Dict]) -> Dict[str, Any]:
    """Compute statistics for a panel across all data points and modes."""
    stats = {
        "data_points": [],
        "modes": ["pd", "ppd", "replica", "optimizer"],
        "by_mode": defaultdict(lambda: {
            "total_turns": 0,
            "successful_turns": 0,
            "ttft_values": [],
            "tpot_values": [],
            "e2e_values": [],
            "slo_met": {"ttft": 0, "tpot": 0, "e2e": 0},
        })
    }

    for dp_result in panel_data:
        x_value = dp_result.get("x_value")
        stats["data_points"].append(x_value)

        for mode, mode_result in dp_result.get("results", {}).items():
            ms = stats["by_mode"][mode]

            for conv in mode_result.get("conversations", []):
                for turn in conv.get("turns", []):
                    if turn.get("success"):
                        ms["total_turns"] += 1
                        ms["successful_turns"] += 1

                        ttft = turn.get("ttft_ms", 0)
                        tpot = turn.get("tpot_ms", 0)
                        e2e = turn.get("e2e_ms", 0)

                        ms["ttft_values"].append(ttft)
                        ms["tpot_values"].append(tpot)
                        ms["e2e_values"].append(e2e)

                        if ttft <= SLO_THRESHOLDS["ttft_ms"]:
                            ms["slo_met"]["ttft"] += 1
                        if tpot <= SLO_THRESHOLDS["tpot_ms"]:
                            ms["slo_met"]["tpot"] += 1
                        if e2e <= SLO_THRESHOLDS["e2e_ms"]:
                            ms["slo_met"]["e2e"] += 1
                    else:
                        ms["total_turns"] += 1

    return stats


def format_table(headers: List[str], rows: List[List[str]], col_widths: List[int] = None) -> str:
    """Format data as ASCII table."""
    if not col_widths:
        col_widths = [max(len(str(h)), max(len(str(row[i])) for row in rows) if rows else 0)
                      for i, h in enumerate(headers)]

    # Header
    header_line = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
    separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    # Rows
    row_lines = []
    for row in rows:
        row_line = "| " + " | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(headers))) + " |"
        row_lines.append(row_line)

    return "\n".join([header_line, separator] + row_lines)


def generate_report(all_results: Dict[str, Any], output_path: str):
    """Generate markdown report from results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Fig6 Benchmark Complete Report\n")
    lines.append(f"**Generated:** {timestamp}\n")
    lines.append("")

    # Configuration summary
    lines.append("## 1. Configuration Summary\n")
    lines.append("### 1.1 SLO Thresholds")
    lines.append(f"- TTFT: ≤ {SLO_THRESHOLDS['ttft_ms']}ms")
    lines.append(f"- TPOT: ≤ {SLO_THRESHOLDS['tpot_ms']}ms")
    lines.append(f"- E2E: ≤ {SLO_THRESHOLDS['e2e_ms']}ms")
    lines.append("")

    lines.append("### 1.2 Panel Configurations")
    for panel_id in sorted(PANEL_CONFIGS.keys()):
        cfg = PANEL_CONFIGS[panel_id]
        lines.append(f"- **Panel {panel_id}**: {cfg['name']}")
        lines.append(f"  - X-axis: {cfg['x_axis']}")
        lines.append(f"  - Values: {cfg['x_values']}")
        fixed = cfg.get('fixed', {})
        if 'qps' in fixed:
            lines.append(f"  - QPS: {fixed['qps']}")
    lines.append("")

    # Panel results
    lines.append("## 2. Panel Results\n")

    for panel_id in sorted(all_results.keys()):
        panel_data = all_results[panel_id]
        cfg = PANEL_CONFIGS.get(panel_id, {})

        lines.append(f"### Panel {panel_id}: {cfg.get('name', 'Unknown')}\n")

        # Process each data point
        for dp in panel_data:
            x_value = dp.get("x_value")
            lines.append(f"#### X = {x_value}\n")

            # Create comparison table
            headers = ["Mode", "Turns", "Success", "TTFT avg", "TTFT p50", "TPOT avg", "E2E avg", "SLO %"]
            rows = []

            best_slo = 0
            best_mode = ""

            for mode in ["pd", "ppd", "replica", "optimizer"]:
                mode_result = dp.get("results", {}).get(mode, {})

                # Compute stats
                total_turns = 0
                successful_turns = 0
                ttft_values = []
                tpot_values = []
                e2e_values = []
                slo_met = 0

                for conv in mode_result.get("conversations", []):
                    obj = conv.get("objective", "mixed")
                    for turn in conv.get("turns", []):
                        total_turns += 1
                        if turn.get("success"):
                            successful_turns += 1
                            ttft = turn.get("ttft_ms", 0)
                            tpot = turn.get("tpot_ms", 0)
                            e2e = turn.get("e2e_ms", 0)

                            ttft_values.append(ttft)
                            tpot_values.append(tpot)
                            e2e_values.append(e2e)

                            # Check SLO based on objective
                            if obj == "ttft" and ttft <= SLO_THRESHOLDS["ttft_ms"]:
                                slo_met += 1
                            elif obj == "tpot" and tpot <= SLO_THRESHOLDS["tpot_ms"]:
                                slo_met += 1
                            elif obj == "e2e" and e2e <= SLO_THRESHOLDS["e2e_ms"]:
                                slo_met += 1
                            elif obj == "mixed":
                                # For mixed, count if ALL thresholds met
                                if (ttft <= SLO_THRESHOLDS["ttft_ms"] and
                                    tpot <= SLO_THRESHOLDS["tpot_ms"] and
                                    e2e <= SLO_THRESHOLDS["e2e_ms"]):
                                    slo_met += 1

                if successful_turns > 0:
                    avg_ttft = sum(ttft_values) / len(ttft_values)
                    p50_ttft = sorted(ttft_values)[len(ttft_values) // 2]
                    avg_tpot = sum(tpot_values) / len(tpot_values)
                    avg_e2e = sum(e2e_values) / len(e2e_values)
                    slo_pct = (slo_met / successful_turns) * 100
                else:
                    avg_ttft = p50_ttft = avg_tpot = avg_e2e = 0
                    slo_pct = 0

                if slo_pct > best_slo:
                    best_slo = slo_pct
                    best_mode = mode

                rows.append([
                    mode.upper(),
                    total_turns,
                    successful_turns,
                    f"{avg_ttft:.1f}",
                    f"{p50_ttft:.1f}",
                    f"{avg_tpot:.2f}",
                    f"{avg_e2e:.1f}",
                    f"{slo_pct:.1f}%"
                ])

            # Mark best mode
            for row in rows:
                if row[0].lower() == best_mode and best_slo > 0:
                    row[7] = row[7] + " ✓"

            lines.append(format_table(headers, rows))
            lines.append("")

            # Show routing stats for optimizer
            opt_result = dp.get("results", {}).get("optimizer", {})
            routing = opt_result.get("routing_stats", {})
            if routing:
                routing_str = ", ".join(f"{k}: {v}" for k, v in routing.items())
                lines.append(f"**Optimizer Routing:** {routing_str}")
                lines.append("")

        lines.append("---\n")

    # Summary section
    lines.append("## 3. Summary\n")
    lines.append("### Winner by Panel\n")

    summary_headers = ["Panel", "Name", "Best Mode", "Condition"]
    summary_rows = []

    for panel_id in sorted(all_results.keys()):
        cfg = PANEL_CONFIGS.get(panel_id, {})
        # Simple heuristic - would need actual winner computation
        summary_rows.append([
            panel_id,
            cfg.get('name', 'Unknown')[:30],
            "TBD",
            "See detailed results"
        ])

    lines.append(format_table(summary_headers, summary_rows))
    lines.append("")

    # Write report
    report_content = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report_content)

    print(f"\nReport generated: {output_path}")
    return report_content


def main():
    parser = argparse.ArgumentParser(description="Merge Fig6 benchmark results")
    parser.add_argument("--results-dir", default="results/fig6",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", default="docs",
                        help="Output directory for merged results and report")
    args = parser.parse_args()

    print("=" * 60)
    print("Fig6 Results Merger")
    print("=" * 60)

    # Load results
    print(f"\nLoading results from: {args.results_dir}")
    all_results = load_results(args.results_dir)

    if not all_results:
        print("ERROR: No results found!")
        return 1

    print(f"\nFound results for panels: {sorted(all_results.keys())}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save merged JSON
    merged_json_path = os.path.join(args.results_dir, "merged_results.json")
    with open(merged_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMerged JSON: {merged_json_path}")

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.output_dir, f"fig6_report_{timestamp}.md")
    generate_report(all_results, report_path)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
