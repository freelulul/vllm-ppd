#!/usr/bin/env python3
"""
Analyze benchmark results to identify anomalies and their causes.

Run: python src/analyze_benchmark.py results/benchmark_20251224_175358.json
"""

import json
import sys
import statistics
from pathlib import Path


def detect_bimodal(values: list[float], threshold: float = 0.4) -> tuple[bool, str]:
    """Detect if values show bimodal distribution (suggesting cache hit/miss pattern)."""
    if len(values) < 3:
        return False, ""

    sorted_vals = sorted(values)
    mean = statistics.mean(values)

    # Check if there's a large gap in the middle
    gaps = []
    for i in range(len(sorted_vals) - 1):
        gap = sorted_vals[i + 1] - sorted_vals[i]
        relative_gap = gap / mean
        gaps.append((i, gap, relative_gap))

    max_gap = max(gaps, key=lambda x: x[2])
    if max_gap[2] > threshold:
        lower = sorted_vals[:max_gap[0] + 1]
        upper = sorted_vals[max_gap[0] + 1:]
        return True, f"Gap at {max_gap[1]:.0f}ms ({max_gap[2]*100:.0f}% of mean). Lower: {[f'{v:.0f}' for v in lower]}, Upper: {[f'{v:.0f}' for v in upper]}"

    return False, ""


def detect_outliers(values: list[float], iqr_multiplier: float = 1.5) -> list[tuple[int, float, str]]:
    """Detect outliers using IQR method."""
    if len(values) < 4:
        return []

    sorted_vals = sorted(values)
    q1 = sorted_vals[len(sorted_vals) // 4]
    q3 = sorted_vals[3 * len(sorted_vals) // 4]
    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    outliers = []
    for i, v in enumerate(values):
        if v < lower_bound:
            outliers.append((i, v, "LOW"))
        elif v > upper_bound:
            outliers.append((i, v, "HIGH"))
        elif v == 0.0:
            outliers.append((i, v, "ZERO (request failure)"))

    return outliers


def analyze_config(mr: dict) -> dict:
    """Analyze a single config's results."""
    analysis = {
        "config_name": mr["config_name"],
        "pd_times": mr["pd_times_ms"],
        "ppd_times": mr["ppd_times_ms"],
        "issues": [],
        "severity": "OK",
    }

    pd_times = mr["pd_times_ms"]
    ppd_times = mr["ppd_times_ms"]

    # Check for zeros (request failures)
    pd_zeros = [i for i, v in enumerate(pd_times) if v == 0.0]
    ppd_zeros = [i for i, v in enumerate(ppd_times) if v == 0.0]
    if pd_zeros or ppd_zeros:
        analysis["issues"].append(f"REQUEST FAILURE: PD zeros at {pd_zeros}, PPD zeros at {ppd_zeros}")
        analysis["severity"] = "CRITICAL"

    # Check CV (coefficient of variation)
    pd_cv = (statistics.stdev(pd_times) / statistics.mean(pd_times) * 100) if len(pd_times) > 1 and statistics.mean(pd_times) > 0 else 0
    ppd_cv = (statistics.stdev(ppd_times) / statistics.mean(ppd_times) * 100) if len(ppd_times) > 1 and statistics.mean(ppd_times) > 0 else 0

    analysis["pd_cv"] = pd_cv
    analysis["ppd_cv"] = ppd_cv

    if pd_cv > 30 or ppd_cv > 30:
        analysis["issues"].append(f"HIGH VARIANCE: PD CV={pd_cv:.1f}%, PPD CV={ppd_cv:.1f}%")
        if analysis["severity"] != "CRITICAL":
            analysis["severity"] = "HIGH"

    # Check for bimodal distribution
    pd_bimodal, pd_bimodal_info = detect_bimodal(pd_times)
    ppd_bimodal, ppd_bimodal_info = detect_bimodal(ppd_times)

    if pd_bimodal:
        analysis["issues"].append(f"PD BIMODAL: {pd_bimodal_info}")
        if analysis["severity"] == "OK":
            analysis["severity"] = "MEDIUM"
    if ppd_bimodal:
        analysis["issues"].append(f"PPD BIMODAL: {ppd_bimodal_info}")
        if analysis["severity"] == "OK":
            analysis["severity"] = "MEDIUM"

    # Check for outliers
    pd_outliers = detect_outliers(pd_times)
    ppd_outliers = detect_outliers(ppd_times)

    if pd_outliers:
        analysis["issues"].append(f"PD OUTLIERS: {pd_outliers}")
    if ppd_outliers:
        analysis["issues"].append(f"PPD OUTLIERS: {ppd_outliers}")

    # Check if trimmed mean significantly differs from mean
    def trimmed_mean(vals):
        if len(vals) <= 2:
            return statistics.mean(vals)
        return statistics.mean(sorted(vals)[1:-1])

    pd_trim_diff = abs(mr["pd_trimmed_mean_ms"] - mr["pd_mean_ms"]) / mr["pd_mean_ms"] * 100 if mr["pd_mean_ms"] > 0 else 0
    ppd_trim_diff = abs(mr["ppd_trimmed_mean_ms"] - mr["ppd_mean_ms"]) / mr["ppd_mean_ms"] * 100 if mr["ppd_mean_ms"] > 0 else 0

    if pd_trim_diff > 15 or ppd_trim_diff > 15:
        analysis["issues"].append(f"TRIM DIFF: PD trim differs by {pd_trim_diff:.1f}%, PPD by {ppd_trim_diff:.1f}%")

    return analysis


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_benchmark.py <benchmark_json_file>")
        print("Example: python analyze_benchmark.py results/benchmark_20251224_175358.json")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    with open(filepath) as f:
        data = json.load(f)

    print("=" * 80)
    print(f"Benchmark Analysis: {filepath.name}")
    print("=" * 80)

    multi_run = data.get("multi_run_results", [])
    if not multi_run:
        print("No multi-run results found in file.")
        sys.exit(1)

    print(f"\nAnalyzing {len(multi_run)} configurations...")
    print()

    # Analyze each config
    critical = []
    high = []
    medium = []
    ok = []

    for mr in multi_run:
        analysis = analyze_config(mr)
        if analysis["severity"] == "CRITICAL":
            critical.append(analysis)
        elif analysis["severity"] == "HIGH":
            high.append(analysis)
        elif analysis["severity"] == "MEDIUM":
            medium.append(analysis)
        else:
            ok.append(analysis)

    # Report
    if critical:
        print("=" * 80)
        print("CRITICAL ISSUES (Request Failures)")
        print("=" * 80)
        for a in critical:
            print(f"\n{a['config_name']}:")
            print(f"  PD times:  {[f'{t:.0f}' for t in a['pd_times']]}")
            print(f"  PPD times: {[f'{t:.0f}' for t in a['ppd_times']]}")
            for issue in a["issues"]:
                print(f"  - {issue}")

    if high:
        print("\n" + "=" * 80)
        print("HIGH VARIANCE ISSUES (CV > 30%)")
        print("=" * 80)
        for a in high:
            print(f"\n{a['config_name']} (PD CV={a['pd_cv']:.1f}%, PPD CV={a['ppd_cv']:.1f}%):")
            print(f"  PD times:  {[f'{t:.0f}' for t in a['pd_times']]}")
            print(f"  PPD times: {[f'{t:.0f}' for t in a['ppd_times']]}")
            for issue in a["issues"]:
                print(f"  - {issue}")

    if medium:
        print("\n" + "=" * 80)
        print("MEDIUM ISSUES (Bimodal/Outliers)")
        print("=" * 80)
        for a in medium:
            print(f"\n{a['config_name']}:")
            print(f"  PD times:  {[f'{t:.0f}' for t in a['pd_times']]}")
            print(f"  PPD times: {[f'{t:.0f}' for t in a['ppd_times']]}")
            for issue in a["issues"]:
                print(f"  - {issue}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total configs: {len(multi_run)}")
    print(f"  CRITICAL:      {len(critical)} configs")
    print(f"  HIGH:          {len(high)} configs")
    print(f"  MEDIUM:        {len(medium)} configs")
    print(f"  OK:            {len(ok)} configs")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if critical:
        print("\n1. FIX REQUEST FAILURES:")
        print("   - Check server logs for errors during benchmark")
        print("   - Increase timeout for large output configs")
        print("   - Add error handling to skip failed runs")

    if high or medium:
        print("\n2. REDUCE VARIANCE FROM PREFIX CACHING:")
        print("   - Run: python src/test_cache_hypothesis.py")
        print("   - If confirmed, add unique run IDs to prompts")
        print("   - Add longer delays between runs (5-10s)")

    print("\n3. FOR SPECIFIC CONFIG DEBUGGING:")
    problematic = critical + high[:3]
    if problematic:
        config_name = problematic[0]["config_name"]
        print(f"   Test single config with extended runs:")
        print(f"   python src/comprehensive_benchmark.py --config-name {config_name} --runs 10 --warmup 2")

    print()


if __name__ == "__main__":
    main()
