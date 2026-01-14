#!/usr/bin/env python3
"""
Build Training Data for Optimizer Models

Reads benchmark results and creates training data where:
- Features: input_length, output_length, turn_number, has_cache, queue_depth, optimization_objective
- Label: best_mode (pd, ppd, replica)

Key insight: Same input features + different optimization objective = different best mode
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class OptimizationObjective(Enum):
    """Optimization objectives that affect mode selection"""
    TTFT = "ttft"           # Time To First Token (latency-sensitive)
    TPOT = "tpot"           # Time Per Output Token (generation speed)
    THROUGHPUT = "throughput"  # Tokens per second (batch efficiency)
    E2E = "e2e"             # End-to-end latency


@dataclass
class WorkloadFeatures:
    """Features extracted from a workload configuration"""
    workload_name: str
    t1_input: int
    t1_output: int
    t2_input: int
    t2_output: int
    num_turns: int
    target_qps: int

    @property
    def input_output_ratio(self) -> float:
        """Ratio of input to output tokens (high = Big-Paste scenario)"""
        return self.t1_input / max(self.t1_output, 1)

    @property
    def is_long_output(self) -> bool:
        """Whether this workload generates long outputs"""
        return self.t1_output > 512

    @property
    def is_big_paste(self) -> bool:
        """Big-Paste: long input, short output"""
        return self.t1_input > 1000 and self.t1_output < 256


@dataclass
class ModePerformance:
    """Performance metrics for a specific mode on a workload"""
    mode: str
    avg_ttft: float
    p99_ttft: float
    avg_tpot: float
    p99_tpot: float
    avg_throughput: float
    avg_e2e: float


def load_merged_results(results_path: str) -> Dict:
    """Load merged benchmark results"""
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_mode_performance(result: Dict, turn: int = 2) -> ModePerformance:
    """Extract performance metrics for a specific mode and turn"""
    turn_key = f"turn{turn}_metrics"
    metrics = result.get(turn_key, {})

    return ModePerformance(
        mode=result['mode'],
        avg_ttft=metrics.get('avg_ttft', float('inf')),
        p99_ttft=metrics.get('p99_ttft', float('inf')),
        avg_tpot=metrics.get('avg_tpot', float('inf')),
        p99_tpot=metrics.get('p99_tpot', float('inf')),
        avg_throughput=metrics.get('avg_throughput_tps', 0),
        avg_e2e=metrics.get('avg_e2e', float('inf'))
    )


def determine_best_mode(
    performances: List[ModePerformance],
    objective: OptimizationObjective
) -> str:
    """
    Determine the best mode for a given optimization objective.

    Lower is better for: TTFT, TPOT, E2E
    Higher is better for: THROUGHPUT
    """
    if not performances:
        return "ppd"  # default

    if objective == OptimizationObjective.TTFT:
        # Minimize TTFT
        best = min(performances, key=lambda p: p.avg_ttft)
    elif objective == OptimizationObjective.TPOT:
        # Minimize TPOT
        best = min(performances, key=lambda p: p.avg_tpot)
    elif objective == OptimizationObjective.THROUGHPUT:
        # Maximize throughput
        best = max(performances, key=lambda p: p.avg_throughput)
    elif objective == OptimizationObjective.E2E:
        # Minimize E2E latency
        best = min(performances, key=lambda p: p.avg_e2e)
    else:
        best = performances[0]

    return best.mode


def build_training_data(results_path: str) -> pd.DataFrame:
    """
    Build training dataset from benchmark results.

    Each row represents a (workload, qps, objective) combination.
    The label is the best mode for that specific objective.
    """
    data = load_merged_results(results_path)
    results = data['results']

    # Group results by (workload, qps)
    grouped = {}
    for r in results:
        key = (r['workload'], r['target_qps'])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r['mode']] = r

    training_rows = []

    for (workload, qps), mode_results in grouped.items():
        # Skip if we don't have all three modes
        if len(mode_results) < 2:
            continue

        # Get a reference result for workload features
        ref = list(mode_results.values())[0]

        features = WorkloadFeatures(
            workload_name=workload,
            t1_input=ref.get('t1_input', 0),
            t1_output=ref.get('t1_output', 0),
            t2_input=ref.get('t2_input', 0),
            t2_output=ref.get('t2_output', 0),
            num_turns=ref.get('num_turns', 2),
            target_qps=qps
        )

        # Extract Turn 2 performance (where cache matters)
        performances = []
        for mode, result in mode_results.items():
            perf = extract_mode_performance(result, turn=2)
            performances.append(perf)

        # Create one training sample for each optimization objective
        for objective in OptimizationObjective:
            best_mode = determine_best_mode(performances, objective)

            # Get the actual performance values for context
            perf_dict = {p.mode: p for p in performances}

            row = {
                # Input features
                'workload': workload,
                'target_qps': qps,
                't1_input': features.t1_input,
                't1_output': features.t1_output,
                't2_input': features.t2_input,
                't2_output': features.t2_output,
                'num_turns': features.num_turns,
                'input_output_ratio': features.input_output_ratio,
                'is_long_output': int(features.is_long_output),
                'is_big_paste': int(features.is_big_paste),

                # Optimization objective (KEY FEATURE)
                'objective': objective.value,
                'objective_ttft': int(objective == OptimizationObjective.TTFT),
                'objective_tpot': int(objective == OptimizationObjective.TPOT),
                'objective_throughput': int(objective == OptimizationObjective.THROUGHPUT),
                'objective_e2e': int(objective == OptimizationObjective.E2E),

                # Label
                'best_mode': best_mode,

                # Performance context (for analysis)
                'pd_ttft': perf_dict.get('pd', ModePerformance('pd', float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'))).avg_ttft,
                'ppd_ttft': perf_dict.get('ppd', ModePerformance('ppd', float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'))).avg_ttft,
                'replica_ttft': perf_dict.get('replica', ModePerformance('replica', float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'))).avg_ttft,
                'pd_tpot': perf_dict.get('pd', ModePerformance('pd', float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'))).avg_tpot,
                'ppd_tpot': perf_dict.get('ppd', ModePerformance('ppd', float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'))).avg_tpot,
                'replica_tpot': perf_dict.get('replica', ModePerformance('replica', float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'))).avg_tpot,
                'pd_throughput': perf_dict.get('pd', ModePerformance('pd', float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'))).avg_throughput,
                'ppd_throughput': perf_dict.get('ppd', ModePerformance('ppd', float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'))).avg_throughput,
                'replica_throughput': perf_dict.get('replica', ModePerformance('replica', float('inf'), float('inf'), float('inf'), float('inf'), 0, float('inf'))).avg_throughput,
            }
            training_rows.append(row)

    df = pd.DataFrame(training_rows)
    return df


def analyze_training_data(df: pd.DataFrame) -> Dict:
    """Analyze the training data distribution"""
    analysis = {
        'total_samples': len(df),
        'unique_workloads': df['workload'].nunique(),
        'unique_qps': sorted(df['target_qps'].unique().tolist()),
        'objectives': df['objective'].unique().tolist(),
        'mode_distribution': df['best_mode'].value_counts().to_dict(),
        'mode_by_objective': {}
    }

    for obj in df['objective'].unique():
        obj_df = df[df['objective'] == obj]
        analysis['mode_by_objective'][obj] = obj_df['best_mode'].value_counts().to_dict()

    return analysis


def save_training_data(df: pd.DataFrame, output_dir: str):
    """Save training data and analysis"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    df.to_csv(output_path / 'training_data.csv', index=False)
    df.to_json(output_path / 'training_data.json', orient='records', indent=2)

    # Save analysis
    analysis = analyze_training_data(df)
    with open(output_path / 'data_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Saved {len(df)} training samples to {output_path}")
    print(f"\nData Analysis:")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Unique workloads: {analysis['unique_workloads']}")
    print(f"  QPS levels: {analysis['unique_qps']}")
    print(f"\n  Mode distribution by objective:")
    for obj, dist in analysis['mode_by_objective'].items():
        print(f"    {obj}: {dist}")

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/final/merged_results.json',
                        help='Path to merged benchmark results')
    parser.add_argument('--output', default='optimizer/data',
                        help='Output directory for training data')
    args = parser.parse_args()

    print("Building training data from benchmark results...")
    df = build_training_data(args.results)
    analysis = save_training_data(df, args.output)
