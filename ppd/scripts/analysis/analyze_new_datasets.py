#!/usr/bin/env python3
"""
Analyze input/output length distributions of WildChat-1M and LMSYS-Chat-1M datasets.
Compare with ShareGPT and identify prefill-heavy conversations.

Prefill-heavy criteria:
- input_tokens / output_tokens > 2
- input_tokens > 256
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

DATA_DIR = Path(PROJECT_DIR) / "data"

# Prefill-heavy criteria
MIN_RATIO = 2.0  # input/output ratio
MIN_INPUT_TOKENS = 256


def estimate_tokens(text: str) -> int:
    """Estimate token count (4 chars per token approximation)."""
    return len(text) // 4


def analyze_wildchat(sample_path: Path = None):
    """Analyze WildChat-1M dataset."""
    print("\n" + "="*60)
    print("Analyzing WildChat-1M")
    print("="*60)

    try:
        from datasets import load_dataset

        if sample_path and sample_path.exists():
            print(f"Loading sample from: {sample_path}")
            data = json.load(open(sample_path))
        else:
            print("Loading full dataset (this may take a while)...")
            dataset = load_dataset("allenai/WildChat-1M")
            # Sample for analysis
            data = [dataset['train'][i] for i in range(min(10000, len(dataset['train'])))]

        # Analyze conversations
        stats = analyze_conversations(data, dataset_type='wildchat')
        return stats

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def analyze_lmsys(sample_path: Path = None):
    """Analyze LMSYS-Chat-1M dataset."""
    print("\n" + "="*60)
    print("Analyzing LMSYS-Chat-1M")
    print("="*60)

    try:
        from datasets import load_dataset

        if sample_path and sample_path.exists():
            print(f"Loading sample from: {sample_path}")
            data = json.load(open(sample_path))
        else:
            print("Loading full dataset (this may take a while)...")
            dataset = load_dataset("lmsys/lmsys-chat-1m")
            data = [dataset['train'][i] for i in range(min(10000, len(dataset['train'])))]

        # Analyze conversations
        stats = analyze_conversations(data, dataset_type='lmsys')
        return stats

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def analyze_sharegpt(sample_path: Path = None):
    """Analyze ShareGPT dataset for comparison."""
    print("\n" + "="*60)
    print("Analyzing ShareGPT (baseline)")
    print("="*60)

    try:
        sharegpt_path = DATA_DIR / "ShareGPT_V3_unfiltered_cleaned_split.json"
        if not sharegpt_path.exists():
            print(f"ShareGPT not found at {sharegpt_path}")
            return None

        print(f"Loading from: {sharegpt_path}")
        data = json.load(open(sharegpt_path))[:10000]  # Sample for comparison

        # Convert to common format
        converted = []
        for conv in data:
            messages = []
            for msg in conv.get('conversations', []):
                role = 'user' if msg.get('from') == 'human' else 'assistant'
                messages.append({'role': role, 'content': msg.get('value', '')})
            converted.append({'conversation': messages})

        stats = analyze_conversations(converted, dataset_type='sharegpt')
        return stats

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def analyze_conversations(data: list, dataset_type: str):
    """Analyze conversation list and compute statistics."""
    all_inputs = []
    all_outputs = []
    all_ratios = []

    multi_turn_count = 0
    prefill_heavy_convs = []
    prefill_heavy_turns = 0
    total_turns = 0

    for conv in data:
        # Extract conversation based on dataset type
        if dataset_type == 'wildchat':
            messages = conv.get('conversation', [])
        elif dataset_type == 'lmsys':
            messages = conv.get('conversation', [])
        else:  # sharegpt converted
            messages = conv.get('conversation', [])

        if not messages:
            continue

        # Count turns
        turns = 0
        conv_prefill_heavy = False

        for i, msg in enumerate(messages):
            if msg.get('role') == 'user':
                input_tokens = estimate_tokens(msg.get('content', ''))

                # Find corresponding assistant response
                if i + 1 < len(messages) and messages[i + 1].get('role') == 'assistant':
                    output_tokens = estimate_tokens(messages[i + 1].get('content', ''))

                    if output_tokens > 0:
                        ratio = input_tokens / output_tokens
                        all_inputs.append(input_tokens)
                        all_outputs.append(output_tokens)
                        all_ratios.append(ratio)
                        turns += 1
                        total_turns += 1

                        # Check prefill-heavy
                        if ratio > MIN_RATIO and input_tokens > MIN_INPUT_TOKENS:
                            prefill_heavy_turns += 1
                            conv_prefill_heavy = True

        if turns >= 2:
            multi_turn_count += 1

        if conv_prefill_heavy:
            prefill_heavy_convs.append(conv)

    # Compute statistics
    if not all_inputs:
        print("No valid conversations found!")
        return None

    inputs_arr = np.array(all_inputs)
    outputs_arr = np.array(all_outputs)
    ratios_arr = np.array(all_ratios)

    # Workload classification
    decode_heavy = sum(1 for r in all_ratios if r < 0.25)
    balanced = sum(1 for r in all_ratios if 0.25 <= r <= 2.0)
    prefill_heavy = sum(1 for r in all_ratios if r > 2.0)

    stats = {
        'total_conversations': len(data),
        'multi_turn_conversations': multi_turn_count,
        'multi_turn_rate': multi_turn_count / len(data) * 100,
        'total_turns': total_turns,
        'prefill_heavy_turns': prefill_heavy_turns,
        'prefill_heavy_rate': prefill_heavy_turns / total_turns * 100 if total_turns > 0 else 0,
        'prefill_heavy_convs_count': len(prefill_heavy_convs),
        'input_tokens': {
            'min': int(inputs_arr.min()),
            'p25': int(np.percentile(inputs_arr, 25)),
            'p50': int(np.percentile(inputs_arr, 50)),
            'p75': int(np.percentile(inputs_arr, 75)),
            'p90': int(np.percentile(inputs_arr, 90)),
            'p99': int(np.percentile(inputs_arr, 99)),
            'max': int(inputs_arr.max()),
            'mean': float(inputs_arr.mean()),
        },
        'output_tokens': {
            'min': int(outputs_arr.min()),
            'p25': int(np.percentile(outputs_arr, 25)),
            'p50': int(np.percentile(outputs_arr, 50)),
            'p75': int(np.percentile(outputs_arr, 75)),
            'p90': int(np.percentile(outputs_arr, 90)),
            'p99': int(np.percentile(outputs_arr, 99)),
            'max': int(outputs_arr.max()),
            'mean': float(outputs_arr.mean()),
        },
        'ratio': {
            'p10': float(np.percentile(ratios_arr, 10)),
            'p50': float(np.percentile(ratios_arr, 50)),
            'p90': float(np.percentile(ratios_arr, 90)),
        },
        'workload_distribution': {
            'decode_heavy': decode_heavy,
            'decode_heavy_pct': decode_heavy / total_turns * 100,
            'balanced': balanced,
            'balanced_pct': balanced / total_turns * 100,
            'prefill_heavy': prefill_heavy,
            'prefill_heavy_pct': prefill_heavy / total_turns * 100,
        }
    }

    # Print summary
    print(f"\n总对话数: {stats['total_conversations']}")
    print(f"多轮对话数: {stats['multi_turn_conversations']} ({stats['multi_turn_rate']:.1f}%)")
    print(f"总轮次: {stats['total_turns']}")
    print(f"\n输入tokens分布:")
    print(f"  min={stats['input_tokens']['min']}, p25={stats['input_tokens']['p25']}, "
          f"p50={stats['input_tokens']['p50']}, p75={stats['input_tokens']['p75']}, "
          f"p90={stats['input_tokens']['p90']}, p99={stats['input_tokens']['p99']}, max={stats['input_tokens']['max']}")
    print(f"  mean={stats['input_tokens']['mean']:.1f}")
    print(f"\n输出tokens分布:")
    print(f"  min={stats['output_tokens']['min']}, p25={stats['output_tokens']['p25']}, "
          f"p50={stats['output_tokens']['p50']}, p75={stats['output_tokens']['p75']}, "
          f"p90={stats['output_tokens']['p90']}, p99={stats['output_tokens']['p99']}, max={stats['output_tokens']['max']}")
    print(f"  mean={stats['output_tokens']['mean']:.1f}")
    print(f"\n输入/输出比例:")
    print(f"  p10={stats['ratio']['p10']:.2f}, p50={stats['ratio']['p50']:.2f}, p90={stats['ratio']['p90']:.2f}")
    print(f"\nWorkload分布:")
    print(f"  decode-heavy (ratio<0.25): {stats['workload_distribution']['decode_heavy']} ({stats['workload_distribution']['decode_heavy_pct']:.1f}%)")
    print(f"  balanced (0.25<=ratio<=2): {stats['workload_distribution']['balanced']} ({stats['workload_distribution']['balanced_pct']:.1f}%)")
    print(f"  prefill-heavy (ratio>2): {stats['workload_distribution']['prefill_heavy']} ({stats['workload_distribution']['prefill_heavy_pct']:.1f}%)")
    print(f"\nPrefill-heavy筛选 (ratio>2 AND input>256):")
    print(f"  满足条件的轮次: {stats['prefill_heavy_turns']} ({stats['prefill_heavy_rate']:.1f}%)")
    print(f"  含prefill-heavy轮次的对话数: {stats['prefill_heavy_convs_count']}")

    return stats


def compare_datasets(stats_list: list):
    """Compare statistics across datasets."""
    print("\n" + "="*60)
    print("数据集对比")
    print("="*60)

    headers = ['指标', 'ShareGPT', 'WildChat', 'LMSYS']
    rows = []

    # Multi-turn rate
    rows.append(['多轮对话比例',
                 f"{stats_list[0]['multi_turn_rate']:.1f}%" if stats_list[0] else 'N/A',
                 f"{stats_list[1]['multi_turn_rate']:.1f}%" if stats_list[1] else 'N/A',
                 f"{stats_list[2]['multi_turn_rate']:.1f}%" if stats_list[2] else 'N/A'])

    # Input tokens median
    rows.append(['输入tokens (p50)',
                 f"{stats_list[0]['input_tokens']['p50']}" if stats_list[0] else 'N/A',
                 f"{stats_list[1]['input_tokens']['p50']}" if stats_list[1] else 'N/A',
                 f"{stats_list[2]['input_tokens']['p50']}" if stats_list[2] else 'N/A'])

    # Input tokens p90
    rows.append(['输入tokens (p90)',
                 f"{stats_list[0]['input_tokens']['p90']}" if stats_list[0] else 'N/A',
                 f"{stats_list[1]['input_tokens']['p90']}" if stats_list[1] else 'N/A',
                 f"{stats_list[2]['input_tokens']['p90']}" if stats_list[2] else 'N/A'])

    # Output tokens median
    rows.append(['输出tokens (p50)',
                 f"{stats_list[0]['output_tokens']['p50']}" if stats_list[0] else 'N/A',
                 f"{stats_list[1]['output_tokens']['p50']}" if stats_list[1] else 'N/A',
                 f"{stats_list[2]['output_tokens']['p50']}" if stats_list[2] else 'N/A'])

    # Prefill-heavy rate
    rows.append(['Prefill-heavy比例',
                 f"{stats_list[0]['prefill_heavy_rate']:.1f}%" if stats_list[0] else 'N/A',
                 f"{stats_list[1]['prefill_heavy_rate']:.1f}%" if stats_list[1] else 'N/A',
                 f"{stats_list[2]['prefill_heavy_rate']:.1f}%" if stats_list[2] else 'N/A'])

    # Decode-heavy rate
    rows.append(['Decode-heavy比例',
                 f"{stats_list[0]['workload_distribution']['decode_heavy_pct']:.1f}%" if stats_list[0] else 'N/A',
                 f"{stats_list[1]['workload_distribution']['decode_heavy_pct']:.1f}%" if stats_list[1] else 'N/A',
                 f"{stats_list[2]['workload_distribution']['decode_heavy_pct']:.1f}%" if stats_list[2] else 'N/A'])

    # Print table
    col_widths = [20, 15, 15, 15]
    header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print('-' * len(header_line))
    for row in rows:
        print(' | '.join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def main():
    print("="*60)
    print("新数据集分析 - PPD测试")
    print("="*60)
    print(f"Prefill-heavy标准: ratio > {MIN_RATIO} AND input_tokens > {MIN_INPUT_TOKENS}")

    # Try to load sample files first (faster)
    wildchat_sample = DATA_DIR / "wildchat_sample_1k.json"
    lmsys_sample = DATA_DIR / "lmsys_sample_1k.json"

    # Analyze all datasets
    sharegpt_stats = analyze_sharegpt()
    wildchat_stats = analyze_wildchat(wildchat_sample if wildchat_sample.exists() else None)
    lmsys_stats = analyze_lmsys(lmsys_sample if lmsys_sample.exists() else None)

    # Compare
    if any([sharegpt_stats, wildchat_stats, lmsys_stats]):
        compare_datasets([sharegpt_stats, wildchat_stats, lmsys_stats])

    # Save results
    results = {
        'sharegpt': sharegpt_stats,
        'wildchat': wildchat_stats,
        'lmsys': lmsys_stats,
    }

    output_path = DATA_DIR / "dataset_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
