# vLLM Disaggregated Serving Optimizer (PPD)

Dynamic request router for multi-turn LLM inference that intelligently routes requests across different GPU configurations based on workload characteristics.

---

## Project Overview

### Core Insight

In multi-turn conversations, subsequent user inputs should not always be treated as prefill work:
- **High cache rate + short new input** → should go directly to decode machine (decode-like)
- **Low cache rate + long new input** → should go back to prefill machine (prefill-like)

This project explores the trade-offs between different GPU configurations and routing strategies.

### Machine Type Definitions

| Type | Full Name | Capability | Use Case |
|------|-----------|------------|----------|
| **P** | Prefill-only | Can only do prefill | PD/PPD mode Turn 1 |
| **D** | Decode-only | Can only do decode (receives KV from P) | PD mode |
| **pD** | Prefill-capable Decode | Can do append_prefill and decode | PPD mode (T2+ direct) |
| **PD/R** | Replica/Normal Server | Full model, can do both | Replica mode |

### Key Distinction: D vs pD

Both are `kv_consumer` role in vLLM, but:
- **D (Decode-only)**: Receives KV every turn, cannot process new input locally
- **pD (Prefill-capable Decode)**: Can process new input tokens locally via `append_prefill`

### Execution Modes

| Mode | GPU Config | Turn 1 | Turn 2+ | Trade-off |
|------|-----------|--------|---------|-----------|
| **PD** | P→D | P prefill → KV transfer → D decode | Same path every turn | Isolation ✓, Transfer overhead ✗ |
| **PPD** | P→pD | P prefill → KV transfer → pD decode | pD-direct (prefix cache) | No T2+ transfer ✓, Less isolation ✗ |
| **Replica** | PD/R | Any worker | Same worker (prefix cache) | High throughput ✓, No isolation ✗ |

---

## Project Structure

```
ppd/
├── src/                              # Core source code
│   ├── comprehensive_proxy.py        # Unified proxy for all 17 configs
│   └── config.py                     # Configuration management
│
├── optimizer/                        # Optimizer module (NEW)
│   ├── __init__.py
│   ├── core/                         # Core components
│   │   ├── routing_engine.py         # Workload classification & routing
│   │   ├── evaluate.py               # SLO evaluation framework
│   │   └── oracle_simulation.py      # Theoretical upper bound
│   ├── runners/                      # Execution runners
│   │   ├── run_baselines.py          # Run single baseline config
│   │   ├── run_conv_level.py         # Conversation-level optimizer
│   │   └── run_optimizer.py          # Legacy per-turn optimizer
│   ├── data/
│   │   └── prepare_dataset.py        # Dataset preparation
│   ├── visualization/
│   │   ├── plot_results.py           # General plotting
│   │   └── plot_slo_curves.py        # SLO attainment curves
│   ├── scripts/
│   │   ├── sbatch_full_baselines.sh  # Run all 17 baselines
│   │   └── run_full_validation.sh    # Full validation pipeline
│   └── archive/                      # Old optimizer code
│
├── scripts/
│   ├── server/                       # Server management (17 configs)
│   │   ├── start_<CONFIG>.sh         # Start specific config
│   │   ├── stop.sh                   # Unified stop
│   │   ├── cleanup_all.sh            # Force cleanup
│   │   └── config.sh                 # Shell configuration
│   ├── benchmark/                    # Benchmarking
│   │   ├── comprehensive_benchmark.py
│   │   ├── model_scaling_benchmark.py
│   │   └── turn_scaling_benchmark.py
│   ├── analysis/                     # Analysis & visualization
│   │   └── plot_tradeoff.py
│   └── tests/                        # Test suites
│
├── results/                          # Benchmark results
│   ├── comprehensive/                # 17 configs × 18 workloads × 10 QPS
│   ├── model_scaling/                # 8B, 14B model comparisons
│   ├── turn_scaling/                 # 2, 4, 8, 16 turn experiments
│   └── optimizer/                    # Optimizer validation results
│       ├── baselines/                # All 17 baseline results
│       ├── dataset/                  # Prepared validation dataset
│       └── figures/                  # Generated visualizations
│
├── docs/                             # Documentation
├── logs/                             # Server logs
└── data/                             # Datasets
```

---

## GPU Configuration Space (17 Configurations)

### Pure Modes

| Category | Configurations | Description |
|----------|---------------|-------------|
| **Replica** | 4R | 4 full replicas |
| **Pure PD** | 1P_3D, 2P_2D, 3P_1D | Prefill + Decode only |
| **Pure PPD** | 1P_3pD, 2P_2pD, 3P_1pD | Prefill + Prefill-capable Decode |

### Mixed & Hybrid Modes

| Category | Configurations | Description |
|----------|---------------|-------------|
| **Mixed PD/PPD** | 1P_2D_1pD, 1P_1D_2pD, 2P_1D_1pD | Mix of D and pD |
| **Hybrid (R+PD)** | 1R_1P_2D, 1R_2P_1D, 2R_1P_1D | Replica + PD |
| **Hybrid (R+PPD)** | 1R_1P_2pD, 1R_2P_1pD, 2R_1P_1pD | Replica + PPD |
| **Hybrid (R+Mixed)** | 1R_1P_1D_1pD | Replica + Mixed |

---

## Quick Start

### Environment Setup

```bash
conda activate vllm-ppd
cd /net/projects2/ds3lab/zongzel/vllm/ppd
```

### Starting Servers

```bash
# Start any of the 17 configurations
./scripts/server/start_<CONFIG>.sh

# Examples:
./scripts/server/start_2P_2D.sh      # Pure PD mode
./scripts/server/start_2P_2pD.sh     # Pure PPD mode
./scripts/server/start_1R_1P_2pD.sh  # Hybrid (1 Replica + PPD)
./scripts/server/start_4R.sh         # Pure Replica
```

### Stopping Servers

```bash
./scripts/server/stop.sh
# Or forcefully:
pkill -f "vllm serve" && pkill -f proxy
```

---

## Optimizer Validation (NEW)

### Design: Conversation-Level + Oracle Simulation

The optimizer uses **full 17 configurations** as baselines for fair comparison.

**Key Design Decision**: Route entire conversation to ONE config (preserves cache affinity) instead of per-turn routing (which breaks cache benefits).

### Running Optimizer Validation

```bash
# 1. Prepare dataset (50 multi-turn conversations)
python -m optimizer.data.prepare_dataset \
    --output results/optimizer/dataset/prepared_dataset.json

# 2. Run all 17 baselines (SLURM job, ~6-8 hours)
sbatch optimizer/scripts/sbatch_full_baselines.sh

# 3. Run oracle simulation (theoretical upper bound)
python -m optimizer.core.oracle_simulation \
    --dataset results/optimizer/dataset/prepared_dataset.json \
    --baselines results/optimizer/baselines \
    --output results/optimizer/oracle_results.json

# 4. Run conversation-level optimizer
python -m optimizer.runners.run_conv_level \
    --dataset results/optimizer/dataset/prepared_dataset.json \
    --benchmark results/comprehensive \
    --output results/optimizer/conv_level_results.json

# 5. Evaluate
python -m optimizer.core.evaluate \
    --conv-level results/optimizer/conv_level_results.json \
    --oracle results/optimizer/oracle_results.json \
    --baselines results/optimizer/baselines \
    --output results/optimizer/evaluation.json

# 6. Visualize
python -m optimizer.visualization.plot_slo_curves \
    --conv-level results/optimizer/conv_level_results.json \
    --oracle results/optimizer/oracle_results.json \
    --baselines results/optimizer/baselines \
    --output results/optimizer/figures/slo_attainment_full.png
```

### Expected Results

| Method | SLO Scale=0.5 | 1.0 | 1.5 | 2.0 |
|--------|--------------|-----|-----|-----|
| **Oracle** | ~65% | ~92% | ~98% | ~99% |
| **Conv-Level** | ~55% | ~85% | ~95% | ~98% |
| Best Single Baseline | ~50% | ~78% | ~92% | ~97% |

Key insight: Oracle in T2+ selects PPD configs far more than T1, validating PPD's value in subsequent turns.

---

## PPD Core Testing (NEW)

### Goal

Validate PPD mode's value on real datasets, demonstrating:
1. Turn 2+ TTFT improvement (~65-75%)
2. SLO attainment rate curves (DistServe style)
3. Latency-throughput Pareto frontier
4. Multi-turn stability

**Core thesis**: PPD opens up a new space in the latency-throughput trade-off.

### Datasets

| Dataset | Characteristic | Use Case |
|---------|---------------|----------|
| **ShareGPT** | 74.9% decode-heavy | Typical chat workloads |
| **WildChat** (prefill-heavy filtered) | 30.6% prefill-heavy | Stress test PPD TPOT |

### Full Benchmark (17 configs × 2 datasets)

```bash
# Submit all 8 batches (sequential, 4h each)
./scripts/benchmark/submit_sharegpt_benchmarks.sh

# Or submit specific batches
./scripts/benchmark/submit_sharegpt_benchmarks.sh 1 2 3

# Monitor progress
squeue -u $USER

# After completion, merge results
python scripts/benchmark/merge_sharegpt_results.py
```

**Batch Configuration**:
- Batch 1: 4R, 1P_3D
- Batch 2: 1P_2D_1pD, 1P_1D_2pD
- Batch 3: 1P_3pD, 2P_2D
- Batch 4: 2P_1D_1pD, 2P_2pD
- Batch 5: 3P_1D, 3P_1pD
- Batch 6: 1R_1P_2D, 1R_1P_2pD
- Batch 7: 1R_2P_1D, 1R_2P_1pD
- Batch 8: 2R_1P_1D, 2R_1P_1pD

### Quick Test (Single Config)

```bash
# ShareGPT (decode-heavy)
python scripts/benchmark/sharegpt_benchmark.py \
    --config 2P_2D \
    --num-conversations 100 \
    --qps 2 4 8

# WildChat (prefill-heavy)
python scripts/benchmark/sharegpt_benchmark.py \
    --config 2P_2D \
    --num-conversations 100 \
    --qps 2 4 8 \
    --sharegpt-path data/WildChat_1M.json \
    --output-dir results/wildchat
```

### Dynamic PPD Mode

Enable PPD routing decisions on PD configs based on benchmark data:

```bash
# Start proxy with dynamic PPD mode
python src/comprehensive_proxy.py \
    --config 2P_2D \
    --enable-ppd-mode \
    --ppd-benchmark-path results/comprehensive
```

The PPD Decision Engine (`optimizer/ppd_decision_engine.py`) uses **weighted multi-metric scoring**:
```python
score = w_ttft * TTFT_improvement - w_tpot * TPOT_degradation
use_ppd = score > 0
```

Decision factors:
- **context_length**: small (≤512), large (512-4096), huge (>4096 extrapolated)
- **input_tokens**: New tokens in current turn
- **output_tokens**: Expected output tokens
- **current_qps**: System load

### Generating Analysis Figures

```bash
# ShareGPT analysis
python scripts/analysis/ppd_analysis.py \
    --dataset-dir results/sharegpt \
    --output-dir results/analysis/figures/sharegpt

# WildChat analysis
python scripts/analysis/ppd_analysis.py \
    --dataset-dir results/wildchat \
    --output-dir results/analysis/figures/wildchat
```

**Generated figures (7 total)**:
- `fig1_t2_ttft_comparison.png`: Turn 2+ TTFT box plot
- `fig2_slo_attainment_ttft.png`: Legacy SLO attainment curves
- `fig3_pareto_frontier.png`: Latency vs throughput trade-off
- `fig4_turn_stability.png`: TTFT stability across turns
- `fig5_ppd_improvement.png`: PPD improvement percentage
- `fig6_slo_combined.png`: **3×2 SLO grid** (TTFT/TPOT/E2E × Scale/Rate)
- `fig7_metrics_table.png`: **Full metrics summary table** (p50/p90/p99/avg)

---

## Benchmark System

### Comprehensive Benchmark

```bash
# Run benchmark for specific config
python scripts/benchmark/comprehensive_benchmark.py \
    --config 2P_2pD \
    --workload all \
    --qps 1 2 4 8

# Submit SLURM batch jobs
./scripts/benchmark/submit_comprehensive.sh
```

### Workload Matrix

**T1 Configurations**: small (128→128), large (1024→1024)

**T2 Configurations** (9 types):
| Type | Input→Output | Characteristic |
|------|--------------|----------------|
| tiny | 16→32 | Minimal overhead |
| short_gen | 32→256 | Short question, long answer |
| long_gen | 32→512 | Very long answer |
| big_paste | 512→64 | Prefill-heavy |
| huge_paste | 1024→32 | Extreme prefill-heavy |

**Total**: 2 T1 × 9 T2 = 18 workloads × 10 QPS levels × 17 configs

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token (critical for responsiveness) |
| **TPOT** | Time Per Output Token (affects streaming quality) |
| **E2E** | End-to-End latency (total request time) |
| **SLO Attainment** | % of requests meeting latency target |

### Weight Profiles for Objectives

| Objective | TTFT Weight | TPOT Weight | E2E Weight |
|-----------|------------|------------|-----------|
| ttft | 0.70 | 0.10 | 0.20 |
| tpot | 0.10 | 0.70 | 0.20 |
| e2e | 0.20 | 0.20 | 0.60 |

---

## Port Assignments

| Component | Port |
|-----------|------|
| Comprehensive Proxy | 10001 |
| Replica Proxy (4R only) | 10002 |
| Prefill servers | 8100-8101 |
| Decode servers | 8200-8201 |
| Replica workers | 8300-8600 |

---

## Hardware

- **GPUs**: 4x H100 80GB
- **Model**: Llama-3.1-8B (primary), Llama-3.1-14B (scaling)
- **vLLM Version**: v0.13.0rc2

---

## References

- vLLM Disaggregated Prefill: [vLLM Documentation](https://docs.vllm.ai/)
- NCCL P2P Transfer: [NVIDIA NCCL](https://developer.nvidia.com/nccl)
