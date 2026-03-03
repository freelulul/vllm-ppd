# PPD: Dynamic Append-Prefill Routing for Disaggregated LLM Serving

A dynamic request router for multi-turn LLM inference that intelligently routes append-prefill operations across prefill and decode nodes in disaggregated serving systems.

## Key Insight

In multi-turn conversations, Prefill-Decode (PD) disaggregation re-transfers the entire KV cache every turn, even when subsequent inputs are short. We observe that **append-prefill causes an order-of-magnitude less interference** to decode than full prefill (~2% vs ~48% TPOT degradation). This enables a new routing dimension: selectively processing append-prefill locally on decode nodes to eliminate KV transfer overhead while preserving decode quality.

## Architecture

### Machine Types

| Type | Name | Capability | Use Case |
|------|------|------------|----------|
| **P** | Prefill-only | Full prefill | Turn 1 prefill |
| **D** | Decode-only | Decode (receives KV from P) | Traditional PD |
| **pD** | Prefill-capable Decode | Append-prefill + decode | PPD mode (Turn 2+ local) |
| **R** | Replica | Full model (prefill + decode) | Baseline replica mode |

### Routing Modes

| Mode | Config | Turn 1 | Turn 2+ | Trade-off |
|------|--------|--------|---------|-----------|
| **PD** (x=0) | P + D | P prefill -> KV transfer -> D | Same path every turn | Isolation, KV transfer overhead |
| **PPD** (x=1) | P + pD | P prefill -> KV transfer -> pD | pD-direct (prefix cache) | No T2+ transfer, less isolation |
| **Replica** | R | Any worker | Same worker (prefix cache) | High throughput, no isolation |

The routing parameter **x in [0,1]** controls the fraction of decode nodes that handle append-prefill locally. PPD dynamically selects x per-request based on workload characteristics.

## Configuration Space

17 GPU configurations on 4 GPUs:

| Category | Configurations |
|----------|---------------|
| **Replica** | 4R |
| **PD (x=0)** | 1P_3D, 2P_2D, 3P_1D |
| **Full AP-to-D (x=1)** | 1P_3pD, 2P_2pD, 3P_1pD |
| **Partial (0<x<1)** | 1P_2D_1pD, 1P_1D_2pD, 2P_1D_1pD |
| **Hybrid (R + disagg.)** | 1R_1P_2D, 1R_1P_1D_1pD, 1R_1P_2pD, 1R_2P_1D, 1R_2P_1pD, 2R_1P_1D, 2R_1P_1pD |

## Project Structure

```
ppd/
├── src/
│   ├── comprehensive_proxy.py    # Unified proxy for all 17 configurations
│   └── config.py                 # Centralized configuration
│
├── optimizer/
│   └── ppd_decision_engine.py    # Dynamic PPD routing decision engine
│
├── scripts/
│   ├── server/                   # Server management
│   │   ├── config.sh             # Unified shell configuration
│   │   ├── common.sh             # Shared utilities
│   │   ├── start_<CONFIG>.sh     # 17 config-specific startup scripts
│   │   ├── stop.sh               # Unified stop
│   │   ├── cleanup_all.sh        # Force cleanup
│   │   └── generate_configs.py   # Config generation utility
│   │
│   ├── benchmark/                # Benchmarking
│   │   ├── comprehensive_benchmark.py   # Synthetic workload benchmark
│   │   ├── sharegpt_benchmark.py        # Real dataset benchmark
│   │   ├── model_scaling_benchmark.py   # Model size scaling
│   │   └── turn_scaling_benchmark.py    # Turn count scaling
│   │
│   ├── tests/
│   │   └── interference_benchmark.py    # Prefill-decode interference test
│   │
│   └── download_datasets.py     # Dataset download utility
│
├── data/                         # Datasets (download separately)
└── results/                      # Benchmark results (download separately)
    └── comprehensive/            # Decision engine lookup data
```

## Quick Start

### Prerequisites

- 4x NVIDIA H100 80GB GPUs (or equivalent)
- Python 3.10+
- vLLM with disaggregated serving support

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set model path
export MODEL_PATH="/path/to/meta-llama/Llama-3.1-8B"

# 3. Download datasets
python scripts/download_datasets.py
```

### Running Servers

```bash
# Start a configuration (e.g., 2P_2D = 2 Prefill + 2 Decode)
./scripts/server/start_2P_2D.sh

# Stop all servers
./scripts/server/stop.sh
```

### Running Benchmarks

```bash
# Synthetic workload benchmark
python scripts/benchmark/comprehensive_benchmark.py \
    --config 2P_2D \
    --workload all \
    --qps 1 2 4 8

# Real dataset benchmark (ShareGPT multi-turn conversations)
python scripts/benchmark/sharegpt_benchmark.py \
    --config 2P_2D \
    --num-conversations 500 \
    --qps 2 4 8 12

# With dynamic PPD routing enabled
python scripts/benchmark/sharegpt_benchmark.py \
    --config 2P_2D \
    --enable-ppd \
    --num-conversations 500 \
    --qps 2 4 8 12
```

### Interference Benchmark

```bash
# Measure prefill-decode interference (full vs append prefill)
python scripts/tests/interference_benchmark.py
```

## PPD Decision Engine

The decision engine (`optimizer/ppd_decision_engine.py`) uses benchmark data to make per-request routing decisions.

**Scoring formula:**
```
score = w_ttft * TTFT_improvement - w_tpot * TPOT_degradation
use_ppd = (score > 0)
```

**Decision factors:**
- `context_length`: small (<=512), large (512-4096), huge (>4096)
- `input_tokens`: New tokens in current turn
- `output_tokens`: Expected output tokens
- `current_qps`: System load

**Weight configuration:**
```bash
# Balanced (default)
--w-ttft 1.0 --w-tpot 1.0

# TTFT-priority
--w-ttft 2.0 --w-tpot 1.0

# TPOT-priority
--w-ttft 1.0 --w-tpot 2.0
```

## Benchmark Workload Matrix

**Turn 1 contexts (2):** small (128->128), large (1024->1024)

**Turn 2 workloads (9):**

| Type | Input->Output | Category |
|------|---------------|----------|
| tiny | 16->32 | Minimal |
| short_gen | 32->256 | Decode-heavy |
| long_gen | 32->512 | Decode-heavy |
| very_long_gen | 64->1024 | Decode-heavy |
| small_bal | 64->64 | Balanced |
| mid_bal | 128->128 | Balanced |
| mid_paste | 256->64 | Prefill-heavy |
| big_paste | 512->64 | Prefill-heavy |
| huge_paste | 1024->32 | Prefill-heavy |

**QPS levels:** 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20

## Port Assignments

| Component | Port |
|-----------|------|
| Comprehensive Proxy | 10001 |
| Replica Proxy (4R only) | 10002 |
| Prefill servers | 8100-8101 |
| Decode servers | 8200-8201 |
| Replica workers | 8300-8600 |

## Hardware Requirements

- **GPUs**: 4x H100 80GB (tested configuration)
- **Model**: Llama-3.1-8B (primary), supports other models via MODEL_PATH
- **vLLM**: v0.7.0+ with disaggregated serving support

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `meta-llama/Llama-3.1-8B` | Path to model weights |
| `MAX_MODEL_LEN` | `8192` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.85` | GPU memory fraction |
| `ENABLE_PPD_MODE` | `false` | Enable dynamic PPD routing |
| `PPD_BENCHMARK_PATH` | `results/comprehensive` | Benchmark data for decision engine |
| `W_TTFT` | `1.0` | TTFT improvement weight |
| `W_TPOT` | `1.0` | TPOT degradation penalty weight |
