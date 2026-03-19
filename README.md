# PPD: Not All Prefills Are Equal

**Dynamic Append-Prefill Routing for Disaggregated Multi-turn LLM Serving**

[\[Paper\]](https://arxiv.org/abs/2603.13358)

In multi-turn conversations, Prefill-Decode (PD) disaggregation re-transfers the entire KV cache every turn, even when subsequent inputs are short. We observe that **append-prefill causes an order-of-magnitude less interference** to decode than full prefill (~2% vs ~48% TPOT degradation). This enables a new routing dimension: selectively processing append-prefill locally on decode nodes to eliminate KV transfer overhead while preserving decode quality.

PPD introduces a **prefill-capable decode node (pD)** that performs append-prefill + decode locally for Turn 2+ requests, and a per-request **decision engine** that dynamically selects between PD and PPD routing based on workload characteristics.

## Architecture

### Request Flow

```
Turn 1:  Client ──> Proxy ──> P (prefill) ──KV transfer via NCCL──> pD (decode)
Turn 2+: Client ──> Proxy ──> pD (append-prefill + decode, local prefix cache)
                                  └── no KV transfer needed
```

### Machine Types

| Type | Name | Capability | Use Case |
|------|------|------------|----------|
| **P** | Prefill-only | Full prefill | Turn 1 prefill, sends KV to decode nodes via NCCL P2P |
| **D** | Decode-only | Decode (receives KV from P) | Traditional PD disaggregation |
| **pD** | Prefill-capable Decode | Append-prefill + decode | PPD mode: Turn 2+ processing with local prefix cache |
| **R** | Replica | Full model (prefill + decode) | Baseline replica mode, no disaggregation |

### Routing Modes

| Mode | Config | Turn 1 | Turn 2+ | Trade-off |
|------|--------|--------|---------|-----------|
| **PD** (x=0) | P + D | P prefill &rarr; KV transfer &rarr; D | Same path every turn | Full isolation, KV transfer overhead |
| **PPD** (x=1) | P + pD | P prefill &rarr; KV transfer &rarr; pD | pD-direct (prefix cache) | No T2+ transfer, less isolation |
| **PPD-Dynamic** | P + pD | Decision engine selects | Per-request PD or PPD | Adaptive, best of both |
| **Replica** | R | Any worker | Same worker (prefix cache) | High throughput, no isolation |

The routing parameter **x &isin; [0,1]** controls the fraction of decode nodes that handle append-prefill locally. PPD dynamically selects x per-request based on workload characteristics.

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
ppd/                              # PPD source code
    comprehensive_proxy.py        # Unified async proxy (Quart + aiohttp + ZMQ)
    config.py                     # Centralized configuration (ports, model, env vars)
    optimizer/
        ppd_decision_engine.py    # Per-request PD vs PPD routing decision engine

scripts/
    server/                       # Server management
        config.sh                 # Unified shell configuration
        common.sh                 # Shared utilities (GPU checks, cleanup)
        start_<CONFIG>.sh         # 17 config-specific startup scripts
        start_single_replica.sh   # Single GPU server for micro-benchmarks
        stop.sh                   # Unified stop
        cleanup_all.sh            # Force cleanup all processes
        generate_configs.py       # Config script generation utility
    benchmark/
        comprehensive_benchmark.py   # Synthetic workload benchmark
        sharegpt_benchmark.py        # Real multi-turn dataset benchmark
        model_scaling_benchmark.py   # Model size scaling benchmark
        turn_scaling_benchmark.py    # Conversation turn scaling benchmark
    tests/
        interference_benchmark.py    # Prefill-decode interference micro-benchmark
    download_datasets.py             # Dataset download utility

vllm-source/                      # Vendored vLLM with PPD KV connector patch
data/                             # Datasets (download separately)
results/                          # Benchmark results
    comprehensive/                # Decision engine lookup data
```

**vLLM patch:** A single modification in `vllm-source/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` adds D-direct mode detection &mdash; when a request arrives at a pD node without PD-format request ID, it skips external KV loading and uses the local prefix cache instead.

## Quick Start

### Prerequisites

- 4x NVIDIA H100 80GB GPUs (or equivalent)
- Python 3.10+, CUDA toolkit, CMake, Ninja

### Installation

```bash
# Install vLLM (from vendored source with PPD patch) + PPD dependencies
./setup.sh

# Set model path
export MODEL_PATH="/path/to/meta-llama/Llama-3.1-8B"

# Download datasets (optional, for ShareGPT benchmark)
python scripts/download_datasets.py
```

### Running Servers

```bash
# Start a configuration (e.g., 2 Prefill + 2 pD)
./scripts/server/start_2P_2pD.sh

# Stop all servers
./scripts/server/stop.sh

# Force cleanup all GPU processes
./scripts/server/cleanup_all.sh
```

All 17 configurations have dedicated start scripts in `scripts/server/`:

```bash
./scripts/server/start_4R.sh          # 4 Replicas (baseline)
./scripts/server/start_2P_2D.sh       # 2 Prefill + 2 Decode (PD baseline)
./scripts/server/start_2P_2pD.sh      # 2 Prefill + 2 pD (PPD)
./scripts/server/start_1P_2D_1pD.sh   # Partial PPD
# ... etc
```

Model and server parameters can be overridden via environment variables:

```bash
export MODEL_PATH="/path/to/Qwen2.5-14B"
export MAX_MODEL_LEN=16384
export GPU_MEMORY_UTILIZATION=0.90
./scripts/server/start_2P_2pD.sh
```

### Quick Verification

```bash
# Start 2P_2pD and run a single benchmark point
export MODEL_PATH="meta-llama/Llama-3.1-8B"
./scripts/server/start_2P_2pD.sh

python scripts/benchmark/comprehensive_benchmark.py \
    --config 2P_2pD --workload small_tiny --qps 1 --skip-startup

./scripts/server/stop.sh
```

## Benchmarks

### Comprehensive Benchmark (Synthetic Workloads)

Tests 17 GPU configurations &times; 18 workloads &times; 10 QPS points = **3,060 test points** with OOM resilience (automatic restart on failure, per-test-point timeout).

```bash
# Run all workloads for a specific config
python scripts/benchmark/comprehensive_benchmark.py \
    --config 2P_2pD \
    --workload all \
    --qps 1 2 4 8

# Run specific workloads
python scripts/benchmark/comprehensive_benchmark.py \
    --config 2P_2D 2P_2pD \
    --workload small_tiny small_mid_bal large_mid_bal \
    --qps 1 2 4 8 \
    --output-dir results/comprehensive

# List available configurations and workloads
python scripts/benchmark/comprehensive_benchmark.py --list-configs
python scripts/benchmark/comprehensive_benchmark.py --list-workloads
```

### ShareGPT Multi-turn Benchmark

Tests with real multi-turn conversations from the ShareGPT dataset. Records metrics for all turns but aggregates focus on Turn 2+ (where PPD benefits materialize).

```bash
# Baseline test (standard PD routing)
python scripts/benchmark/sharegpt_benchmark.py \
    --config 2P_2D \
    --num-conversations 500 \
    --qps 2 4 8 12

# With dynamic PPD routing enabled
python scripts/benchmark/sharegpt_benchmark.py \
    --config 2P_2pD \
    --enable-ppd \
    --num-conversations 500 \
    --qps 2 4 8 12

# Custom PPD decision weights (TTFT-priority)
python scripts/benchmark/sharegpt_benchmark.py \
    --config 2P_2pD \
    --enable-ppd \
    --w-ttft 2.0 --w-tpot 1.0 \
    --num-conversations 500 \
    --qps 4
```

Output includes per-turn and aggregated metrics with SLO attainment percentages (TTFT 100/200/500ms, TPOT 15/30ms, E2E 5000/10000ms).

### Interference Micro-benchmark

Demonstrates the key observation: full-prefill causes ~48% TPOT degradation while append-prefill causes only ~2%. Runs on a single GPU.

```bash
# Start single replica server
./scripts/server/start_single_replica.sh

# Core experiment: decode-only vs decode+full-prefill vs decode+append-prefill
python scripts/tests/interference_benchmark.py --core

# Sensitivity experiment: varying context lengths (2K-102K tokens)
python scripts/tests/interference_benchmark.py --sensitivity

# Run all experiments
python scripts/tests/interference_benchmark.py --all

# Custom server URL
python scripts/tests/interference_benchmark.py --core --server-url http://localhost:8300
```

### Model Scaling Benchmark

Tests across different model sizes: 8B (Llama-3.1-8B), 14B (Qwen2.5-14B), 27B (Gemma-2-27B).

```bash
python scripts/benchmark/model_scaling_benchmark.py \
    --models 8B 14B \
    --configs 4R 2P_2D 2P_2pD \
    --qps 4
```

### Turn Scaling Benchmark

Tests varying conversation lengths (2, 4, 8, 16 turns) to measure how PPD benefits scale with conversation depth.

```bash
python scripts/benchmark/turn_scaling_benchmark.py \
    --configs 4R 2P_2D 2P_2pD \
    --turns 2 4 8 16 \
    --qps 4
```

## Benchmark Workload Matrix

**Turn 1 contexts (2):** small (128&rarr;128), large (1024&rarr;1024)

**Turn 2 workloads (9):**

| Type | Input&rarr;Output | Category |
|------|-------------------|----------|
| tiny | 16&rarr;32 | Minimal |
| short_gen | 32&rarr;256 | Decode-heavy |
| long_gen | 32&rarr;512 | Decode-heavy |
| very_long_gen | 64&rarr;1024 | Decode-heavy |
| small_bal | 64&rarr;64 | Balanced |
| mid_bal | 128&rarr;128 | Balanced |
| mid_paste | 256&rarr;64 | Prefill-heavy |
| big_paste | 512&rarr;64 | Prefill-heavy |
| huge_paste | 1024&rarr;32 | Prefill-heavy |

**QPS levels:** 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20

## PPD Decision Engine

The decision engine (`ppd/optimizer/ppd_decision_engine.py`) loads comprehensive benchmark results to build per-(workload, QPS) lookup tables that compare PPD vs PD configurations.

**Scoring formula:**

```
score = w_ttft * TTFT_improvement - w_tpot * TPOT_degradation
use_ppd = (score > 0)
```

where:

- `TTFT_improvement = (pd_ttft - ppd_ttft) / pd_ttft`
- `TPOT_degradation = (ppd_tpot - pd_tpot) / pd_tpot`

**Request classification:**

| Factor | Categories |
|--------|------------|
| Context length | small (&le;512), large (512&ndash;4096), huge (&gt;4096) |
| Workload type | 9 categories based on input/output ratio (see matrix above) |
| System load | Current QPS mapped to nearest benchmark QPS point |

**Weight configuration:**

```bash
# Balanced (default)
--w-ttft 1.0 --w-tpot 1.0

# TTFT-priority (latency-sensitive)
--w-ttft 2.0 --w-tpot 1.0

# TPOT-priority (throughput-sensitive)
--w-ttft 1.0 --w-tpot 2.0
```

The engine always returns PD for Turn 1 requests (no prefix cache available). For Turn 2+, it scores the request and selects PPD when the expected TTFT improvement outweighs the potential TPOT degradation.

## Port Assignments

| Component | Port |
|-----------|------|
| Comprehensive Proxy | 10001 |
| Replica Proxy (4R only) | 10002 |
| ZMQ Registration | 30001 |
| Prefill servers | 8100&ndash;8101 |
| Decode servers | 8200&ndash;8201 |
| Replica workers | 8300&ndash;8600 |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `meta-llama/Llama-3.1-8B` | Path to model weights |
| `MODEL_NAME` | `Llama-3.1-8B` | Model name for process management |
| `MAX_MODEL_LEN` | `8192` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.85` | GPU VRAM fraction |
| `ENABLE_PPD_MODE` | `false` | Enable dynamic PPD routing in proxy |
| `PPD_BENCHMARK_PATH` | `results/comprehensive` | Benchmark data for decision engine |
| `W_TTFT` | `1.0` | TTFT improvement weight |
| `W_TPOT` | `1.0` | TPOT degradation penalty weight |
| `MAX_WAIT` | `300` | Server startup timeout (seconds) |

## Hardware Requirements

- **GPUs**: 4x H100 80GB (tested configuration)
- **Model**: Llama-3.1-8B (primary), supports other models via `MODEL_PATH`
- **Software**: Python 3.10+, CUDA toolkit, vLLM (vendored in `vllm-source/`)

## Citation

If you use PPD in your research, please cite:

```bibtex
@misc{li2026prefillsequalppddisaggregation,
      title={Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving},
      author={Zongze Li and Jingyu Liu and Zach Xu and Yineng Zhang and Tahseen Rabbani and Ce Zhang},
      year={2026},
      eprint={2603.13358},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2603.13358},
}
```

## License

Apache 2.0
