# vLLM Disaggregated Serving Optimizer

Dynamic request routing optimizer for multi-turn LLM inference across heterogeneous GPU pools.

---

## Project Overview

### Core Idea

Modern LLM serving systems face a fundamental trade-off between **resource efficiency** (separating prefill and decode) and **latency overhead** (KV cache transfer). This project explores how to **dynamically route requests** across a heterogeneous GPU pool based on workload characteristics.

**The central hypothesis**: Different workload patterns (conversation length, I/O ratio, context size) favor different serving strategies. Rather than picking one mode globally, we can achieve better performance by routing each request to the optimal execution path.

### Four Execution Modes

| Mode | Description | Turn 1 | Turn 2+ | Trade-off |
|------|-------------|--------|---------|-----------|
| **PD** | Full disaggregation | P→D (KV transfer) | P→D (KV transfer) | Isolation ✓, Transfer overhead ✗ |
| **PPD** | Prefix-aware PD | P→D (KV transfer) | D-direct (prefix cache) | No transfer ✓, Less isolation ✗ |
| **Replica** | Data parallelism | Worker X | Worker X (prefix cache) | Double capacity ✓, No isolation ✗ |
| **Optimizer** | Rule-based routing | Adaptive | Adaptive | Best of all modes ✓ |

### Optimizer: Rule-Based Request Router

The Optimizer mode implements intelligent request routing using a rule-based selector that dynamically chooses between PD, PPD, and Replica modes based on:

1. **Hybrid Architecture**: Uses PD for heavy prefill (input_tokens > 1024), PPD for light prefill
2. **Load-Aware Routing**: Routes to Replica when both P and D servers are busy
3. **E2E SLO Consideration**: Factors in end-to-end latency requirements

---

## Project Structure

```
ppd/
├── src/                              # Core source code
│   ├── benchmark_common.py           # Shared benchmark utilities
│   ├── benchmark_pd.py               # PD mode benchmark
│   ├── benchmark_ppd.py              # PPD mode benchmark
│   ├── benchmark_replica.py          # Replica mode benchmark
│   ├── disagg_proxy_ppd_4gpu.py      # PD/PPD proxy with cache affinity
│   ├── replication_proxy_4gpu.py     # Replication proxy (4 workers)
│   └── simple_replica_proxy.py       # Simple replica load balancer
├── scripts/
│   ├── server/                       # Server management scripts
│   │   ├── start_servers_4gpu.sh     # Start 2P+2D servers
│   │   ├── stop_servers_4gpu.sh      # Stop PD/PPD servers
│   │   ├── start_replication_servers_4gpu.sh  # Start 4 replicas
│   │   ├── stop_replication_servers_4gpu.sh   # Stop replicas
│   │   └── start_optimizer_servers.sh         # Start optimizer (1P+1D+2R)
│   ├── benchmark/                    # Benchmark job scripts
│   │   ├── sbatch_pd.sh              # SLURM job for PD benchmark
│   │   ├── sbatch_ppd.sh             # SLURM job for PPD benchmark
│   │   ├── sbatch_replica.sh         # SLURM job for Replica benchmark
│   │   └── submit_all.sh             # Submit all benchmark jobs
│   ├── analysis/                     # Analysis and visualization
│   │   ├── comprehensive_analysis.py # Full benchmark analysis
│   │   ├── generate_trend_data.py    # Generate SLO trend data
│   │   ├── merge_results.py          # Merge benchmark results
│   │   └── plot_trend_figures.py     # Generate publication figures
│   └── tests/                        # Test scripts
│       ├── test_decode_standalone.py # Test D server standalone capability
│       ├── test_mode_switching_cost.py        # Mode switching cost analysis
│       ├── test_optimizer_comparison.py       # 4-mode comparison test
│       ├── test_optimizer_value.py            # Optimizer value demonstration
│       ├── test_production_scenario.py        # Production scenario tests
│       └── verify_mode_switching.py           # Verify PD/PPD switching
├── optimizer/                        # Optimizer module
│   ├── improved_selector.py          # Hybrid mode selection (lookup + model)
│   ├── optimizer_router.py           # Request routing logic
│   └── metrics_collector.py          # Performance metrics collection
├── docs/                             # Documentation
│   ├── SLO_TEST_ANALYSIS.md          # Comprehensive SLO test analysis
│   ├── Cache_Mechanism_Analysis.md   # Cache mechanism deep analysis
│   ├── KV_CACHE_ARCHITECTURE_ANALYSIS.md  # KV cache architecture
│   ├── OPTIMIZER_ROUTER_DESIGN.md    # Optimizer design documentation
│   ├── TRADEOFF_ANALYSIS.md          # Mode trade-off analysis
│   ├── BENCHMARK.md                  # Benchmark methodology
│   └── notes.md                      # Development notes
├── results/                          # Benchmark results
│   ├── pd/                           # PD mode results
│   ├── ppd/                          # PPD mode results
│   ├── replica/                      # Replica mode results
│   ├── final/                        # Merged final results
│   ├── trend_data.json               # SLO trend data
│   └── production_scenario.json      # Production scenario results
├── figures/                          # Generated figures
│   └── fig1-fig7.png/pdf             # Publication figures
└── logs/                             # Server logs
```

---

## Key Results

### SLO Attainment Comparison (4 Modes)

| Mode | QPS=4 | QPS=8 | QPS=16 | QPS=32 |
|------|-------|-------|--------|--------|
| **PD** | 66.7% | 66.7% | 66.7% | 65.0% |
| **PPD** | 100.0% | 98.3% | 78.3% | 66.7% |
| **Replica** | 100.0% | 100.0% | 86.7% | 71.7% |
| **Optimizer** | 100.0% | 100.0% | 95.0% | 80.0% |

### Key Findings

1. **Optimizer achieves highest SLO attainment** across all QPS levels
2. **PPD excels at low-medium QPS** (≤8 QPS): Near-perfect SLO attainment
3. **Replica scales better at high QPS** (>16 QPS): Double capacity advantage
4. **PD shows consistent but limited performance**: ~66% baseline regardless of load

### Trade-off Insights

| Factor | PD | PPD | Replica | Optimizer |
|--------|-----|-----|---------|-----------|
| Turn 2 TTFT | Worst | Best | Good | Adaptive |
| High QPS capacity | Medium | Low | High | High |
| E2E latency focus | Poor | Good | Best | Best |
| Resource isolation | Best | Medium | None | Adaptive |

---

## Quick Start

### Environment Setup

```bash
conda activate vllm-ppd
cd /net/projects2/ds3lab/zongzel/vllm/ppd
```

### Start Optimizer Servers (1P + 1D + 2 Replica)

```bash
./scripts/server/start_optimizer_servers.sh
# Wait for "All Optimizer Servers Ready!" message

# Check status
curl -s http://localhost:10001/mode   # PD/PPD proxy
curl -s http://localhost:10002/status # Replica proxy
```

### Run SLO Test

```bash
# Generate trend data across 4 dimensions
python scripts/analysis/generate_trend_data.py

# Run production scenario tests
python scripts/tests/test_production_scenario.py

# Generate publication figures
python scripts/analysis/plot_trend_figures.py
```

### Run Full Benchmark Suite

```bash
# Submit all benchmark jobs (SLURM)
./scripts/benchmark/submit_all.sh

# Or run individually
sbatch scripts/benchmark/sbatch_ppd.sh 1
sbatch scripts/benchmark/sbatch_pd.sh 1
sbatch scripts/benchmark/sbatch_replica.sh 1

# Merge results after completion
python scripts/analysis/merge_results.py
```

---

## Server Configurations

### PD/PPD Mode (2P + 2D, 4 GPUs)

```
GPU 0: Prefill 0 (P0) - port 8100
GPU 1: Prefill 1 (P1) - port 8101
GPU 2: Decode 0 (D0)  - port 8200
GPU 3: Decode 1 (D1)  - port 8201
Proxy: port 10001
```

### Replica Mode (4 Workers)

```
GPU 0: Worker 0 - port 8300
GPU 1: Worker 1 - port 8400
GPU 2: Worker 2 - port 8500
GPU 3: Worker 3 - port 8600
Proxy: port 10002
```

### Optimizer Mode (1P + 1D + 2R)

```
GPU 0: Prefill (P)  - port 8100
GPU 1: Decode (D)   - port 8200
GPU 2: Replica 0    - port 8300
GPU 3: Replica 1    - port 8400
PD/PPD Proxy: port 10001
Replica Proxy: port 10002
```

---

## Workload Design

### Benchmark Workloads

| Context | Input Tokens | Output Tokens | Use Case |
|---------|-------------|---------------|----------|
| XS | 128 | 64/256/512 | Quick chat |
| S | 256 | 64/256/512 | Standard chat |
| M | 512 | 64/256/512 | Medium context |
| L | 1024 | 64/256/512 | Long context |
| XL | 2048 | 64/256/512 | Document analysis |

### Multi-Turn Protocol

**Turn 1**: Initial request through full P→D flow (establishes KV cache)
**Turn 2**: Subsequent request (measures cache efficiency)
- PD: Always goes through P→D
- PPD: Direct to D using prefix cache
- Replica: Same worker using local cache

### Metrics

- **TTFT**: Time to First Token
- **TPOT**: Time Per Output Token
- **E2E**: End-to-End latency
- **SLO Attainment**: % requests meeting latency target

---

## Analysis and Visualization

### Generated Figures (in `figures/`)

| Figure | Description |
|--------|-------------|
| fig1 | QPS vs SLO Attainment (4 modes) |
| fig2 | E2E Ratio vs SLO Attainment |
| fig3 | SLO Strictness vs Attainment |
| fig4 | Input Length vs SLO Attainment |
| fig5 | Production Scenarios Bar Chart |
| fig6 | Mode Comparison Radar Chart |
| fig7 | Summary Dashboard |

### Key Documents

- `docs/SLO_TEST_ANALYSIS.md`: Comprehensive SLO test analysis with quantitative results
- `docs/Cache_Mechanism_Analysis.md`: Deep dive into cache mechanisms across modes
- `docs/KV_CACHE_ARCHITECTURE_ANALYSIS.md`: Why PPD succeeds where PD fails
- `docs/TRADEOFF_ANALYSIS.md`: Mode trade-off analysis with recommendations

---

## API Endpoints

### PD/PPD Proxy (port 10001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Inference endpoint |
| `/v1/chat/completions` | POST | Chat inference |
| `/mode` | GET | Get current mode |
| `/mode/pd` | POST | Switch to PD mode |
| `/mode/ppd` | POST | Switch to PPD mode |
| `/conversations` | GET | List active conversations |
| `/conversations/clear` | POST | Clear conversation state |
| `/status` | GET | Server status |

### Replica Proxy (port 10002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Inference endpoint |
| `/v1/chat/completions` | POST | Chat inference |
| `/status` | GET | Proxy status and metrics |
| `/metrics/clear` | POST | Clear metrics |

---

## Troubleshooting

### Check Server Logs

```bash
tail -f logs/prefill.log    # Prefill server
tail -f logs/decode.log     # Decode server
tail -f logs/proxy.log      # PD/PPD proxy
tail -f logs/replica*.log   # Replica workers
```

### Stop All Servers

```bash
./scripts/server/stop_servers_4gpu.sh
./scripts/server/stop_replication_servers_4gpu.sh
# Or forcefully:
pkill -f "vllm serve"
pkill -f proxy
```

### NCCL Issues

First request may take 2-3 seconds due to NCCL initialization. Warmup is included in benchmarks to avoid affecting results.

---

## Hardware

- **Tested on**: 4x A100 80GB GPUs
- **Model**: Llama-3.1-8B
- **vLLM Version**: v0.13.0rc2

---

## References

- vLLM Disaggregated Prefill: [vLLM Documentation](https://docs.vllm.ai/)
- NCCL P2P Transfer: [NVIDIA NCCL](https://developer.nvidia.com/nccl)
