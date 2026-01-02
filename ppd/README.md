# vLLM PD/PPD/Replication Benchmark

Benchmark suite comparing three inference modes for multi-turn conversations on vLLM.

## Modes Overview

| Mode | Description | Turn 1 | Turn 2+ |
|------|-------------|--------|---------|
| **PD** | Prefill-Decode separation | P→D (KV transfer) | P→D (KV transfer) |
| **PPD** | Prompt-aware PD | P→D (KV transfer) | D-direct (prefix cache) |
| **Replication** | Data parallelism (2 workers) | Worker X | Worker X (prefix cache) |

## Project Structure

```
ppd/
├── src/
│   ├── disagg_proxy_ppd.py      # PD/PPD proxy server
│   ├── replication_proxy.py     # Replication proxy server
│   ├── qps_benchmark.py         # PD vs PPD benchmark
│   └── replication_benchmark.py # Replication benchmark
├── scripts/
│   ├── start_servers.sh         # Start PD/PPD servers
│   ├── stop_servers.sh          # Stop PD/PPD servers
│   ├── start_replication_servers.sh
│   ├── stop_replication_servers.sh
│   ├── merge_results.py         # Merge benchmark results
│   └── plot_qps_curves.py       # Generate plots
├── results/                     # Benchmark results (JSON + PNG plots)
└── logs/                        # Server logs
```

## Hardware Requirements

- 2x GPUs (tested on H100 80GB)
- Model: Llama-3.1-8B at `/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B`

## Quick Start

### Environment Setup

```bash
conda activate vllm-ppd
cd /net/projects2/ds3lab/zongzel/vllm/ppd
```

---

## 1. PD/PPD Benchmark

### Start Servers

```bash
./scripts/start_servers.sh
# Wait for "All servers ready!" message
# GPU 0: Prefill server (port 8100)
# GPU 1: Decode server (port 8200)
# Proxy: port 10001
```

### Run Benchmark

```bash
# Basic test (W1 workload, single QPS)
python src/qps_benchmark.py --workload W1 --qps 0.1 --duration 60

# Full sweep (multiple QPS values)
python src/qps_benchmark.py --workload W1 --qps 0.1,0.5,1.0,2.0,4.0,8.0,16.0 --duration 60

# All workloads with multiple runs for statistical significance
python src/qps_benchmark.py --workload all --qps 0.1,0.5,1.0,2.0,4.0,8.0,16.0 --duration 60 --runs 3
```

### Stop Servers

```bash
./scripts/stop_servers.sh
```

---

## 2. Replication Benchmark

### Start Servers

```bash
./scripts/start_replication_servers.sh
# Wait for "Replication Mode Ready!" message
# GPU 0: Worker 0 (port 8300)
# GPU 1: Worker 1 (port 8400)
# Proxy: port 10002
```

### Run Benchmark

```bash
# Basic test
python src/replication_benchmark.py --workload W1 --qps 0.1 --duration 60

# Full sweep with multiple runs
python src/replication_benchmark.py --workload all --qps 0.1,0.5,1.0,2.0,4.0,8.0,16.0 --duration 60 --runs 3
```

### Stop Servers

```bash
./scripts/stop_replication_servers.sh
```

---

## 3. Merge Results and Plot

### Merge Results

```bash
# Auto-detect latest result files
python scripts/merge_results.py --auto

# Or specify files manually
python scripts/merge_results.py \
    results/qps_benchmark_YYYYMMDD_HHMMSS.json \
    results/replication_benchmark_YYYYMMDD_HHMMSS.json
```

### Generate Plots

```bash
python scripts/plot_qps_curves.py results/merged_3mode_*.json
# Outputs: results/qps_*.png
```

---

## Workloads

| Workload | Input Tokens | Output Tokens | Description |
|----------|--------------|---------------|-------------|
| **W1** | 512 | 256 | Chat balanced, latency sensitive |
| **W2** | 4096 | 128 | RAG/retrieval, long input |
| **W3** | 256 | 1024 | Agent style, long generation |
| **W4** | 8192 | 64 | Extreme context stress test |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time To First Token (request start → first decoded token) |
| **TPOT** | Time Per Output Token = (E2E - TTFT) / (tokens - 1) |
| **E2E** | End-to-End latency (total request time) |

---

## Key Results

### PPD vs PD (Turn 2 TTFT Improvement)

| Workload | Avg Improvement | Min | Max |
|----------|-----------------|-----|-----|
| W1 Chat | 70.6% | 60.0% | 82.8% |
| W2 RAG | 44.4% | 5.9% | 71.7% |
| W3 Agent | 61.8% | 13.2% | 75.4% |
| W4 Extreme | 34.1% | 21.4% | 56.9% |

### PPD vs Replication

| Workload | Result |
|----------|--------|
| W1 Chat | PPD ~8% better (efficient single-GPU utilization) |
| W2 RAG | Replication better at high QPS (double capacity) |
| W3 Agent | PPD ~15% better (PD isolation benefits long output) |
| W4 Extreme | Replication better (handles extreme context) |

### Key Findings

1. **PPD consistently beats PD**: 50-75% faster Turn 2 TTFT by using prefix cache
2. **PPD vs Replication**: Workload-dependent
   - Chat/Agent workloads: PPD slightly better
   - RAG/Extreme context: Replication better at high load
3. **Replication advantage**: Double GPU capacity, better for extreme workloads
4. **PPD advantage**: Prefill-decode isolation prevents HOL blocking

---

## Generated Plots

After running `plot_qps_curves.py`:

| File | Description |
|------|-------------|
| `qps_p99_ttft.png` | P99 TTFT vs QPS for all workloads |
| `qps_avg_e2e.png` | Average E2E latency vs QPS |
| `qps_ratio_crossover.png` | Performance ratio analysis |
| `qps_success_rate.png` | Success rate vs QPS |
| `qps_throughput.png` | Throughput vs QPS |
| `qps_throughput_comparison.png` | Throughput comparison bar chart |
| `qps_crossover_analysis.json` | Crossover point analysis |

---

## Server Endpoints

### PD/PPD Proxy (port 10001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Inference endpoint |
| `/mode` | GET | Get current mode (pd/ppd) |
| `/mode/pd` | POST | Switch to PD mode |
| `/mode/ppd` | POST | Switch to PPD mode |
| `/conversations/clear` | POST | Clear conversation state |

### Replication Proxy (port 10002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Inference endpoint |
| `/status` | GET | Get proxy status |
| `/metrics/clear` | POST | Clear metrics |

---

## Troubleshooting

### Check Server Logs

```bash
tail -f logs/prefill.log   # Prefill server (PD/PPD)
tail -f logs/decode.log    # Decode server (PD/PPD)
tail -f logs/proxy.log     # Proxy server
tail -f logs/worker0.log   # Worker 0 (Replication)
tail -f logs/worker1.log   # Worker 1 (Replication)
```

### Port Conflicts

```bash
pkill -f "vllm.entrypoints"
pkill -f "disagg_proxy"
pkill -f "replication_proxy"
```

### NCCL Warmup

First request takes 2-3 seconds due to NCCL initialization. The benchmark includes warmup to avoid affecting results.
