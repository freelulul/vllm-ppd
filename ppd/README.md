# vLLM Disaggregated Serving Optimizer

Dynamic request routing optimizer for multi-turn LLM inference across heterogeneous GPU pools.

---

## Project Overview

### Core Idea

Modern LLM serving systems face a fundamental trade-off between **resource efficiency** (separating prefill and decode) and **latency overhead** (KV cache transfer). This project explores how to **dynamically route requests** across a heterogeneous GPU pool based on workload characteristics.

**The central hypothesis**: Different workload patterns (conversation length, I/O ratio, context size) favor different serving strategies. Rather than picking one mode globally, we can achieve better performance by routing each request to the optimal execution path.

### GPU Pool Architecture

The system manages a heterogeneous pool of GPU resources with different capabilities:

| Label | Capability | Use Case |
|-------|------------|----------|
| **P** | Prefill only | Dedicated prefill workers for long-context requests |
| **D** | Decode only | Dedicated decode workers for generation-heavy tasks |
| **PD** | Both P and D | Full disaggregation with KV transfer between phases |
| **pD** | Prefix extension + D | Leverages prefix cache, skips KV transfer for cached prompts |

### Three Execution Modes

| Mode | Description | Turn 1 | Turn 2+ | Trade-off |
|------|-------------|--------|---------|-----------|
| **PD** | Full disaggregation | P→D (KV transfer) | P→D (KV transfer) | Isolation ✓, Transfer overhead ✗ |
| **PPD** | Prefix-aware PD | P→D (KV transfer) | D-direct (prefix cache) | No transfer ✓, Less isolation ✗ |
| **Replication** | Data parallelism | Worker X | Worker X (prefix cache) | Double capacity ✓, No isolation ✗ |

### Discovered Trade-offs

Through comprehensive benchmarking, we identified the following performance characteristics:

#### 1. PPD vs PD

**PPD wins in Turn 2 latency** (50-75% faster TTFT) by using prefix cache instead of KV transfer:
- **W1 (Chat)**: 70.6% faster on average
- **W2 (RAG)**: 44.4% faster (diminishes at high QPS due to HOL blocking)
- **W3 (Agent)**: 61.8% faster
- **W4 (Extreme)**: 34.1% faster (bandwidth still matters)

**When PD wins**: Never in Turn 2 TTFT, but provides better isolation at extreme load (prevents HOL blocking from long prefills).

#### 2. PPD vs Replication

**Workload-dependent winner**:
- **W1 (Chat)**: PPD ~8% better (efficient single-GPU utilization)
- **W2 (RAG)**: Replication better at high QPS (double GPU capacity prevents saturation)
- **W3 (Agent)**: PPD ~15% better (prefill-decode isolation helps long outputs)
- **W4 (Extreme)**: Replication wins (no network overhead, double memory bandwidth)

**Key insight**: Replication's advantage grows with context size and QPS. PPD's advantage grows with output length and moderate QPS.

#### 3. Critical Crossover Points

- **Low QPS (<2)**: PPD wins almost universally (prefix cache benefit dominates)
- **Medium QPS (2-8)**: Trade-off zone (depends on workload I/O ratio)
- **High QPS (>8)**: Replication or PD wins (isolation/capacity matters more than transfer cost)

---

## Project Goal

**Build a dynamic request router** that:

1. **Profiles incoming requests**: Extract features (input length, expected output length, conversation turn, historical latency)
2. **Predicts optimal mode**: Use learned decision boundaries from benchmark data
3. **Routes to appropriate GPU pool**: Assign request to P, D, PD, or pD workers based on prediction
4. **Adapts to load**: Consider current queue depths and GPU utilization

**Success metric**: Achieve better P99 latency than any single static mode across diverse workload mixes.

**Future work**:
- Machine learning-based routing (train on benchmark data)
- Online learning to adapt to deployment-specific patterns
- Cost-aware routing ($/token optimization, not just latency)

---

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

---

## Workload Design

### Benchmark Methodology

- **Arrival Process**: Poisson distribution (realistic traffic simulation)
- **QPS Sweep**: 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0 (finds saturation points)
- **Duration**: 60s per QPS point
- **Runs**: 3 runs averaged (filters anomalies >30% failure rate)
- **Warmup**: NCCL connection establishment before each benchmark

### Multi-Turn Conversation Protocol

Each benchmark run simulates realistic multi-turn conversations:

**Phase 1 - Conversation Establishment (Turn 1)**:
- Send initial greeting: `"Hello, I am user {conv_id}. Please acknowledge."`
- Server responds with ~20 tokens
- Conversation state cached on decode server
- **Non-streaming** for reliability

**Phase 2 - Actual Requests (Turn 2)**:
- Generate requests with Poisson arrival times
- Reuse established conversations (triggers prefix cache or KV transfer)
- Send workload-specific prompts
- **Streaming** to measure real TTFT

**Warmup Protocol**:
- Before any benchmark: Send 2 warmup requests (1 per mode)
- Eliminates NCCL connection overhead (~2-3s first request)
- Clear state before actual measurement

### Workload Definitions

| Workload | Input | Output | Turns | Description |
|----------|-------|--------|-------|-------------|
| **W1_Chat_Balanced** | 512 | 256 | 2 | Daily conversation, latency-sensitive |
| **W2_RAG_ReadHeavy** | 4096 | 128 | 2 | Long context retrieval, HOL blocking risk |
| **W3_Agent_WriteHeavy** | 256 | 1024 | 2 | Code/long generation, decode-bound |
| **W4_Limit_Context** | 8192 | 64 | 2 | Extreme context, bandwidth stress |

### Input Generation

**Prompt Template**:
```python
base_text = (
    "This is a comprehensive benchmark test for the vLLM disaggregated serving system. "
    "The system separates prefill and decode phases across different GPU instances. "
    "We are measuring latency under various load conditions to understand performance characteristics. "
)
```

**Length Control**:
- Target tokens → `target_chars = num_tokens * 4` (approximation: 1 token ≈ 4 chars)
- Repeat base_text to reach target length
- Add unique request ID: `[REQ:{req_id}] Analyze and respond: {repeated_text}`

**Output Control**:
- `max_tokens` parameter in API request
- Temperature: 0.8 (realistic diversity)
- Streaming: `True` for Turn 2 (measure TTFT accurately)

### Measured Metrics

**Per-Request Metrics**:
- **TTFT**: `time_first_token - request_start` (ms)
- **TPOT**: `(e2e_latency - TTFT) / (output_tokens - 1)` (ms/token)
- **E2E Latency**: `request_end - request_start` (ms)
- **Throughput**: `output_tokens / e2e_latency` (tokens/s)

**Aggregated Metrics** (across all requests at each QPS):
- P50, P90, P99 TTFT
- Average TTFT, TPOT, E2E
- Success rate (% requests completing without timeout)
- Real QPS achieved

**Turn-Specific Metrics**:
- Separate metrics for Turn 1 and Turn 2
- Turn 2 TTFT is **critical metric** (shows prefix cache vs KV transfer difference)

---

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

## Benchmark Results

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
