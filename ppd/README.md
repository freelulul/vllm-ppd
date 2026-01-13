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
│   ├── disagg_proxy_ppd.py         # PD/PPD proxy server (2-GPU)
│   ├── disagg_proxy_ppd_4gpu.py    # PD/PPD proxy with cache affinity (4-GPU)
│   ├── replication_proxy.py        # Replication proxy server (2-GPU)
│   ├── replication_proxy_4gpu.py   # Replication proxy (4-GPU, N-worker)
│   ├── qps_benchmark.py            # PD vs PPD benchmark
│   └── replication_benchmark.py    # Replication benchmark
├── scripts/
│   ├── start_servers.sh            # Start PD/PPD servers (2-GPU)
│   ├── stop_servers.sh             # Stop PD/PPD servers (2-GPU)
│   ├── start_servers_4gpu.sh       # Start 2P+2D servers (4-GPU) ⭐ NEW
│   ├── stop_servers_4gpu.sh        # Stop 4-GPU servers ⭐ NEW
│   ├── start_replication_servers.sh      # Start replication (2-GPU)
│   ├── stop_replication_servers.sh       # Stop replication (2-GPU)
│   ├── start_replication_servers_4gpu.sh # Start 4 replicas (4-GPU) ⭐ NEW
│   ├── stop_replication_servers_4gpu.sh  # Stop 4 replicas ⭐ NEW
│   ├── test_4gpu.py                # Test script for 4-GPU setup ⭐ NEW
│   ├── merge_results.py            # Merge benchmark results
│   └── plot_qps_curves.py          # Generate plots
├── docs/
│   ├── 4GPU_Architecture_Explained.md  # Detailed 4-GPU architecture ⭐ NEW
│   └── 4GPU_Complete_Summary.md        # Complete summary & comparison ⭐ NEW
├── results/                        # Benchmark results (JSON + PNG plots)
└── logs/                           # Server logs
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
- `ignore_eos`: True (prevents early termination, ensures precise output length)
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

### 2-GPU Setup (Original Benchmarks)
- 2x GPUs (tested on H100 80GB)
- Model: Llama-3.1-8B at `/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B`

### 4-GPU Setup (New - For Dynamic Router)
- 4x GPUs (tested on H100 80GB)
- Same model
- Supports:
  - 2P+2D (2 Prefill + 2 Decode) with cache affinity
  - 4 Replicas for baseline comparison

## Quick Start

### Environment Setup

```bash
conda activate vllm-ppd
cd /net/projects2/ds3lab/zongzel/vllm/ppd
```

---

## 🆕 4-GPU Setup (For Dynamic Router Development)

See detailed documentation:
- **Architecture Deep Dive**: `docs/4GPU_Architecture_Explained.md`
- **Complete Summary**: `docs/4GPU_Complete_Summary.md`

### Quick Test (2P+2D Mode)

```bash
# 1. Start 2P+2D (2 Prefill + 2 Decode)
./scripts/start_servers_4gpu.sh ppd
# Wait for "All 4-GPU servers ready!" message

# 2. Test Turn 1 (PD flow with KV transfer)
curl -s -w "\nTime: %{time_total}s\n" \
  -X POST http://localhost:10001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B",
       "prompt":"User: Hello conv_123.\nAssistant:",
       "max_tokens":30,
       "ignore_eos":true}'
# Expected: ~4s (includes NCCL setup)

# 3. Test Turn 2 (D-Direct with prefix cache)
cat > /tmp/turn2.json << 'EOF'
{"model":"/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B",
 "prompt":"User: Hello conv_123.\nAssistant: Hi!\nUser: Joke?\nAssistant:",
 "max_tokens":50,
 "ignore_eos":true}
EOF
curl -s -w "\nTime: %{time_total}s\n" \
  -X POST http://localhost:10001/v1/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/turn2.json
# Expected: ~0.4s (9.6x faster!)

# 4. Check conversation state (verify cache affinity)
curl -s http://localhost:10001/conversations | python3 -m json.tool

# 5. Stop
./scripts/stop_servers_4gpu.sh
```

### Quick Test (4 Replicas Mode)

```bash
# 1. Start 4 replicas
./scripts/start_replication_servers_4gpu.sh

# 2. Test
python scripts/test_4gpu.py --mode replication --port 10002

# 3. Check status
curl -s http://localhost:10002/status | python3 -m json.tool

# 4. Stop
./scripts/stop_replication_servers_4gpu.sh
```

### Key Findings (4-GPU)

**Architecture:**
- P0 ↔ P1: ❌ NO communication (independent instances)
- D0 ↔ D1: ❌ NO communication (independent instances)
- Each instance: Complete Llama-3.1-8B model (data parallelism, NOT model parallelism)

**Performance (PPD Mode with Cache Affinity):**
- Turn 1: 4.13s (PD flow with KV transfer)
- Turn 2: 0.43s (D-Direct using prefix cache)
- **Speedup: 9.6x** 🚀

**NCCL Groups:**
- 4 total groups: P0↔D0, P0↔D1, P1↔D0, P1↔D1
- Each group: world_size=2, created on-demand

**Cache Affinity:**
- Critical for PPD performance
- Conversations pinned to specific D instance
- Proxy tracks: `conversation_hash → decode_addr`

---

## 1. PD/PPD Benchmark (2-GPU)

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
