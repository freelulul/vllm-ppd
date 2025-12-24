# vLLM PD Separation (Prefill/Decode Disaggregation)

## Overview

This project tests and researches vLLM's Prefill/Decode (PD) disaggregation functionality. The goal is to study KV cache transfer decisions in multi-turn dialogue scenarios to optimize routing between prefill and decode instances.

## Hardware Configuration

- **GPU**: 2x NVIDIA H100 80GB HBM3
  - GPU 0: Prefill instance (kv_producer)
  - GPU 1: Decode instance (kv_consumer)
- **Driver**: 555.42.06

## Software Environment

- **Conda Environment**: `vllm-ppd`
- **vLLM Version**: 0.13.0rc2.dev292+gd6b3d39b6 (installed from source)
- **vLLM Source Path**: `/net/projects2/ds3lab/zongzel/vllm`
- **Model**: Meta-Llama-3.1-8B at `/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B`

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n vllm-ppd python=3.11 -y
conda activate vllm-ppd
```

### 2. Install vLLM from Source

```bash
cd /net/projects2/ds3lab/zongzel/vllm
pip install -e .
```

### 3. Install Additional Dependencies

```bash
pip install quart aiohttp
```

### 4. Verify Installation

```bash
conda activate vllm-ppd
python -c "import vllm; print('vLLM version:', vllm.__version__)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

## PD Disaggregation Architecture

```
+---------------------------------------------------------------------+
|                    PD Disaggregation Architecture                    |
+---------------------------------------------------------------------+
|                                                                      |
|    Client Request                                                    |
|         |                                                            |
|         v                                                            |
|    +-------------+                                                   |
|    |   Proxy     |  (Port 10001)                                     |
|    |   Server    |  disagg_proxy_p2p_nccl_xpyd.py                    |
|    +------+------+                                                   |
|           |                                                          |
|           | 1. Route to Prefill                                      |
|           v                                                          |
|    +-------------+         NCCL (TCP)         +-------------+        |
|    |  Prefill    | -------------------------> |   Decode    |        |
|    |  Instance   |      KV Cache Transfer     |  Instance   |        |
|    |  (GPU 0)    |                            |  (GPU 1)    |        |
|    |  Port 8100  |                            |  Port 8200  |        |
|    | kv_producer |                            | kv_consumer |        |
|    +-------------+                            +------+------+        |
|                                                      |               |
|                                                      v               |
|                                               Response to Client     |
|                                                                      |
+---------------------------------------------------------------------+
```

## Connector Configuration

This project uses **P2pNcclConnector** for KV cache transfer via TCP.

### Configuration Parameters

```json
{
    "kv_connector": "P2pNcclConnector",
    "kv_role": "kv_producer/kv_consumer",
    "kv_buffer_size": "1e9",
    "kv_port": "14579",
    "kv_connector_extra_config": {
        "proxy_ip": "0.0.0.0",
        "proxy_port": "30001",
        "http_port": "8100",
        "send_type": "PUT_ASYNC"
    }
}
```

### Force TCP Transport

These environment variables disable RDMA/NVLink and force TCP Socket transport:

```bash
export NCCL_IB_DISABLE=1    # Disable InfiniBand
export NCCL_NET=Socket      # Force Socket (TCP) transport
```

## Quick Start

### Run PD Separation Test

```bash
cd /net/projects2/ds3lab/zongzel/vllm/ppd
conda activate vllm-ppd
./scripts/run_pd_separation_test.sh
```

The script automatically cleans old logs and keeps only the latest run.

### Run Bandwidth Test

After the PD separation test is running:

```bash
python src/bandwidth_test.py --prompt-lengths 100 500 1000 --iterations 3
```

## KV Cache Transfer Bandwidth

### KV Cache Size Calculation (Llama-3.1-8B)

```
KV size per token = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
                  = 32 * 2 * 8 * 128 * 2
                  = 131,072 bytes = 128 KB/token

For 1000 tokens: 128 KB * 1000 = 128 MB
```

### Hardware Bandwidth Reference

| Transport | Theoretical | Practical |
|-----------|-------------|-----------|
| Localhost loopback | N/A | 10-50 GB/s (memory-bound) |
| PCIe 4.0 x16 | 32 GB/s | ~25 GB/s |
| 100GbE Network | 12.5 GB/s | ~10 GB/s |
| 25GbE Network | 3.1 GB/s | ~2.5 GB/s |
| 10GbE Network | 1.25 GB/s | ~1.0 GB/s |

### Bandwidth Verification

The `bandwidth_test.py` script measures apparent bandwidth and compares with hardware limits to verify TCP transfer is working correctly.

## Native Metrics Interfaces

### vLLM Request Metrics

Available in `vllm/v1/metrics/stats.py`:

| Metric | Description |
|--------|-------------|
| `e2e_latency` | End-to-end latency |
| `queued_time` | Request queue time |
| `prefill_time` | Prefill phase time (from scheduling to first token) |
| `decode_time` | Decode phase time (from first to last token) |
| `inference_time` | Total inference time |
| `mean_time_per_output_token` | Average time per output token |
| `num_cached_tokens` | Number of cache-hit tokens |

### KV Transfer Metrics (NixlConnector)

Available in `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`:

| Metric | Description |
|--------|-------------|
| `transfer_duration` | KV transfer time |
| `post_duration` | Post-processing time |
| `bytes_transferred` | Bytes transferred |
| `num_descriptors` | Number of transfer descriptors |

**Note**: P2pNcclConnector does not have built-in detailed transfer metrics. Use `bandwidth_test.py` to measure transfer performance.

## File Structure

```
ppd/
├── README.md                    # This documentation
├── run_pd_separation_test.sh    # Main test script (auto-cleans old logs)
├── bandwidth_test.py            # Bandwidth measurement and verification
├── test_client.py               # Python test client
└── logs/                        # Log directory (latest run only)
    ├── prefill_*.log           # Prefill instance log
    ├── decode_*.log            # Decode instance log
    └── proxy_*.log             # Proxy server log
```

## Available Connector Types

| Connector | Protocol | Features |
|-----------|----------|----------|
| P2pNcclConnector | NCCL (TCP/RDMA) | Dynamic scaling, currently used |
| NixlConnector | UCX (RDMA/TCP) | Detailed metrics, production recommended |
| MooncakeConnector | RDMA | High performance, requires RDMA hardware |
| LMCacheConnector | Redis/File | KV cache persistence |
| OffloadingConnector | PCIe/NVMe | CPU/SSD offloading |

## Troubleshooting

### Port Already in Use

```bash
lsof -i:8000 -i:8100 -i:8200
kill -9 <PID>
```

### GPU Out of Memory

- Reduce `--max-model-len`
- Reduce `--gpu-memory-utilization`

### NCCL Connection Failed

- Check firewall settings
- Ensure NCCL ports (14579, 14580) are accessible
- Set `NCCL_DEBUG=INFO` for detailed logs

### Model Loading Slow

- Initial model loading takes several minutes
- Check logs to confirm loading progress

## Multi-Turn Dialogue Support

### Understanding Multi-Turn PD Separation

In multi-turn dialogue scenarios, the key question is: **where should subsequent prompts be processed?**

```
Turn 1: User prompt → P (prefill) → [KV Transfer] → D (decode) → Response
Turn 2: User prompt → ???
        Option A: → P (prefill all) → [Full KV Transfer] → D → Response
        Option B: → D directly (reuse cached KV, prefill new tokens locally)
```

### Current Proxy Behavior (xpyd)

The current `disagg_proxy_p2p_nccl_xpyd.py` **always routes to P first**:

1. Every turn: Proxy sends request to P with `max_tokens=1`
2. P computes prefill for **entire context** (cumulative)
3. P transfers **full KV cache** to D
4. Proxy then sends original request to D for decoding

### vLLM's KV Cache Reuse Mechanism

vLLM has built-in prefix caching that helps D reuse KV cache:

| Metric | Description |
|--------|-------------|
| `Prefix cache hit rate` | Local cache reuse on D (previous turns) |
| `External prefix cache hit rate` | KV cache from P→D transfer |

**Observed Results from Multi-Turn Test:**

| Scenario | Local Prefix Cache | External (P→D) Cache |
|----------|--------------------|-----------------------|
| Single requests | 0.0% | 93.8% |
| Multi-turn dialogue | **64.4%** | 35.0% |

**Key Insight:** D machine IS reusing KV cache locally across turns (64.4% hit rate)!

### Running Multi-Turn Test

```bash
# Start PD servers first
./scripts/run_pd_separation_test.sh &

# Wait for initialization, then run multi-turn test
python src/multi_turn_test.py --turns 5 --input-tokens 100 --output-tokens 30

# Check cache hit rates in decode logs
grep "prefix cache" logs/decode_*.log
```

### Multi-Turn Test Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--turns` | Number of conversation turns | 5 |
| `--input-tokens` | Approximate input tokens per turn | 100 |
| `--output-tokens` | Output tokens per turn | 50 |
| `--format` | API format (chat/completion) | completion |
| `--output` | JSON file for results | None |

### Performance Characteristics

For a 5-turn conversation with ~100 tokens per turn:

| Turn | Cumulative Tokens | KV Size | Latency |
|------|-------------------|---------|---------|
| 1 | 91 | 11.38 MB | ~150 ms |
| 2 | 189 | 23.62 MB | ~160 ms |
| 3 | 288 | 36.00 MB | ~165 ms |
| 4 | 387 | 48.38 MB | ~150 ms |
| 5 | 485 | 60.62 MB | ~165 ms |

**Note:** Latency remains relatively stable due to D's local prefix cache reuse!

### PPD Optimization Opportunity

The current flow has inefficiency:
- **Current:** Every turn transfers FULL cumulative KV cache from P→D
- **Ideal:** Only transfer KV for NEW tokens each turn

For a 10-turn conversation with 100 tokens each:
- **Current transfer:** 100 + 200 + 300 + ... + 1000 = 5500 × 128KB = 688 MB
- **Optimal transfer:** 100 × 10 × 128KB = 125 MB (5.5x reduction!)

This is where the PPD routing decision becomes critical.

## File Structure

```
ppd/
├── README.md                      # This documentation
├── .gitignore                     # Git ignore rules (excludes logs/ and results/)
├── scripts/                       # Shell scripts
│   ├── run_pd_separation_test.sh  # Original PD test script
│   └── run_ppd_test.sh            # PPD test script (supports pd/ppd modes)
├── src/                           # Python source code
│   ├── disagg_proxy_ppd.py        # PPD-aware proxy server
│   ├── compare_pd_ppd.py          # PD vs PPD comparison test
│   ├── bandwidth_test.py          # Bandwidth measurement
│   ├── multi_turn_test.py         # Multi-turn dialogue test
│   └── test_client.py             # Basic Python test client
├── results/                       # Test results (not tracked by git)
│   └── comparison_results.json    # Latest comparison test results
└── logs/                          # Log files (not tracked by git)
    ├── prefill_*.log             # Prefill instance log
    ├── decode_*.log              # Decode instance log
    └── proxy_*.log               # Proxy server log
```

## Key Metrics to Monitor

### From Decode Logs

```bash
# Monitor cache hit rates
grep "prefix cache" logs/decode_*.log

# Example output:
# Prefix cache hit rate: 64.4%        ← Local D cache reuse
# External prefix cache hit rate: 35.0% ← From P→D transfer
```

### From Bandwidth Test

```bash
python src/bandwidth_test.py --prompt-lengths 100 500 1000 --iterations 3
```

### From Multi-Turn Test

```bash
python src/multi_turn_test.py --turns 5 --input-tokens 100 --output-tokens 30
```

## PPD Mode (Implemented)

PPD (Prefill-Prefill-Decode direct) mode optimizes multi-turn dialogues by routing subsequent turns directly to the Decode machine, bypassing the Prefill machine and KV transfer overhead.

### Routing Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `pd` | Always P → D for every turn | Single-turn requests, baseline comparison |
| `ppd` | Turn 1: P → D, Turn 2+: D-direct | Multi-turn dialogues |

### How D-Direct Mode Works

```
PD Mode (every turn):
  Proxy → P (prefill all) → [KV Transfer] → D (decode) → Response

PPD Mode:
  Turn 1: Proxy → P (prefill) → [KV Transfer] → D (decode) → Response
  Turn 2+: Proxy → D directly (uses local prefix cache) → Response
```

### Implementation Details

1. **Modified P2pNcclConnector** (`vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`):
   - Added `is_pd_request()` to detect PD vs D-direct requests based on request_id format
   - D-direct requests (no special format) return 0 external tokens, forcing use of local prefix cache

2. **PPD-Aware Proxy** (`disagg_proxy_ppd.py`):
   - Tracks conversation state via prompt hash
   - Routes Turn 1 with special request_id format (triggers P→D flow)
   - Routes Turn 2+ with normal request_id (D-direct mode)

### Running PPD Tests

```bash
cd /net/projects2/ds3lab/zongzel/vllm/ppd

# Start with PPD mode
./scripts/run_ppd_test.sh ppd

# Run comparison test (tests both PD and PPD modes)
python src/compare_pd_ppd.py --turns 5 --output-tokens 30

# Switch modes dynamically
curl -X POST http://localhost:10001/mode/pd   # Switch to PD mode
curl -X POST http://localhost:10001/mode/ppd  # Switch to PPD mode

# Check current mode
curl http://localhost:10001/mode

# Clear conversation state
curl -X POST http://localhost:10001/conversations/clear
```

### Performance Results

5-turn conversation with 30 output tokens per turn:

| Turn | PD Mode (ms) | PPD Mode (ms) | Speedup | PPD Note |
|------|--------------|---------------|---------|----------|
| 1 | 144.4 | 140.3 | 1.03x | P→D |
| 2 | 407.2 | 240.6 | 1.69x | D-direct |
| 3 | 296.6 | 297.5 | 1.00x | D-direct |
| 4 | 305.6 | 294.4 | 1.04x | D-direct |
| 5 | 300.4 | 298.1 | 1.01x | D-direct |
| **Total** | **1454.2** | **1271.0** | **1.14x** | |

**Key Observations:**
- Turn 1: Similar latency (both use P→D flow)
- Turn 2: Significant speedup (1.69x) as PPD skips P→D overhead
- Turn 3+: Similar latency, D handles prefill efficiently with local cache
- Overall: **14% latency reduction** with PPD mode

### Cache Hit Rates

With PPD mode enabled:
- **Local Prefix Cache Hit Rate: 72.3%** (D reuses its own cache)
- **External Prefix Cache Hit Rate: 26.0%** (from P→D transfer in Turn 1)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mode` | GET | Get current routing mode |
| `/mode/pd` | POST | Switch to PD mode (always P→D) |
| `/mode/ppd` | POST | Switch to PPD mode (D-direct for Turn 2+) |
| `/conversations` | GET | Get conversation state (for debugging) |
| `/conversations/clear` | POST | Clear all conversation state |

## Future Work

1. **Selective KV Transfer:** Modify P2pNcclConnector to transfer only new KV blocks

2. **Real-time Metrics:** Add transfer timing instrumentation inside P2pNcclConnector

3. **Multi-D Scaling:** Test with multiple decode instances for load balancing

4. **Adaptive Routing:** Dynamically choose PD vs D-direct based on:
   - Cache hit rate on D
   - New prompt length
   - Network bandwidth vs. compute trade-off
