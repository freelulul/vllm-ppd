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
./run_pd_separation_test.sh
```

The script automatically cleans old logs and keeps only the latest run.

### Run Bandwidth Test

After the PD separation test is running:

```bash
python bandwidth_test.py --prompt-lengths 100 500 1000 --iterations 3
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

## Future Work

1. Implement PPD (Prefill-Prefill-Decode) decision logic
2. Add multi-turn dialogue KV cache transfer optimization
3. Extend P2pNcclConnector with detailed transfer metrics
4. Test performance across different prompt lengths and cache rates
