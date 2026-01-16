#!/bin/bash
# ============================================================================
# Optimizer Server Configuration V2: 1P + 1D_pure + 1pD + 1R
#
# New Architecture for optimal performance per objective:
#   GPU 0: P (Prefill only, kv_producer)
#          - Sends KV to GPU1 for TPOT optimization (PD mode)
#          - Sends KV to GPU2 for E2E optimization (PPD mode)
#
#   GPU 1: D_pure (Decode only, kv_consumer)
#          - Pure decode, NO append-prefill
#          - Best TPOT: no prefill interference
#          - Used for TPOT optimization
#
#   GPU 2: pD (Decode + append-prefill, kv_consumer)
#          - Receives KV from P, does append-prefill locally
#          - Supports KV cache reuse for multi-turn
#          - Used for E2E optimization
#
#   GPU 3: Replica (Standalone, no KV transfer)
#          - Fastest TTFT: no P→D overhead
#          - Used for TTFT optimization
#
# Routing by Objective:
#   TTFT → GPU3 (Replica)     - No KV transfer delay
#   TPOT → GPU0→GPU1 (PD)     - Pure decode, best TPOT
#   E2E  → GPU0→GPU2 (PPD)    - KV reuse for multi-turn
#
# Usage: ./start_optimizer_servers_v2.sh
# ============================================================================

set -e

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source common functions
source "$SCRIPT_DIR/common.sh" 2>/dev/null || true

# Check environment
check_environment || exit 1

# Check GPU availability (GPUs 0-3)
echo ""
check_gpu_availability "0,1,2,3"
gpu_status=$?
if [ $gpu_status -eq 1 ]; then
    exit 1
elif [ $gpu_status -eq 2 ]; then
    # Force cleanup requested
    force_cleanup
fi

# Configuration
MODEL_PATH="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"

# GPU 0: Prefill (P) - kv_producer
PREFILL_PORT=8100
KV_PORT_PREFILL=14579

# GPU 1: Pure Decode (D) - kv_consumer (NO append-prefill)
DECODE_PURE_PORT=8200
KV_PORT_DECODE_PURE=14580

# GPU 2: PPD Decode (pD) - kv_consumer (CAN append-prefill)
DECODE_PPD_PORT=8201
KV_PORT_DECODE_PPD=14581

# GPU 3: Replica (standalone, no KV transfer)
REPLICA_PORT=8300

# Proxies
OPTIMIZER_PROXY_PORT=10001
OPTIMIZER_CONTROL_PORT=30001

MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85

LOG_DIR="$PROJECT_DIR/logs"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/optimizer_v2_*.log 2>/dev/null || true

echo "=============================================="
echo "Starting Optimizer V2 Servers"
echo "=============================================="
echo "  GPU 0: P (Prefill)       - port $PREFILL_PORT"
echo "  GPU 1: D_pure (Decode)   - port $DECODE_PURE_PORT [TPOT]"
echo "  GPU 2: pD (PPD Decode)   - port $DECODE_PPD_PORT [E2E]"
echo "  GPU 3: Replica           - port $REPLICA_PORT [TTFT]"
echo "=============================================="

# NCCL settings for multi-GPU P2P (matching working 4gpu config)
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=32

# Cleanup existing
pkill -f "vllm serve.*$MODEL_PATH" 2>/dev/null || true
pkill -f "optimizer_proxy" 2>/dev/null || true
sleep 2

# [1/5] Start Optimizer Proxy
echo "[1/5] Starting Optimizer Proxy..."
python "$SRC_DIR/optimizer_proxy_v2.py" \
    --http-port $OPTIMIZER_PROXY_PORT \
    --zmq-port $OPTIMIZER_CONTROL_PORT \
    --prefill-port $PREFILL_PORT \
    --decode-pure-port $DECODE_PURE_PORT \
    --decode-ppd-port $DECODE_PPD_PORT \
    --replica-port $REPLICA_PORT \
    > "$LOG_DIR/optimizer_v2_proxy.log" 2>&1 &
sleep 2

# [2/5] Start Prefill (GPU 0) - kv_producer
echo "[2/5] Starting Prefill P (GPU 0, port $PREFILL_PORT)..."
PREFILL_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":1000000000,\"kv_port\":$KV_PORT_PREFILL,\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"$OPTIMIZER_CONTROL_PORT\",\"http_port\":\"$PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PREFILL_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$PREFILL_KV_CONFIG" \
    > "$LOG_DIR/optimizer_v2_prefill.log" 2>&1 &

# [3/5] Start Pure Decode (GPU 1) - kv_consumer (NO append-prefill capability)
echo "[3/5] Starting Pure Decode D (GPU 1, port $DECODE_PURE_PORT) [TPOT]..."
DECODE_PURE_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":10000000000,\"kv_port\":$KV_PORT_DECODE_PURE,\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"$OPTIMIZER_CONTROL_PORT\",\"http_port\":\"$DECODE_PURE_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $DECODE_PURE_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$DECODE_PURE_KV_CONFIG" \
    > "$LOG_DIR/optimizer_v2_decode_pure.log" 2>&1 &

# [4/5] Start PPD Decode (GPU 2) - kv_consumer (receives KV from P, does append-prefill locally)
echo "[4/5] Starting PPD Decode pD (GPU 2, port $DECODE_PPD_PORT) [E2E]..."
DECODE_PPD_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":10000000000,\"kv_port\":$KV_PORT_DECODE_PPD,\"kv_connector_extra_config\":{\"proxy_ip\":\"127.0.0.1\",\"proxy_port\":\"$OPTIMIZER_CONTROL_PORT\",\"http_port\":\"$DECODE_PPD_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $DECODE_PPD_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$DECODE_PPD_KV_CONFIG" \
    > "$LOG_DIR/optimizer_v2_decode_ppd.log" 2>&1 &

# [5/5] Start Replica (GPU 3) - standalone, no KV transfer
echo "[5/5] Starting Replica (GPU 3, port $REPLICA_PORT) [TTFT]..."
CUDA_VISIBLE_DEVICES=3 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $REPLICA_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/optimizer_v2_replica.log" 2>&1 &

# Wait for vLLM servers
echo ""
echo "Waiting for vLLM servers..."
MAX_WAIT=300
WAITED=0

for PORT in $PREFILL_PORT $DECODE_PURE_PORT $DECODE_PPD_PORT $REPLICA_PORT; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 2; WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for port $PORT"; exit 1; fi
        [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting for port $PORT... ($WAITED s)"
    done
    echo "  Port $PORT: READY"
done

sleep 3
echo "  Proxy: $(curl -s http://localhost:$OPTIMIZER_PROXY_PORT/status 2>/dev/null | head -c 100)"

echo ""
echo "=============================================="
echo "Optimizer V2 Servers Ready!"
echo "=============================================="
echo ""
echo "Architecture: 1P + 1D_pure + 1pD + 1R"
echo ""
echo "Routing by Objective:"
echo "  TTFT → GPU3 (Replica)     http://localhost:$REPLICA_PORT"
echo "  TPOT → GPU0→GPU1 (PD)     P:$PREFILL_PORT → D:$DECODE_PURE_PORT"
echo "  E2E  → GPU0→GPU2 (PPD)    P:$PREFILL_PORT → pD:$DECODE_PPD_PORT"
echo ""
echo "Optimizer Proxy: http://localhost:$OPTIMIZER_PROXY_PORT"
echo ""
echo "To stop: pkill -f 'vllm serve'; pkill -f optimizer_proxy"
echo "=============================================="
