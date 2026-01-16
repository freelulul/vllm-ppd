#!/bin/bash
# ============================================================================
# 4-GPU Start Script for vLLM PD/PPD Servers (2P + 2D)
#
# Architecture:
#   GPU 0: Prefill 0 (P0) - port 8100
#   GPU 1: Prefill 1 (P1) - port 8101
#   GPU 2: Decode 0 (D0)  - port 8200
#   GPU 3: Decode 1 (D1)  - port 8201
#
# Routing: Round-robin 1P1D pairing
#   Request 1: P0 → D0
#   Request 2: P1 → D1
#   Request 3: P0 → D0
#   ...
#
# Usage: ./start_servers_4gpu.sh [pd|ppd]
# ============================================================================

set -e

MODE="${1:-ppd}"

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

# Prefill servers (GPU 0, 1)
PREFILL0_PORT=8100
PREFILL1_PORT=8101
KV_PORT_PREFILL0=14579
KV_PORT_PREFILL1=14581

# Decode servers (GPU 2, 3)
DECODE0_PORT=8200
DECODE1_PORT=8201
KV_PORT_DECODE0=14580
KV_PORT_DECODE1=14582

# Proxy
PROXY_PORT=10001
PROXY_CONTROL_PORT=30001

MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85

LOG_DIR="$PROJECT_DIR/logs"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log 2>/dev/null || true

echo "=============================================="
echo "Starting vLLM 4-GPU PD/PPD Servers (2P + 2D)"
echo "=============================================="
echo "  Mode: $MODE"
echo "  Architecture: 2 Prefill + 2 Decode"
echo "  KV Transfer: NVLink P2P"
echo "=============================================="

# NCCL settings for multi-GPU P2P
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=32

# Cleanup existing
pkill -f "vllm serve.*$MODEL_PATH" 2>/dev/null || true
pkill -f "disagg_proxy" 2>/dev/null || true
sleep 2

# Start Proxy (4-GPU version with cache affinity)
echo "[1/5] Starting Proxy (4-GPU with cache affinity)..."
python "$SRC_DIR/disagg_proxy_ppd_4gpu.py" \
    --mode "$MODE" \
    --http-port $PROXY_PORT \
    --zmq-port $PROXY_CONTROL_PORT \
    > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2

# Start Prefill 0 (GPU 0)
echo "[2/5] Starting Prefill 0 (GPU 0, port $PREFILL0_PORT)..."
PREFILL0_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":1000000000,\"kv_port\":$KV_PORT_PREFILL0,\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_CONTROL_PORT\",\"http_port\":\"$PREFILL0_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PREFILL0_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$PREFILL0_KV_CONFIG" \
    > "$LOG_DIR/prefill0.log" 2>&1 &

# Start Prefill 1 (GPU 1)
echo "[3/5] Starting Prefill 1 (GPU 1, port $PREFILL1_PORT)..."
PREFILL1_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":1000000000,\"kv_port\":$KV_PORT_PREFILL1,\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_CONTROL_PORT\",\"http_port\":\"$PREFILL1_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PREFILL1_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$PREFILL1_KV_CONFIG" \
    > "$LOG_DIR/prefill1.log" 2>&1 &

# Start Decode 0 (GPU 2)
echo "[4/5] Starting Decode 0 (GPU 2, port $DECODE0_PORT)..."
DECODE0_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":10000000000,\"kv_port\":$KV_PORT_DECODE0,\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_CONTROL_PORT\",\"http_port\":\"$DECODE0_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $DECODE0_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$DECODE0_KV_CONFIG" \
    > "$LOG_DIR/decode0.log" 2>&1 &

# Start Decode 1 (GPU 3)
echo "[5/5] Starting Decode 1 (GPU 3, port $DECODE1_PORT)..."
DECODE1_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":10000000000,\"kv_port\":$KV_PORT_DECODE1,\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_CONTROL_PORT\",\"http_port\":\"$DECODE1_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=3 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $DECODE1_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$DECODE1_KV_CONFIG" \
    > "$LOG_DIR/decode1.log" 2>&1 &

# Wait for servers
echo ""
echo "Waiting for servers to be ready..."
MAX_WAIT=300
WAITED=0

for PORT in $PREFILL0_PORT $PREFILL1_PORT $DECODE0_PORT $DECODE1_PORT; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 2; WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for port $PORT"; exit 1; fi
        [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting for port $PORT... ($WAITED s)"
    done
    echo "  Port $PORT: READY"
done

sleep 3
echo "  Proxy: $(curl -s http://localhost:$PROXY_PORT/mode)"

echo ""
echo "=============================================="
echo "All 4-GPU servers ready! Mode: $MODE"
echo "=============================================="
echo "Proxy:     http://localhost:$PROXY_PORT"
echo "Prefill 0: http://localhost:$PREFILL0_PORT (GPU 0)"
echo "Prefill 1: http://localhost:$PREFILL1_PORT (GPU 1)"
echo "Decode 0:  http://localhost:$DECODE0_PORT (GPU 2)"
echo "Decode 1:  http://localhost:$DECODE1_PORT (GPU 3)"
echo ""
echo "Routing: Round-robin 1P1D pairing"
echo "  Request 1: P0 → D0"
echo "  Request 2: P1 → D1"
echo "  Request 3: P0 → D0"
echo "  ..."
echo ""
echo "To stop: ./scripts/server/stop_servers_4gpu.sh"
echo "=============================================="
