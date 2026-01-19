#!/bin/bash
# ============================================================================
# Start Script for vLLM Configuration: 3P_1pD
#
# Architecture: 3P + 1pD
# 
#   GPU0: P (port 8100)
#   GPU1: P (port 8101)
#   GPU2: P (port 8102)
#   GPU3: pD (port 8200)
#
# Usage: ./start_3P_1pD.sh
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
    force_cleanup
fi

# Source unified configuration
source "$SCRIPT_DIR/config.sh"

# Server parameters
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85

LOG_DIR="$PROJECT_DIR/logs/3P_1pD"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log 2>/dev/null || true

echo "=============================================="
echo "Starting vLLM Configuration: 3P_1pD"
echo "Architecture: 3P + 1pD"
echo "=============================================="

# NCCL settings for multi-GPU P2P
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=32

# Cleanup existing processes using the comprehensive cleanup script
if [ -f "$SCRIPT_DIR/cleanup_all.sh" ]; then
    bash "$SCRIPT_DIR/cleanup_all.sh"
else
    echo "Cleaning up existing processes..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "comprehensive_proxy" 2>/dev/null || true
    pkill -9 -f "disagg_proxy" 2>/dev/null || true
    pkill -9 -f "simple_replica_proxy" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "$MODEL_NAME" 2>/dev/null || true
    sleep 10
fi

# Start Proxy
echo "[1/5] Starting Proxy..."
PROXY_PORT=10001
PROXY_CONTROL_PORT=30001
python "$SRC_DIR/comprehensive_proxy.py" \
    --config 3P_1pD \
    --http-port $PROXY_PORT \
    --zmq-port $PROXY_CONTROL_PORT \
    > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2

# Start Prefill (GPU 0)
echo "[2/5] Starting Prefill (GPU 0, port 8100)..."
KV_CONFIG='{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":1000000000,"kv_port":14579,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8100","send_type":"PUT_ASYNC"}}'
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8100 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG" \
    > "$LOG_DIR/prefill0.log" 2>&1 &

# Start Prefill (GPU 1)
echo "[3/5] Starting Prefill (GPU 1, port 8101)..."
KV_CONFIG='{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":1000000000,"kv_port":14580,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8101","send_type":"PUT_ASYNC"}}'
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8101 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG" \
    > "$LOG_DIR/prefill1.log" 2>&1 &

# Start Prefill (GPU 2)
echo "[4/5] Starting Prefill (GPU 2, port 8102)..."
KV_CONFIG='{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":1000000000,"kv_port":14581,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8102","send_type":"PUT_ASYNC"}}'
CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8102 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG" \
    > "$LOG_DIR/prefill2.log" 2>&1 &

# Start PPD-Decode (GPU 3)
echo "[5/5] Starting PPD-Decode (GPU 3, port 8200)..."
KV_CONFIG='{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":10000000000,"kv_port":14582,"kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"8200","send_type":"PUT_ASYNC"}}'
CUDA_VISIBLE_DEVICES=3 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8200 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$KV_CONFIG" \
    > "$LOG_DIR/ppd_decode3.log" 2>&1 &

# Wait for servers
echo ""
echo "Waiting for servers to be ready..."
MAX_WAIT=300
WAITED=0

for PORT in 8100 8101 8102 8200; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 2; WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for port $PORT"; exit 1; fi
        [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting for port $PORT... ($WAITED s)"
    done
    echo "  Port $PORT: READY"
done

sleep 3
echo "  Proxy: $(curl -s http://localhost:$PROXY_PORT/status)"

echo ""
echo "=============================================="
echo "Configuration 3P_1pD ready!"
echo "Architecture: 3P + 1pD"
echo "=============================================="
echo "P: http://localhost:8100 (GPU 0)"
echo "P: http://localhost:8101 (GPU 1)"
echo "P: http://localhost:8102 (GPU 2)"
echo "pD: http://localhost:8200 (GPU 3)"
echo ""
echo "Proxy: http://localhost:$PROXY_PORT"
echo ""
echo "To stop: ./scripts/server/stop_3P_1pD.sh"
echo "=============================================="
