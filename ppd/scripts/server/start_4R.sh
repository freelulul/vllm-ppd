#!/bin/bash
# ============================================================================
# Start Script for vLLM Configuration: 4R
#
# Architecture: 4R
# 
#   GPU0: R (port 8300)
#   GPU1: R (port 8400)
#   GPU2: R (port 8500)
#   GPU3: R (port 8600)
#
# Usage: ./start_4R.sh
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

LOG_DIR="$PROJECT_DIR/logs/4R"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log 2>/dev/null || true

echo "=============================================="
echo "Starting vLLM Configuration: 4R"
echo "Architecture: 4R"
echo "=============================================="

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

# Start Comprehensive Proxy (unified proxy for all configs)
echo "[1/5] Starting Comprehensive Proxy..."
python "$SRC_DIR/comprehensive_proxy.py" \
    --config 4R \
    --http-port $PROXY_HTTP_PORT \
    --zmq-port $PROXY_ZMQ_PORT \
    > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2

# Start Replica 0 (GPU 0)
echo "[2/5] Starting Replica (GPU 0, port 8300)..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8300 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replica0.log" 2>&1 &

# Start Replica 1 (GPU 1)
echo "[3/5] Starting Replica (GPU 1, port 8400)..."
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8400 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replica1.log" 2>&1 &

# Start Replica 2 (GPU 2)
echo "[4/5] Starting Replica (GPU 2, port 8500)..."
CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8500 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replica2.log" 2>&1 &

# Start Replica 3 (GPU 3)
echo "[5/5] Starting Replica (GPU 3, port 8600)..."
CUDA_VISIBLE_DEVICES=3 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8600 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replica3.log" 2>&1 &

# Wait for servers
echo ""
echo "Waiting for servers to be ready..."
MAX_WAIT=300
WAITED=0

for PORT in 8300 8400 8500 8600; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 2; WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for port $PORT"; exit 1; fi
        [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting for port $PORT... ($WAITED s)"
    done
    echo "  Port $PORT: READY"
done

sleep 3

# Check proxy status
echo "  Proxy: $(curl -s http://localhost:$PROXY_HTTP_PORT/status)"

echo ""
echo "=============================================="
echo "Configuration 4R ready!"
echo "Architecture: 4R"
echo "=============================================="
echo "R: http://localhost:8300 (GPU 0)"
echo "R: http://localhost:8400 (GPU 1)"
echo "R: http://localhost:8500 (GPU 2)"
echo "R: http://localhost:8600 (GPU 3)"
echo ""
echo "Proxy: http://localhost:$PROXY_HTTP_PORT"
echo ""
echo "To stop: ./scripts/server/stop.sh"
echo "=============================================="
