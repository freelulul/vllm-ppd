#!/bin/bash
# ============================================================================
# 4-GPU Start Script for Replication Mode (4 Replicas)
#
# Architecture:
#   GPU 0: Worker 0 - port 8300
#   GPU 1: Worker 1 - port 8400
#   GPU 2: Worker 2 - port 8500
#   GPU 3: Worker 3 - port 8600
#
# Routing: Conversation-aware with hash-based assignment
#   Each conversation is pinned to one worker for cache affinity
#
# Usage: ./start_replication_servers_4gpu.sh
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

WORKER0_PORT=8300
WORKER1_PORT=8400
WORKER2_PORT=8500
WORKER3_PORT=8600
PROXY_PORT=10002

MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85

LOG_DIR="$PROJECT_DIR/logs"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/replication_*.log 2>/dev/null || true

echo "=============================================="
echo "Starting 4-GPU Replication Mode (4 Replicas)"
echo "=============================================="
echo "  4 standalone vLLM servers (NO KV transfer)"
echo "  Conversation-aware routing for cache affinity"
echo "=============================================="

export VLLM_LOGGING_LEVEL=INFO

# Cleanup existing replication servers
pkill -f "vllm serve.*:$WORKER0_PORT" 2>/dev/null || true
pkill -f "vllm serve.*:$WORKER1_PORT" 2>/dev/null || true
pkill -f "vllm serve.*:$WORKER2_PORT" 2>/dev/null || true
pkill -f "vllm serve.*:$WORKER3_PORT" 2>/dev/null || true
pkill -f "replication_proxy" 2>/dev/null || true
sleep 2

# Start Worker 0 (GPU 0)
echo "[1/5] Starting Worker 0 (GPU 0, port $WORKER0_PORT)..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $WORKER0_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replication_worker0.log" 2>&1 &

# Start Worker 1 (GPU 1)
echo "[2/5] Starting Worker 1 (GPU 1, port $WORKER1_PORT)..."
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $WORKER1_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replication_worker1.log" 2>&1 &

# Start Worker 2 (GPU 2)
echo "[3/5] Starting Worker 2 (GPU 2, port $WORKER2_PORT)..."
CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $WORKER2_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replication_worker2.log" 2>&1 &

# Start Worker 3 (GPU 3)
echo "[4/5] Starting Worker 3 (GPU 3, port $WORKER3_PORT)..."
CUDA_VISIBLE_DEVICES=3 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $WORKER3_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replication_worker3.log" 2>&1 &

# Wait for workers to be ready
echo ""
echo "Waiting for workers to be ready..."
MAX_WAIT=300
WAITED=0

for PORT in $WORKER0_PORT $WORKER1_PORT $WORKER2_PORT $WORKER3_PORT; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 2; WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for Worker at port $PORT"; exit 1; fi
        [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting for port $PORT... ($WAITED s)"
    done
    echo "  Worker (port $PORT): READY"
done

# Start Replication Proxy (4 workers)
echo "[5/5] Starting Replication Proxy..."
python "$SRC_DIR/simple_replica_proxy.py" \
    --workers "localhost:$WORKER0_PORT,localhost:$WORKER1_PORT,localhost:$WORKER2_PORT,localhost:$WORKER3_PORT" \
    --port $PROXY_PORT \
    > "$LOG_DIR/replication_proxy.log" 2>&1 &

sleep 3
echo "  Proxy: $(curl -s http://localhost:$PROXY_PORT/status 2>/dev/null || echo 'starting...')"

echo ""
echo "=============================================="
echo "4-GPU Replication Mode Ready!"
echo "=============================================="
echo "Proxy:    http://localhost:$PROXY_PORT"
echo "Worker 0: http://localhost:$WORKER0_PORT (GPU 0)"
echo "Worker 1: http://localhost:$WORKER1_PORT (GPU 1)"
echo "Worker 2: http://localhost:$WORKER2_PORT (GPU 2)"
echo "Worker 3: http://localhost:$WORKER3_PORT (GPU 3)"
echo ""
echo "To stop: ./scripts/server/stop_replication_servers_4gpu.sh"
echo "=============================================="
