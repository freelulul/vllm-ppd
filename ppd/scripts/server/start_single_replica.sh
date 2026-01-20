#!/bin/bash
# ============================================================================
# Start Script for Single Replica Server (Interference Benchmark)
#
# Architecture: Single vLLM server on GPU 0
# Purpose: Micro-benchmark for prefill-decode interference testing
#
# This script starts a single vLLM server without any proxy.
# The benchmark script communicates directly with the server.
#
# Usage: ./start_single_replica.sh [GPU_ID]
#        GPU_ID defaults to 0
# ============================================================================

set -e

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default GPU ID
GPU_ID=${1:-0}
PORT=8300

# Source common functions
source "$SCRIPT_DIR/common.sh" 2>/dev/null || true

# Check environment
check_environment || exit 1

# Source unified configuration
source "$SCRIPT_DIR/config.sh"

# Server parameters
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTIL=0.90  # Higher utilization for single server

LOG_DIR="$PROJECT_DIR/logs/single_replica"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log 2>/dev/null || true

echo "=============================================="
echo "Starting Single Replica Server"
echo "Purpose: Interference Benchmark"
echo "=============================================="
echo "GPU: $GPU_ID"
echo "Port: $PORT"
echo "Model: $MODEL_NAME"
echo "=============================================="

# Cleanup existing processes
echo ""
echo "Cleaning up existing processes..."
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "$MODEL_NAME" 2>/dev/null || true
sleep 5

# Check if port is free
if lsof -i:$PORT > /dev/null 2>&1; then
    echo "ERROR: Port $PORT is still in use"
    exit 1
fi

# Check GPU availability
echo ""
echo "Checking GPU $GPU_ID availability..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null | head -1)
if [ "$GPU_MEM" -gt 1000 ]; then
    echo "WARNING: GPU $GPU_ID has $GPU_MEM MiB memory in use"
    echo "Attempting cleanup..."
    sleep 5
fi

# Start single replica server
echo ""
echo "Starting vLLM server on GPU $GPU_ID..."
CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo ""
echo "Waiting for server to be ready..."
MAX_WAIT=${MAX_WAIT:-300}  # Can be overridden via environment variable
WAITED=0

while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Timeout waiting for server"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting... ($WAITED s)"
done

echo "  Server ready! (waited $WAITED s)"

# Verify server
echo ""
echo "Verifying server..."
MODELS=$(curl -s "http://localhost:$PORT/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
echo "  Model: $MODELS"

echo ""
echo "=============================================="
echo "Single Replica Server Ready!"
echo "=============================================="
echo "URL: http://localhost:$PORT"
echo "GPU: $GPU_ID"
echo "Log: $LOG_DIR/server.log"
echo ""
echo "To run interference benchmark:"
echo "  python scripts/tests/interference_benchmark.py --server-url http://localhost:$PORT"
echo ""
echo "To stop:"
echo "  pkill -f 'vllm serve'"
echo "=============================================="
