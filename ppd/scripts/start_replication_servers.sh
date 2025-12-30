#!/bin/bash
# ============================================================================
# Start Script for Replication Mode (Data Parallelism Baseline)
#
# This starts 2 STANDALONE vLLM servers (no KV transfer config).
# Each GPU acts as an independent worker.
#
# Usage: ./start_replication_servers.sh
# ============================================================================

set -e

# Environment checks - verify we're using the right Python
PYTHON_PATH=$(which python)
if [[ "$PYTHON_PATH" != *"vllm-ppd"* ]]; then
    echo "ERROR: Please activate vllm-ppd first:"
    echo "  conda activate vllm-ppd"
    echo "  Current python: $PYTHON_PATH"
    exit 1
fi

# Check quart
if ! python -c "import quart" 2>/dev/null; then
    echo "ERROR: quart not installed"
    echo "  Fix: pip install quart"
    exit 1
fi

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL_PATH="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
WORKER0_PORT=8300  # GPU 0 standalone server
WORKER1_PORT=8400  # GPU 1 standalone server
PROXY_PORT=10002   # Replication proxy port (different from PD proxy)
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85

LOG_DIR="$PROJECT_DIR/logs"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"

# Clean old logs
rm -f "$LOG_DIR"/replication_*.log 2>/dev/null || true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Starting Replication Mode (Data Parallelism)"
echo "=============================================="
echo "  2 standalone vLLM servers (NO KV transfer)"
echo "  Worker 0 (GPU 0): http://localhost:$WORKER0_PORT"
echo "  Worker 1 (GPU 1): http://localhost:$WORKER1_PORT"
echo "  Proxy: http://localhost:$PROXY_PORT"
echo "=============================================="

# Environment
export VLLM_LOGGING_LEVEL=INFO

# Cleanup existing replication servers
pkill -f "vllm serve.*:$WORKER0_PORT" 2>/dev/null || true
pkill -f "vllm serve.*:$WORKER1_PORT" 2>/dev/null || true
pkill -f "replication_proxy" 2>/dev/null || true
sleep 2

# Start Worker 0 (GPU 0) - Standalone vLLM server
echo "[1/3] Starting Worker 0 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $WORKER0_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    > "$LOG_DIR/replication_worker0.log" 2>&1 &

# Start Worker 1 (GPU 1) - Standalone vLLM server
echo "[2/3] Starting Worker 1 (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $WORKER1_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    > "$LOG_DIR/replication_worker1.log" 2>&1 &

# Wait for workers to be ready
echo ""
echo "Waiting for workers to be ready..."
MAX_WAIT=300
WAITED=0

while ! curl -s "http://localhost:$WORKER0_PORT/v1/models" > /dev/null 2>&1; do
    sleep 2; WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for Worker 0"; exit 1; fi
    [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting... ($WAITED s)"
done
echo "  Worker 0: READY"

while ! curl -s "http://localhost:$WORKER1_PORT/v1/models" > /dev/null 2>&1; do
    sleep 2; WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for Worker 1"; exit 1; fi
done
echo "  Worker 1: READY"

# Start Replication Proxy with prefix-aware routing
echo "[3/3] Starting Replication Proxy..."
python "$SRC_DIR/replication_proxy.py" \
    --worker0 "localhost:$WORKER0_PORT" \
    --worker1 "localhost:$WORKER1_PORT" \
    --port $PROXY_PORT \
    > "$LOG_DIR/replication_proxy.log" 2>&1 &

sleep 3
echo "  Proxy: $(curl -s http://localhost:$PROXY_PORT/status 2>/dev/null || echo 'starting...')"

echo ""
echo "=============================================="
echo "Replication Mode Ready!"
echo "=============================================="
echo "Proxy:    http://localhost:$PROXY_PORT"
echo "Worker 0: http://localhost:$WORKER0_PORT (GPU 0)"
echo "Worker 1: http://localhost:$WORKER1_PORT (GPU 1)"
echo ""
echo "Benchmark endpoints:"
echo "  GET  /status         - Get proxy status"
echo "  GET  /metrics        - Get all metrics"
echo "  POST /metrics/clear  - Clear metrics"
echo ""
echo "Run benchmark:"
echo "  python src/replication_benchmark.py"
echo ""
echo "To stop: ./scripts/stop_replication_servers.sh"
echo "=============================================="
