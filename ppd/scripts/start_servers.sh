#!/bin/bash
# ============================================================================
# Quick Start Script for vLLM PD/PPD Servers
# Usage: ./start_servers.sh [pd|ppd] [--benchmark]
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

# Check for conflicting vllm installation
if [ -f "$HOME/.local/bin/vllm" ] && head -1 "$HOME/.local/bin/vllm" 2>/dev/null | grep -q "/opt/conda/bin/python"; then
    echo "ERROR: Found conflicting ~/.local/bin/vllm using base Python"
    echo "  This will cause --kv-transfer-config to fail"
    echo "  Fix: rm ~/.local/bin/vllm"
    echo "  Or run: ./scripts/fix_env.sh"
    exit 1
fi

# Check quart
if ! python -c "import quart" 2>/dev/null; then
    echo "ERROR: quart not installed"
    echo "  Fix: pip install quart"
    echo "  Or run: ./scripts/fix_env.sh"
    exit 1
fi

MODE="${1:-ppd}"
BENCHMARK_MODE=false

# Check for --benchmark flag
for arg in "$@"; do
    if [ "$arg" == "--benchmark" ]; then
        BENCHMARK_MODE=true
    fi
done
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL_PATH="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=10001
PROXY_CONTROL_PORT=30001
KV_PORT_PREFILL=14579
KV_PORT_DECODE=14580
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85

LOG_DIR="$PROJECT_DIR/logs"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"

# Clean old logs
rm -f "$LOG_DIR"/*.log 2>/dev/null || true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Starting vLLM PD/PPD Servers"
echo "  Mode: $MODE"
echo "  Benchmark: $BENCHMARK_MODE"
echo "=============================================="

# Environment
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=INFO

# Cleanup existing
pkill -f "vllm serve.*$MODEL_PATH" 2>/dev/null || true
pkill -f "disagg_proxy" 2>/dev/null || true
sleep 2

# Start Proxy
echo "[1/3] Starting Proxy..."
if [ "$BENCHMARK_MODE" = true ]; then
    PROXY_SCRIPT="$SRC_DIR/disagg_proxy_benchmark.py"
    echo "  Using benchmark proxy with metrics collection"
else
    PROXY_SCRIPT="$SRC_DIR/disagg_proxy_ppd.py"
fi

python "$PROXY_SCRIPT" \
    --mode "$MODE" \
    --http-port $PROXY_PORT \
    --zmq-port $PROXY_CONTROL_PORT \
    > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2

# Start Prefill (GPU 0)
echo "[2/3] Starting Prefill server (GPU 0)..."
PREFILL_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":1000000000,\"kv_port\":$KV_PORT_PREFILL,\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_CONTROL_PORT\",\"http_port\":\"$PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PREFILL_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --kv-transfer-config "$PREFILL_KV_CONFIG" \
    > "$LOG_DIR/prefill.log" 2>&1 &

# Start Decode (GPU 1)
echo "[3/3] Starting Decode server (GPU 1)..."
DECODE_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":10000000000,\"kv_port\":$KV_PORT_DECODE,\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_CONTROL_PORT\",\"http_port\":\"$DECODE_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $DECODE_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --kv-transfer-config "$DECODE_KV_CONFIG" \
    > "$LOG_DIR/decode.log" 2>&1 &

# Wait for servers
echo ""
echo "Waiting for servers to be ready..."
MAX_WAIT=300
WAITED=0

while ! curl -s "http://localhost:$PREFILL_PORT/v1/models" > /dev/null 2>&1; do
    sleep 2; WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for Prefill"; exit 1; fi
    [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting... ($WAITED s)"
done
echo "  Prefill: READY"

while ! curl -s "http://localhost:$DECODE_PORT/v1/models" > /dev/null 2>&1; do
    sleep 2; WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for Decode"; exit 1; fi
done
echo "  Decode: READY"

sleep 3
echo "  Proxy: $(curl -s http://localhost:$PROXY_PORT/mode)"

echo ""
echo "=============================================="
echo "All servers ready! Mode: $MODE"
echo "=============================================="
echo "Proxy:   http://localhost:$PROXY_PORT"
echo "Prefill: http://localhost:$PREFILL_PORT"
echo "Decode:  http://localhost:$DECODE_PORT"
echo ""
if [ "$BENCHMARK_MODE" = true ]; then
    echo "Benchmark endpoints:"
    echo "  GET  /metrics         - Get all metrics"
    echo "  GET  /metrics/summary - Get metrics summary"
    echo "  POST /metrics/clear   - Clear metrics"
    echo ""
    echo "Run benchmark:"
    echo "  python src/comprehensive_benchmark.py --list                   # List all 24 configs"
    echo "  python src/comprehensive_benchmark.py                          # Run all configs"
    echo "  python src/comprehensive_benchmark.py --runs 5 --warmup 1      # Multi-run mode"
fi
echo ""
echo "To stop: ./scripts/stop_servers.sh"
echo "=============================================="
