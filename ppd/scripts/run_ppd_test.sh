#!/bin/bash
# ============================================================================
# vLLM PPD Test Script
# Tests both PD mode and PPD mode for comparison
# ============================================================================

set -e

# Configuration
MODEL_PATH="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=10001
PROXY_CONTROL_PORT=30001
KV_PORT_PREFILL=14579
KV_PORT_DECODE=14580

MAX_MODEL_LEN=2048
GPU_MEMORY_UTIL=0.85

# Routing mode: pd or ppd
MODE="${1:-ppd}"

# Directory setup
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"

# Auto-clean old logs
echo "[CLEANUP] Removing old log files..."
rm -f "$LOG_DIR"/prefill_*.log 2>/dev/null || true
rm -f "$LOG_DIR"/decode_*.log 2>/dev/null || true
rm -f "$LOG_DIR"/proxy_*.log 2>/dev/null || true
echo "[CLEANUP] Old logs removed"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "vLLM PPD Test (Mode: $MODE)"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Prefill Port: $PREFILL_PORT (GPU 0)"
echo "Decode Port: $DECODE_PORT (GPU 1)"
echo "Proxy Port: $PROXY_PORT"
echo "Routing Mode: $MODE"
echo "  - pd:  Always P -> D"
echo "  - ppd: First turn P -> D, rest direct to D"
echo "=============================================="

# Environment Setup - Force TCP transport
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=INFO

echo ""
echo "[ENV] NCCL_IB_DISABLE=1 (InfiniBand disabled)"
echo "[ENV] NCCL_NET=Socket (TCP transport forced)"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "[CLEANUP] Stopping all processes..."
    pkill -f "vllm serve.*$MODEL_PATH" 2>/dev/null || true
    pkill -f "disagg_proxy" 2>/dev/null || true
    sleep 2
    echo "[CLEANUP] Done"
}

trap cleanup EXIT INT TERM

# Kill existing processes
echo "[INIT] Cleaning up existing processes..."
cleanup
sleep 2

# ============================================================================
# Start PPD Proxy Server
# ============================================================================
echo ""
echo "[PROXY] Starting PPD Proxy (mode: $MODE)..."

python "$SRC_DIR/disagg_proxy_ppd.py" \
    --mode "$MODE" \
    --http-port $PROXY_PORT \
    --zmq-port $PROXY_CONTROL_PORT \
    > "$LOG_DIR/proxy_${TIMESTAMP}.log" 2>&1 &

PROXY_PID=$!
echo "[PROXY] Started with PID: $PROXY_PID"
echo "[PROXY] Log: $LOG_DIR/proxy_${TIMESTAMP}.log"

sleep 3

# ============================================================================
# Start Prefill Instance (GPU 0)
# ============================================================================
echo ""
echo "[PREFILL] Starting Prefill instance on GPU 0..."

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PREFILL_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code \
    --disable-log-requests \
    --kv-transfer-config "{
        \"kv_connector\": \"P2pNcclConnector\",
        \"kv_role\": \"kv_producer\",
        \"kv_buffer_size\": \"1e9\",
        \"kv_port\": \"$KV_PORT_PREFILL\",
        \"kv_connector_extra_config\": {
            \"proxy_ip\": \"0.0.0.0\",
            \"proxy_port\": \"$PROXY_CONTROL_PORT\",
            \"http_port\": \"$PREFILL_PORT\",
            \"send_type\": \"PUT_ASYNC\"
        }
    }" \
    > "$LOG_DIR/prefill_${TIMESTAMP}.log" 2>&1 &

PREFILL_PID=$!
echo "[PREFILL] Started with PID: $PREFILL_PID"

# ============================================================================
# Start Decode Instance (GPU 1)
# ============================================================================
echo ""
echo "[DECODE] Starting Decode instance on GPU 1..."

CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $DECODE_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code \
    --disable-log-requests \
    --kv-transfer-config "{
        \"kv_connector\": \"P2pNcclConnector\",
        \"kv_role\": \"kv_consumer\",
        \"kv_buffer_size\": \"1e10\",
        \"kv_port\": \"$KV_PORT_DECODE\",
        \"kv_connector_extra_config\": {
            \"proxy_ip\": \"0.0.0.0\",
            \"proxy_port\": \"$PROXY_CONTROL_PORT\",
            \"http_port\": \"$DECODE_PORT\",
            \"send_type\": \"PUT_ASYNC\"
        }
    }" \
    > "$LOG_DIR/decode_${TIMESTAMP}.log" 2>&1 &

DECODE_PID=$!
echo "[DECODE] Started with PID: $DECODE_PID"

# ============================================================================
# Wait for servers
# ============================================================================
wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=300
    local waited=0

    echo "[WAIT] Waiting for $name server on port $port..."
    while ! curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "[ERROR] Timeout waiting for $name server"
            tail -50 "$LOG_DIR/${name,,}_${TIMESTAMP}.log"
            exit 1
        fi
        if [ $((waited % 30)) -eq 0 ]; then
            echo "[WAIT] Still waiting for $name... ($waited seconds)"
        fi
    done
    echo "[READY] $name server is ready (took $waited seconds)"
}

echo ""
wait_for_server $PREFILL_PORT "Prefill"
wait_for_server $DECODE_PORT "Decode"

echo ""
echo "[WAIT] Waiting for service discovery..."
sleep 5

# Check mode
echo ""
echo "[CHECK] Current proxy mode:"
curl -s "http://localhost:$PROXY_PORT/mode" | python3 -m json.tool 2>/dev/null || echo "Could not get mode"

echo ""
echo "=============================================="
echo "Servers Ready! Mode: $MODE"
echo "=============================================="
echo ""
echo "Test commands:"
echo "  # Run multi-turn comparison test"
echo "  python src/compare_pd_ppd.py"
echo ""
echo "  # Check current mode"
echo "  curl http://localhost:$PROXY_PORT/mode"
echo ""
echo "  # Switch mode (while running)"
echo "  curl -X POST http://localhost:$PROXY_PORT/mode/pd"
echo "  curl -X POST http://localhost:$PROXY_PORT/mode/ppd"
echo ""
echo "  # Clear conversation state"
echo "  curl -X POST http://localhost:$PROXY_PORT/conversations/clear"
echo ""
echo "Log files:"
echo "  - Prefill: $LOG_DIR/prefill_${TIMESTAMP}.log"
echo "  - Decode:  $LOG_DIR/decode_${TIMESTAMP}.log"
echo "  - Proxy:   $LOG_DIR/proxy_${TIMESTAMP}.log"
echo ""
echo "Press Ctrl+C to stop."
echo ""

wait
