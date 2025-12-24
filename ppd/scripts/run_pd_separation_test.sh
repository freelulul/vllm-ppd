#!/bin/bash
# ============================================================================
# vLLM PD Separation Test Script (using P2pNcclConnector with xpyd proxy)
# This script runs disaggregated prefill/decode on two H100 GPUs using TCP
# ============================================================================

set -e

# Configuration
MODEL_PATH="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=10001       # xpyd proxy uses 10001
PROXY_CONTROL_PORT=30001
KV_PORT_PREFILL=14579
KV_PORT_DECODE=14580

# Maximum model length (adjust based on your memory)
MAX_MODEL_LEN=2048
GPU_MEMORY_UTIL=0.85

# Directory setup
SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# ============================================================================
# Auto-clean old logs - keep only the latest run
# ============================================================================
echo "[CLEANUP] Removing old log files..."
rm -f "$LOG_DIR"/prefill_*.log 2>/dev/null || true
rm -f "$LOG_DIR"/decode_*.log 2>/dev/null || true
rm -f "$LOG_DIR"/proxy_*.log 2>/dev/null || true
echo "[CLEANUP] Old logs removed"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "vLLM PD Separation Test (P2pNcclConnector)"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Prefill Port: $PREFILL_PORT (GPU 0)"
echo "Decode Port: $DECODE_PORT (GPU 1)"
echo "Proxy Port: $PROXY_PORT"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "Log Directory: $LOG_DIR"
echo "=============================================="

# ============================================================================
# Environment Setup - Force TCP transport (disable RDMA/NVLink for KV transfer)
# ============================================================================
export NCCL_IB_DISABLE=1           # Disable InfiniBand
export NCCL_NET=Socket             # Force Socket (TCP) transport
export NCCL_DEBUG=WARN             # Set NCCL debug level
export VLLM_LOGGING_LEVEL=INFO     # vLLM logging level

echo ""
echo "[ENV] NCCL_IB_DISABLE=1 (InfiniBand disabled)"
echo "[ENV] NCCL_NET=Socket (TCP transport forced)"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "[CLEANUP] Stopping all processes..."
    pkill -f "vllm serve.*$MODEL_PATH" 2>/dev/null || true
    pkill -f "disagg_proxy_p2p_nccl_xpyd" 2>/dev/null || true
    sleep 2
    echo "[CLEANUP] Done"
}

# Trap signals for cleanup
trap cleanup EXIT INT TERM

# Kill any existing vllm processes
echo "[INIT] Cleaning up existing processes..."
cleanup
sleep 2

# ============================================================================
# Start xpyd Proxy Server FIRST (it needs to be running for service discovery)
# ============================================================================
echo ""
echo "[PROXY] Starting P2pNccl xpyd proxy server..."

PROXY_SCRIPT="/net/projects2/ds3lab/zongzel/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py"

python "$PROXY_SCRIPT" \
    > "$LOG_DIR/proxy_${TIMESTAMP}.log" 2>&1 &

PROXY_PID=$!
echo "[PROXY] Started with PID: $PROXY_PID"
echo "[PROXY] Log: $LOG_DIR/proxy_${TIMESTAMP}.log"
echo "[PROXY] Listening on port $PROXY_PORT (HTTP) and $PROXY_CONTROL_PORT (ZMQ service discovery)"

sleep 3

# ============================================================================
# Start Prefill Instance (GPU 0 - KV Producer)
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
echo "[PREFILL] Log: $LOG_DIR/prefill_${TIMESTAMP}.log"

# ============================================================================
# Start Decode Instance (GPU 1 - KV Consumer)
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
echo "[DECODE] Log: $LOG_DIR/decode_${TIMESTAMP}.log"

# ============================================================================
# Wait for servers to be ready
# ============================================================================
wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=300  # 5 minutes timeout
    local waited=0

    echo "[WAIT] Waiting for $name server on port $port..."
    while ! curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "[ERROR] Timeout waiting for $name server"
            echo "[ERROR] Check log: $LOG_DIR/${name,,}_${TIMESTAMP}.log"
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

# Give some time for service discovery registration
echo ""
echo "[WAIT] Waiting for service discovery registration..."
sleep 5

# Check proxy logs for registration
echo "[CHECK] Checking proxy for registered instances..."
grep -E "Add|HTTP" "$LOG_DIR/proxy_${TIMESTAMP}.log" | tail -5 || echo "No registrations yet"

# ============================================================================
# Run Test Requests
# ============================================================================
echo ""
echo "=============================================="
echo "Running Test Requests"
echo "=============================================="

# Test 1: Simple short prompt
echo ""
echo "[TEST 1] Short prompt test..."
START_TIME=$(date +%s.%N)
RESPONSE1=$(curl -s -X POST "http://localhost:$PROXY_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "prompt": "San Francisco is a",
        "max_tokens": 20,
        "temperature": 0
    }')
END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
echo "Response 1 (${ELAPSED}s): $RESPONSE1"

# Test 2: Another short prompt
echo ""
echo "[TEST 2] Another short prompt test..."
START_TIME=$(date +%s.%N)
RESPONSE2=$(curl -s -X POST "http://localhost:$PROXY_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "temperature": 0
    }')
END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
echo "Response 2 (${ELAPSED}s): $RESPONSE2"

# Test 3: Longer prompt
echo ""
echo "[TEST 3] Longer prompt test..."
START_TIME=$(date +%s.%N)
RESPONSE3=$(curl -s -X POST "http://localhost:$PROXY_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "prompt": "In the field of artificial intelligence, large language models have become increasingly important. These models are trained on vast amounts of text data and can generate human-like responses. The key innovation is",
        "max_tokens": 50,
        "temperature": 0
    }')
END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
echo "Response 3 (${ELAPSED}s): $RESPONSE3"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "Test Complete!"
echo "=============================================="
echo ""
echo "Log files:"
echo "  - Prefill: $LOG_DIR/prefill_${TIMESTAMP}.log"
echo "  - Decode:  $LOG_DIR/decode_${TIMESTAMP}.log"
echo "  - Proxy:   $LOG_DIR/proxy_${TIMESTAMP}.log"
echo ""
echo "To check logs:"
echo "  tail -f $LOG_DIR/prefill_${TIMESTAMP}.log"
echo "  tail -f $LOG_DIR/decode_${TIMESTAMP}.log"
echo "  tail -f $LOG_DIR/proxy_${TIMESTAMP}.log"
echo ""
echo "Servers are still running. Press Ctrl+C to stop."
echo ""

# Keep running until interrupted
wait
