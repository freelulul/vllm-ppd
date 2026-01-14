#!/bin/bash
# ============================================================================
# Optimizer Server Configuration: 1P + 1D + 2 Replica
#
# Architecture:
#   GPU 0: Prefill (P)     - port 8100
#   GPU 1: Decode (D)      - port 8200
#   GPU 2: Replica 0 (R0)  - port 8300
#   GPU 3: Replica 1 (R1)  - port 8400
#
# Proxies:
#   PD/PPD Proxy: port 10001
#   Replica Proxy: port 10002
#
# Usage: ./start_optimizer_servers.sh
# ============================================================================

set -e

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Configuration
MODEL_PATH="/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B"

# PD/PPD ports (GPU 0, 1)
PREFILL_PORT=8100
DECODE_PORT=8200
KV_PORT_PREFILL=14579
KV_PORT_DECODE=14580
PD_PROXY_PORT=10001
PD_CONTROL_PORT=30001

# Replica ports (GPU 2, 3)
REPLICA0_PORT=8300
REPLICA1_PORT=8400
REPLICA_PROXY_PORT=10002

MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85

LOG_DIR="$PROJECT_DIR/logs"
SRC_DIR="$PROJECT_DIR/src"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log 2>/dev/null || true

echo "=============================================="
echo "Starting Optimizer Servers (1P + 1D + 2R)"
echo "=============================================="

# NCCL settings
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0

# Cleanup existing
pkill -f "vllm serve.*$MODEL_PATH" 2>/dev/null || true
pkill -f "disagg_proxy" 2>/dev/null || true
pkill -f "replication_proxy" 2>/dev/null || true
sleep 2

echo "[1/6] Starting PD/PPD Proxy..."
python "$SRC_DIR/disagg_proxy_ppd_4gpu.py" \
    --mode ppd \
    --http-port $PD_PROXY_PORT \
    --zmq-port $PD_CONTROL_PORT \
    > "$LOG_DIR/pd_proxy.log" 2>&1 &
sleep 2

echo "[2/6] Starting Prefill (GPU 0, port $PREFILL_PORT)..."
PREFILL_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":1000000000,\"kv_port\":$KV_PORT_PREFILL,\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PD_CONTROL_PORT\",\"http_port\":\"$PREFILL_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PREFILL_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$PREFILL_KV_CONFIG" \
    > "$LOG_DIR/prefill.log" 2>&1 &

echo "[3/6] Starting Decode (GPU 1, port $DECODE_PORT)..."
DECODE_KV_CONFIG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":10000000000,\"kv_port\":$KV_PORT_DECODE,\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PD_CONTROL_PORT\",\"http_port\":\"$DECODE_PORT\",\"send_type\":\"PUT_ASYNC\"}}"
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $DECODE_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    --kv-transfer-config "$DECODE_KV_CONFIG" \
    > "$LOG_DIR/decode.log" 2>&1 &

echo "[4/6] Starting Replica 0 (GPU 2, port $REPLICA0_PORT)..."
CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $REPLICA0_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replica0.log" 2>&1 &

echo "[5/6] Starting Replica 1 (GPU 3, port $REPLICA1_PORT)..."
CUDA_VISIBLE_DEVICES=3 vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $REPLICA1_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code --disable-log-requests \
    --enable-prefix-caching \
    > "$LOG_DIR/replica1.log" 2>&1 &

# Wait for vLLM servers
echo ""
echo "Waiting for vLLM servers..."
MAX_WAIT=300
WAITED=0

for PORT in $PREFILL_PORT $DECODE_PORT $REPLICA0_PORT $REPLICA1_PORT; do
    while ! curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
        sleep 2; WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT ]; then echo "Timeout waiting for port $PORT"; exit 1; fi
        [ $((WAITED % 30)) -eq 0 ] && echo "  Still waiting for port $PORT... ($WAITED s)"
    done
    echo "  Port $PORT: READY"
done

# Start Replica Proxy
echo "[6/6] Starting Replica Proxy..."
python "$SRC_DIR/simple_replica_proxy.py" \
    --workers "localhost:$REPLICA0_PORT,localhost:$REPLICA1_PORT" \
    --port $REPLICA_PROXY_PORT \
    > "$LOG_DIR/replica_proxy.log" 2>&1 &

sleep 3
echo "  PD/PPD Proxy: $(curl -s http://localhost:$PD_PROXY_PORT/mode 2>/dev/null | head -c 100)"
echo "  Replica Proxy: $(curl -s http://localhost:$REPLICA_PROXY_PORT/status 2>/dev/null | head -c 100)"

echo ""
echo "=============================================="
echo "All Optimizer Servers Ready!"
echo "=============================================="
echo ""
echo "PD/PPD Mode (GPU 0,1):"
echo "  Proxy:   http://localhost:$PD_PROXY_PORT"
echo "  Prefill: http://localhost:$PREFILL_PORT (GPU 0)"
echo "  Decode:  http://localhost:$DECODE_PORT (GPU 1)"
echo ""
echo "Replica Mode (GPU 2,3):"
echo "  Proxy:   http://localhost:$REPLICA_PROXY_PORT"
echo "  Worker0: http://localhost:$REPLICA0_PORT (GPU 2)"
echo "  Worker1: http://localhost:$REPLICA1_PORT (GPU 3)"
echo ""
echo "To stop: pkill -f 'vllm serve'; pkill -f proxy"
echo "=============================================="
