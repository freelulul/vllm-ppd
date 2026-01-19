#!/bin/bash
# ============================================================================
# Comprehensive Cleanup Script
# Ensures all vLLM and proxy processes are killed and GPU memory is released
# ============================================================================

SCRIPT_DIR="$(dirname "$0")"

# Source config for MODEL_NAME (optional, fallback to hardcoded if not available)
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    source "$SCRIPT_DIR/config.sh"
else
    MODEL_NAME="Llama-3.1-8B"
fi

echo "=== Starting comprehensive cleanup ==="

# Step 1: Kill all proxy processes
echo "[1/5] Killing proxy processes..."
pkill -9 -f "comprehensive_proxy" 2>/dev/null
pkill -9 -f "disagg_proxy" 2>/dev/null
pkill -9 -f "simple_replica_proxy" 2>/dev/null
pkill -9 -f "optimizer_proxy" 2>/dev/null

# Step 2: Kill all vLLM serve processes
echo "[2/5] Killing vLLM serve processes..."
pkill -9 -f "vllm serve" 2>/dev/null

# Step 3: Kill EngineCore processes (these hold GPU memory)
echo "[3/5] Killing EngineCore processes..."
pkill -9 -f "EngineCore" 2>/dev/null

# Step 4: Kill any remaining Python processes using our model
echo "[4/5] Killing model-related processes..."
pkill -9 -f "$MODEL_NAME" 2>/dev/null

# Step 5: Wait for processes to die and GPU memory to release
echo "[5/5] Waiting for GPU memory to release..."
MAX_WAIT=60
WAITED=0

while true; do
    # Check GPU memory usage
    TOTAL_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum += $1} END {print sum}')

    if [ "$TOTAL_USED" -lt 100 ]; then
        echo "  GPU memory released (total used: ${TOTAL_USED} MiB)"
        break
    fi

    sleep 2
    WAITED=$((WAITED + 2))

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "  WARNING: Timeout waiting for GPU memory release"
        echo "  Current GPU status:"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
        break
    fi

    if [ $((WAITED % 10)) -eq 0 ]; then
        echo "  Still waiting... (${WAITED}s, total used: ${TOTAL_USED} MiB)"
    fi
done

# Step 6: Verify port availability
echo ""
echo "Checking port availability..."
for port in 8100 8101 8200 8201 8202 8300 8400 8500 8600 10001 10002 30001; do
    if lsof -i :$port >/dev/null 2>&1; then
        echo "  WARNING: Port $port still in use"
        lsof -i :$port | head -2
    fi
done

# Final status
echo ""
echo "=== Cleanup complete ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
