#!/bin/bash
# ============================================================================
# Stop All vLLM Servers and Proxies
# ============================================================================

echo "=============================================="
echo "Stopping All vLLM Servers"
echo "=============================================="

# Kill vLLM serve processes
echo "[1/4] Stopping vLLM servers..."
pkill -f "vllm serve" 2>/dev/null && echo "  vLLM serve processes stopped" || echo "  No vLLM serve processes found"

# Kill EngineCore processes (zombie workers)
echo "[2/4] Stopping EngineCore workers..."
pkill -9 -f "EngineCore" 2>/dev/null && echo "  EngineCore processes stopped" || echo "  No EngineCore processes found"
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true

# Kill proxy processes
echo "[3/4] Stopping proxy processes..."
pkill -f "disagg_proxy" 2>/dev/null && echo "  disagg_proxy stopped" || true
pkill -f "optimizer_proxy" 2>/dev/null && echo "  optimizer_proxy stopped" || true
pkill -f "simple_replica_proxy" 2>/dev/null && echo "  simple_replica_proxy stopped" || true
pkill -f "replication_proxy" 2>/dev/null && echo "  replication_proxy stopped" || true
# Note: "pkill: killing pid X failed: Operation not permitted" for pushprox_client is harmless (system service)

# Wait for processes to terminate
echo "[4/4] Waiting for processes to terminate..."
sleep 3

# Verify
echo ""
echo "Verification:"
remaining=$(ps aux | grep -E "(vllm serve|proxy)" | grep -v grep | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "  All servers stopped successfully."
else
    echo "  WARNING: $remaining processes still running:"
    ps aux | grep -E "(vllm serve|proxy)" | grep -v grep
    echo ""
    echo "  To force kill: kill -9 <PID>"
fi

# Check GPU memory
echo ""
echo "GPU Memory Status:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
    while IFS=', ' read -r idx used total; do
        echo "  GPU $idx: ${used}/${total} MiB"
    done

echo ""
echo "=============================================="
