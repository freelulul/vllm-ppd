#!/bin/bash
# ============================================================================
# Stop 4-GPU PD/PPD Servers (2P + 2D)
# ============================================================================

echo "=============================================="
echo "Stopping 4-GPU PD/PPD Servers"
echo "=============================================="

# Ports used by 4gpu mode
PREFILL0_PORT=8100
PREFILL1_PORT=8101
DECODE0_PORT=8200
DECODE1_PORT=8201
PROXY_PORT=10001

# Kill by port
for PORT in $PREFILL0_PORT $PREFILL1_PORT $DECODE0_PORT $DECODE1_PORT; do
    pid=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "Stopping server on port $PORT (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
    fi
done

# Kill vLLM servers
pkill -f "vllm serve.*/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B" 2>/dev/null || true

# Kill disagg proxy
pkill -f "disagg_proxy_ppd_4gpu" 2>/dev/null && echo "Stopped disagg proxy" || true
pkill -f "disagg_proxy" 2>/dev/null || true

# Kill EngineCore workers (zombie vLLM processes)
pkill -9 -f "EngineCore" 2>/dev/null || true

sleep 2

# Check if any processes are still running
REMAINING=$(pgrep -f "vllm serve" 2>/dev/null | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "Warning: $REMAINING vllm processes still running. Force killing..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
fi

echo ""
echo "4-GPU PD/PPD servers stopped."
echo "=============================================="
