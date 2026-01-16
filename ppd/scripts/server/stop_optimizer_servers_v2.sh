#!/bin/bash
# ============================================================================
# Stop Optimizer V2 Servers (1P + 1D_pure + 1pD + 1R)
# ============================================================================

echo "=============================================="
echo "Stopping Optimizer V2 Servers"
echo "=============================================="

# Ports used by optimizer v2
PREFILL_PORT=8100
DECODE_PURE_PORT=8200
DECODE_PPD_PORT=8201
REPLICA_PORT=8300
OPTIMIZER_PROXY_PORT=10001

# Kill by port
for PORT in $PREFILL_PORT $DECODE_PURE_PORT $DECODE_PPD_PORT $REPLICA_PORT; do
    pid=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "Stopping server on port $PORT (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
    fi
done

# Kill optimizer proxy
pkill -f "optimizer_proxy_v2" 2>/dev/null && echo "Stopped optimizer_proxy_v2" || true
pkill -f "optimizer_proxy" 2>/dev/null || true

# Kill EngineCore workers
pkill -9 -f "EngineCore" 2>/dev/null || true

sleep 2

echo ""
echo "Optimizer V2 servers stopped."
echo "=============================================="
