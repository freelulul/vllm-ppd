#!/bin/bash
# ============================================================================
# Stop 4-GPU Replication Servers
# ============================================================================

echo "=============================================="
echo "Stopping 4-GPU Replication Servers"
echo "=============================================="

# Ports used by replication mode
WORKER0_PORT=8300
WORKER1_PORT=8400
WORKER2_PORT=8500
WORKER3_PORT=8600
PROXY_PORT=10002

# Kill by port
for PORT in $WORKER0_PORT $WORKER1_PORT $WORKER2_PORT $WORKER3_PORT; do
    pid=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "Stopping worker on port $PORT (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
    fi
done

# Kill vLLM workers on replication ports
pkill -f "vllm serve.*:8300" 2>/dev/null || true
pkill -f "vllm serve.*:8400" 2>/dev/null || true
pkill -f "vllm serve.*:8500" 2>/dev/null || true
pkill -f "vllm serve.*:8600" 2>/dev/null || true

# Kill replication proxy
pkill -f "simple_replica_proxy.*$PROXY_PORT" 2>/dev/null || true
pkill -f "replication_proxy" 2>/dev/null && echo "Stopped replication proxy" || true

# Kill EngineCore workers (zombie vLLM processes)
pkill -9 -f "EngineCore" 2>/dev/null || true

sleep 2

echo ""
echo "Replication servers stopped."
echo "=============================================="
