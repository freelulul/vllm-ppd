#!/bin/bash
# Stop all replication mode servers

echo "Stopping replication mode servers..."

# Stop proxy
pkill -f "replication_proxy" 2>/dev/null || true

# Stop workers (by port)
pkill -f "vllm serve.*:8300" 2>/dev/null || true
pkill -f "vllm serve.*:8400" 2>/dev/null || true

sleep 2
echo "Done."
