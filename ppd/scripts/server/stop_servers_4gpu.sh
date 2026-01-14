#!/bin/bash
# ============================================================================
# Stop Script for 4-GPU vLLM Servers
# ============================================================================

echo "Stopping all 4-GPU vLLM servers..."

# Kill vLLM servers
pkill -f "vllm serve.*/net/projects2/ds3lab/zongzel/models--meta-llama--Llama-3.1-8B" 2>/dev/null || true

# Kill proxy
pkill -f "disagg_proxy" 2>/dev/null || true

# Wait a moment
sleep 2

# Check if any processes are still running
REMAINING=$(pgrep -f "vllm serve" 2>/dev/null | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "Warning: $REMAINING vllm processes still running. Force killing..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
fi

echo "All servers stopped."
