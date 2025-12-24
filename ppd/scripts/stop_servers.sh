#!/bin/bash
# ============================================================================
# Stop all vLLM PD/PPD Servers
# ============================================================================

echo "Stopping all PD/PPD servers..."

pkill -f "vllm serve" 2>/dev/null || true
pkill -f "disagg_proxy" 2>/dev/null || true

sleep 2

# Verify
if pgrep -f "vllm serve" > /dev/null 2>&1; then
    echo "Warning: Some vllm processes still running, force killing..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
fi

if pgrep -f "disagg_proxy" > /dev/null 2>&1; then
    echo "Warning: Proxy still running, force killing..."
    pkill -9 -f "disagg_proxy" 2>/dev/null || true
fi

echo "All servers stopped."
