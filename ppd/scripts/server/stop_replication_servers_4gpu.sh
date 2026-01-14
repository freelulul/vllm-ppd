#!/bin/bash
# ============================================================================
# Stop Script for 4-GPU Replication Servers
# ============================================================================

echo "Stopping all 4-GPU replication servers..."

# Kill vLLM workers on replication ports
pkill -f "vllm serve.*:8300" 2>/dev/null || true
pkill -f "vllm serve.*:8400" 2>/dev/null || true
pkill -f "vllm serve.*:8500" 2>/dev/null || true
pkill -f "vllm serve.*:8600" 2>/dev/null || true

# Kill proxy
pkill -f "replication_proxy" 2>/dev/null || true

sleep 2

echo "All replication servers stopped."
