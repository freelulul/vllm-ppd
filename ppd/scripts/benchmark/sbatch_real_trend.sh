#!/bin/bash
#SBATCH --job-name=real_trend
#SBATCH --output=logs/real_trend_%j.log
#SBATCH --error=logs/real_trend_%j.err
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00

# Real Benchmark for Trend Data
# This script runs REAL requests against GPU servers to generate fig6 data
#
# Usage:
#   sbatch scripts/benchmark/sbatch_real_trend.sh           # Run all panels
#   sbatch scripts/benchmark/sbatch_real_trend.sh qps       # Run QPS panel only
#   sbatch scripts/benchmark/sbatch_real_trend.sh e2e_ratio # Run E2E ratio panel only
#
# Estimated time:
#   - QPS panel: ~30 minutes
#   - E2E ratio panel: ~15 minutes
#   - Strictness panel: ~10 minutes
#   - Input length panel: ~20 minutes
#   - All panels: ~75 minutes

set -e

echo "=========================================="
echo "REAL BENCHMARK FOR TREND DATA"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x A100"
echo "Start: $(date)"
echo ""

# Panel selection
PANEL=${1:-all}
echo "Panel: $PANEL"

# Setup
cd /net/projects2/ds3lab/zongzel/vllm/ppd
source ~/.bashrc
conda activate vllm-ppd
echo "Using conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# Create log directory
mkdir -p logs results

# Start servers (1P + 1D + 2 Replica for optimizer mode)
echo ""
echo "=========================================="
echo "Starting Optimizer Servers (1P+1D+2R)"
echo "=========================================="

./scripts/server/start_optimizer_servers.sh

# Wait for servers to be ready
echo ""
echo "Waiting for servers to stabilize..."
sleep 30

# Run real benchmark
echo ""
echo "=========================================="
echo "Running Real Benchmark"
echo "=========================================="

OUTPUT_FILE="results/real_trend_data_${SLURM_JOB_ID}.json"

python scripts/tests/benchmark_real_trend_data.py \
    --panel "$PANEL" \
    --output "$OUTPUT_FILE"

# Copy to standard location
cp "$OUTPUT_FILE" results/real_trend_data.json
cp "${OUTPUT_FILE%.json}_raw.json" results/real_trend_data_raw.json

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "Results: $OUTPUT_FILE"
echo "Raw data: ${OUTPUT_FILE%.json}_raw.json"
echo "End: $(date)"

# Stop servers
echo ""
echo "Stopping servers..."
pkill -f "vllm serve" || true
pkill -f "proxy" || true

echo "Done!"
