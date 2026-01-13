#!/bin/bash
#SBATCH --job-name=bench_pd
#SBATCH --output=logs/bench_pd_%j_%a.out
#SBATCH --error=logs/bench_pd_%j_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# Usage: sbatch --array=1-3 scripts/sbatch_pd.sh
# Or: sbatch scripts/sbatch_pd.sh 1  (for single run)

set -e

# Get run ID from array task ID or argument
RUN_ID=${SLURM_ARRAY_TASK_ID:-${1:-1}}

echo "=============================================="
echo "PD Benchmark - Run ${RUN_ID}"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate vllm-ppd
echo "Using conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

cd /net/projects2/ds3lab/zongzel/vllm/ppd

# Create log directory
mkdir -p logs results/pd

# Start servers
echo "Starting PD/PPD servers..."
./scripts/start_servers_4gpu.sh ppd
sleep 60  # Initial wait, then retry loop checks readiness

# Check proxy
for i in {1..20}; do
    if curl -s http://localhost:10001/status > /dev/null 2>&1; then
        echo "Proxy ready!"
        break
    fi
    echo "Waiting for proxy... ($i/20)"
    sleep 10
done

# Verify proxy is ready
STATUS=$(curl -s http://localhost:10001/status)
echo "Proxy status: ${STATUS}"

# Switch to PD mode
curl -s -X POST http://localhost:10001/mode/pd
sleep 2
echo "Mode set to PD"

# Run benchmark
echo ""
echo "Starting PD benchmark (run ${RUN_ID})..."
echo ""

python src/benchmark_pd.py \
    --run-id ${RUN_ID} \
    --duration 10 \
    --output-dir results/pd

echo ""
echo "=============================================="
echo "Benchmark Complete"
echo "=============================================="

# Stop servers
echo "Stopping servers..."
./scripts/stop_servers_4gpu.sh || true

echo "Done at $(date)"
