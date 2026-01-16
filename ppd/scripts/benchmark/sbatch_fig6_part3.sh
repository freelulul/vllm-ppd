#!/bin/bash
#SBATCH --job-name=fig6_FGH
#SBATCH --output=logs/fig6_FGH_%j.out
#SBATCH --error=logs/fig6_FGH_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=ds3lab-own
#SBATCH --nodelist=n001
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=16

# =============================================================================
# Fig6 Benchmark Part 3: Panel F + G + H
# Estimated time: ~3.5 hours
# =============================================================================

set -e

echo "=============================================="
echo "Fig6 Benchmark - Part 3 (Panel F + G + H)"
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

# Create directories
mkdir -p logs results/fig6

# Run Panel F
echo ""
echo "========== Running Panel F =========="
python scripts/tests/fig6_benchmark.py \
    --panels F \
    --output-dir results/fig6 \
    --duration 30

# Run Panel G
echo ""
echo "========== Running Panel G =========="
python scripts/tests/fig6_benchmark.py \
    --panels G \
    --output-dir results/fig6 \
    --duration 30

# Run Panel H
echo ""
echo "========== Running Panel H =========="
python scripts/tests/fig6_benchmark.py \
    --panels H \
    --output-dir results/fig6 \
    --duration 30

echo ""
echo "=============================================="
echo "Part 3 (Panel F + G + H) Complete!"
echo "=============================================="
echo "Done at $(date)"
