#!/bin/bash
# ============================================================================
# One-Click Submission Script for All Comprehensive Benchmark Tests
# Updated: 8 batches (2 configs each) + 2 supplemental tests
# Note: 1R_1P_1D_1pD already completed, skipped
# ============================================================================

set -e

cd /net/projects2/ds3lab/zongzel/vllm/ppd

echo "=============================================="
echo "Comprehensive Benchmark - Submit All Tests"
echo "=============================================="
echo ""

# ============================================================================
# Stage 1: Main Benchmark (8 batches, sequential)
# ============================================================================

echo "Stage 1: Main Benchmark (16 configs, 18 workloads, 10 QPS)"
echo "  Total test points: 2,880 (16 configs × 180 points)"
echo "  Estimated time: ~24 hours (8 batches × 3h)"
echo "  Note: 1R_1P_1D_1pD already completed (180 points)"
echo ""

# Batch 1: 4R, 1P_3D
JOB1=$(sbatch --parsable scripts/benchmark/sbatch_batch_1.sh)
echo "✓ Batch 1 submitted: Job ${JOB1}"
echo "  Configs: 4R, 1P_3D"

# Batch 2: 1P_2D_1pD, 1P_1D_2pD (depends on Batch 1)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} scripts/benchmark/sbatch_batch_2.sh)
echo "✓ Batch 2 submitted: Job ${JOB2} (after ${JOB1})"
echo "  Configs: 1P_2D_1pD, 1P_1D_2pD"

# Batch 3: 1P_3pD, 2P_2D (depends on Batch 2)
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} scripts/benchmark/sbatch_batch_3.sh)
echo "✓ Batch 3 submitted: Job ${JOB3} (after ${JOB2})"
echo "  Configs: 1P_3pD, 2P_2D"

# Batch 4: 2P_1D_1pD, 2P_2pD (depends on Batch 3)
JOB4=$(sbatch --parsable --dependency=afterok:${JOB3} scripts/benchmark/sbatch_batch_4.sh)
echo "✓ Batch 4 submitted: Job ${JOB4} (after ${JOB3})"
echo "  Configs: 2P_1D_1pD, 2P_2pD"

# Batch 5: 3P_1D, 3P_1pD (depends on Batch 4)
JOB5=$(sbatch --parsable --dependency=afterok:${JOB4} scripts/benchmark/sbatch_batch_5.sh)
echo "✓ Batch 5 submitted: Job ${JOB5} (after ${JOB4})"
echo "  Configs: 3P_1D, 3P_1pD"

# Batch 6: 1R_1P_2D, 1R_1P_2pD (depends on Batch 5)
JOB6=$(sbatch --parsable --dependency=afterok:${JOB5} scripts/benchmark/sbatch_batch_6.sh)
echo "✓ Batch 6 submitted: Job ${JOB6} (after ${JOB5})"
echo "  Configs: 1R_1P_2D, 1R_1P_2pD"

# Batch 7: 1R_2P_1D, 1R_2P_1pD (depends on Batch 6)
JOB7=$(sbatch --parsable --dependency=afterok:${JOB6} scripts/benchmark/sbatch_batch_7.sh)
echo "✓ Batch 7 submitted: Job ${JOB7} (after ${JOB6})"
echo "  Configs: 1R_2P_1D, 1R_2P_1pD"

# Batch 8: 2R_1P_1D, 2R_1P_1pD (depends on Batch 7)
JOB8=$(sbatch --parsable --dependency=afterok:${JOB7} scripts/benchmark/sbatch_batch_8.sh)
echo "✓ Batch 8 submitted: Job ${JOB8} (after ${JOB7})"
echo "  Configs: 2R_1P_1D, 2R_1P_1pD"

echo ""
echo "Stage 1 complete: 8 batches queued"
echo ""

# ============================================================================
# Stage 2: High QPS Boundary Test (depends on Stage 1)
# ============================================================================

echo "Stage 2: High QPS Boundary Test"
echo "  Configs: 3P_1D, 3P_1pD, 2P_2D, 2P_2pD"
echo "  Workloads: large_very_long_gen, large_huge_paste"
echo "  QPS: 30, 40, 50, 60, 80, 100"
echo "  Total test points: 48"
echo "  Estimated time: 2-3 hours"
echo ""

JOB_HIGH_QPS=$(sbatch --parsable --dependency=afterok:${JOB8} scripts/benchmark/sbatch_high_qps.sh)
echo "✓ High QPS test submitted: Job ${JOB_HIGH_QPS} (after ${JOB8})"
echo ""

# ============================================================================
# Stage 3: Extreme Context Test (depends on Stage 2)
# ============================================================================

echo "Stage 3: Extreme Context Test"
echo "  Config: 2P_2pD"
echo "  QPS: 20"
echo "  Context sizes: 4096, 6144, 8192, 12288, 16384 tokens"
echo "  Total test points: 5"
echo "  Estimated time: 30-90 minutes"
echo ""

JOB_EXTREME=$(sbatch --parsable --dependency=afterok:${JOB_HIGH_QPS} scripts/benchmark/sbatch_extreme_context.sh)
echo "✓ Extreme context test submitted: Job ${JOB_EXTREME} (after ${JOB_HIGH_QPS})"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=============================================="
echo "All tests submitted successfully!"
echo "=============================================="
echo ""
echo "Job Chain:"
echo "  ${JOB1} → ${JOB2} → ${JOB3} → ${JOB4} → ${JOB5} → ${JOB6} → ${JOB7} → ${JOB8} → ${JOB_HIGH_QPS} → ${JOB_EXTREME}"
echo ""
echo "Total test points: 2,880 + 48 + 5 = 2,933"
echo "  (Plus 180 already completed from 1R_1P_1D_1pD)"
echo "  Grand total: 3,113 test points"
echo ""
echo "Estimated total time: ~27 hours"
echo "  Stage 1: ~24 hours (8 batches × 3h)"
echo "  Stage 2: ~2-3 hours"
echo "  Stage 3: ~0.5-1.5 hours"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/benchmark/batch1_${JOB1}.out"
echo ""
echo "Results will be saved to:"
echo "  results/comprehensive/<CONFIG>/<WORKLOAD>/"
echo "  results/oom_boundary/"
echo "  results/extreme_context/"
echo "=============================================="
