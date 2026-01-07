#!/bin/bash
# =============================================================================
# Submit All Benchmark Jobs (V3 - 16 Workloads x 2 Modes = 32 Jobs)
# =============================================================================
#
# Usage: ./scripts/submit_all_benchmarks.sh [RUN_NUM]
#        RUN_NUM: 1, 2, 3, ... (default: 2, since run1 already exists)
#
# 3D Orthogonal Design:
#   T1 (Context): S(512), M(1024), L(2048), XL(4096)
#   T2 (Request): a(32→64), b(32→512), c(256→256), d(1024→64)
#
# =============================================================================

RUN_NUM=${1:-2}  # Default to run2 since run1 already exists

cd /net/projects2/ds3lab/zongzel/vllm/ppd
mkdir -p logs "results/run${RUN_NUM}"

echo "========================================"
echo "Submitting Benchmark Jobs V3 (16 Workloads)"
echo "Run Number: $RUN_NUM"
echo "Output Dir: results/run${RUN_NUM}/"
echo "========================================"

# Submit QPS benchmark (PD/PPD) - 16 jobs
echo "Submitting QPS benchmarks (16 jobs)..."
QPS_JOB=$(sbatch --array=0-15 --parsable scripts/sbatch_benchmark.sh qps $RUN_NUM)
echo "  QPS Job ID: $QPS_JOB"

# Submit Replication benchmark - 16 jobs
echo "Submitting Replication benchmarks (16 jobs)..."
REP_JOB=$(sbatch --array=0-15 --parsable scripts/sbatch_benchmark.sh replication $RUN_NUM)
echo "  Replication Job ID: $REP_JOB"

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Run: $RUN_NUM"
echo "Total jobs: 32 (16 workloads x 2 modes)"
echo "  - QPS (PD/PPD): Job $QPS_JOB [0-15]"
echo "  - Replication:  Job $REP_JOB [0-15]"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f logs/bench_*.out"
echo ""
echo "Results: results/run${RUN_NUM}/qps_*.json"
echo "         results/run${RUN_NUM}/replication_*.json"
echo ""
echo "After all jobs complete:"
echo "  1. Merge this run:    python scripts/merge_benchmark_results.py $RUN_NUM"
echo "  2. Analyze this run:  python scripts/analyze_results.py $RUN_NUM"
echo "  3. After all 3 runs:  python scripts/average_runs.py"
echo "========================================"
