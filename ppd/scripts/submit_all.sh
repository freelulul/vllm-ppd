#!/bin/bash
# =============================================================================
# Submit All Benchmark Jobs (PPD, PD, Replica)
# =============================================================================
#
# Usage:
#   ./scripts/submit_all.sh           # Submit runs 1,2,3 for all modes
#   ./scripts/submit_all.sh 1         # Submit only run 1
#   ./scripts/submit_all.sh 1 2       # Submit runs 1 and 2
#
# Jobs are chained with dependencies to avoid GPU conflicts on n001
# Order: PPD -> PD -> Replica (for each run)
#
# Benchmark Configuration:
#   - Duration: 10 seconds per QPS level
#   - Workloads: 20 (XS/S/M/L x a/b + variants)
#   - Total records per mode per run: ~108
#   - Estimated time: ~2-3 hours per mode
#
# =============================================================================

set -e

cd /net/projects2/ds3lab/zongzel/vllm/ppd

# Get run IDs from arguments (default: 1 2 3)
RUNS="${@:-1 2 3}"

echo "=============================================="
echo "PD/PPD/Replica Benchmark Suite"
echo "=============================================="
echo "Partition:  ds3lab-own"
echo "Node:       n001"
echo "GPUs:       4 x GPU"
echo "Duration:   10s per QPS level"
echo "Runs:       ${RUNS}"
echo ""
echo "Estimated time per run: ~6-9 hours (3 modes)"
echo "=============================================="
echo ""

mkdir -p logs results/ppd results/pd results/replica

LAST_JOB=""
TOTAL_JOBS=0

for RUN in ${RUNS}; do
    echo "--- Submitting Run ${RUN} ---"

    # Submit PPD (depends on previous run if exists)
    if [ -z "$LAST_JOB" ]; then
        PPD_JOB=$(sbatch --parsable scripts/sbatch_ppd.sh ${RUN})
    else
        PPD_JOB=$(sbatch --parsable --dependency=afterany:${LAST_JOB} scripts/sbatch_ppd.sh ${RUN})
    fi
    echo "  [1/3] PPD:     Job ${PPD_JOB}"
    TOTAL_JOBS=$((TOTAL_JOBS + 1))

    # Submit PD (depends on PPD)
    PD_JOB=$(sbatch --parsable --dependency=afterany:${PPD_JOB} scripts/sbatch_pd.sh ${RUN})
    echo "  [2/3] PD:      Job ${PD_JOB} (after PPD)"
    TOTAL_JOBS=$((TOTAL_JOBS + 1))

    # Submit Replica (depends on PD)
    REP_JOB=$(sbatch --parsable --dependency=afterany:${PD_JOB} scripts/sbatch_replica.sh ${RUN})
    echo "  [3/3] Replica: Job ${REP_JOB} (after PD)"
    TOTAL_JOBS=$((TOTAL_JOBS + 1))

    LAST_JOB=${REP_JOB}
    echo ""
done

echo "=============================================="
echo "Submitted ${TOTAL_JOBS} jobs successfully!"
echo "=============================================="
echo ""
echo "Commands:"
echo "  squeue -u \$USER                    # Check job status"
echo "  tail -f logs/bench_ppd_*.out       # View PPD logs"
echo "  tail -f logs/bench_pd_*.out        # View PD logs"
echo "  tail -f logs/bench_replica_*.out   # View Replica logs"
echo ""
echo "After all jobs complete:"
echo "  python scripts/merge_results.py    # Merge all results"
echo "  python scripts/plot_results.py     # Generate plots"
echo ""
echo "Output files:"
echo "  results/ppd/ppd_run*_*.json"
echo "  results/pd/pd_run*_*.json"
echo "  results/replica/replica_run*_*.json"
echo "=============================================="
