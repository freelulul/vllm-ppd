#!/bin/bash
# =============================================================================
# Submit all ShareGPT & WildChat benchmark batches
# Usage: ./scripts/benchmark/submit_sharegpt_benchmarks.sh [batch_numbers...]
# Examples:
#   ./scripts/benchmark/submit_sharegpt_benchmarks.sh        # Submit all 8 batches
#   ./scripts/benchmark/submit_sharegpt_benchmarks.sh 1 2 3  # Submit batches 1, 2, 3
# =============================================================================

set -e

cd /net/projects2/ds3lab/zongzel/vllm/ppd

# Create log directories
mkdir -p logs/benchmark

# Determine which batches to submit
if [ $# -eq 0 ]; then
    BATCHES="1 2 3 4 5 6 7 8"
else
    BATCHES="$@"
fi

echo "=============================================="
echo "ShareGPT & WildChat Benchmark Submission"
echo "=============================================="
echo "Batches to submit: ${BATCHES}"
echo ""

# Track job IDs for dependencies (sequential submission)
PREV_JOB_ID=""

for BATCH in ${BATCHES}; do
    SCRIPT="scripts/benchmark/sbatch_sharegpt_batch_${BATCH}.sh"

    if [ ! -f "${SCRIPT}" ]; then
        echo "ERROR: Script not found: ${SCRIPT}"
        continue
    fi

    echo "Submitting batch ${BATCH}..."

    if [ -z "${PREV_JOB_ID}" ]; then
        # First batch: no dependency
        JOB_ID=$(sbatch --parsable ${SCRIPT})
    else
        # Subsequent batches: depend on previous (same GPU node)
        JOB_ID=$(sbatch --parsable --dependency=afterany:${PREV_JOB_ID} ${SCRIPT})
    fi

    echo "  Batch ${BATCH}: Job ID = ${JOB_ID}"
    PREV_JOB_ID=${JOB_ID}
done

echo ""
echo "=============================================="
echo "All batches submitted"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all: scancel -u \$USER"
echo ""
echo "After completion, merge results with:"
echo "  python scripts/benchmark/merge_sharegpt_results.py"
