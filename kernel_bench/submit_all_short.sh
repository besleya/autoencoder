#!/bin/bash
# Master SLURM submission script for all kernel benchmarks (SHORT QUEUE VERSION)
# Uses short_gpu partition for faster turnaround - jobs will run sooner!
# Submits one job per kernel to the queue

set -euo pipefail

KERNELS=(
    kernel_add_bias
    kernel_relu_forward
    kernel_relu_backward
    kernel_fill_ones
    kernel_adam_update
    kernel_sparse_loss_and_grad
    kernel_accumulate_loss
    log_normalize_columns_kernel
)

BENCHMARK_DIR="/mnt/home/besleya/autoencoder/kernel_bench"

echo "==============================================="
echo "Submitting kernel benchmark jobs to SHORT QUEUE"
echo "==============================================="
echo ""
echo "NOTE: Using short_gpu partition for faster turnaround!"
echo ""

SUBMITTED_JOBS=()

for KERNEL in "${KERNELS[@]}"; do
    KERNEL_DIR="${BENCHMARK_DIR}/${KERNEL}"
    LOG_DIR="${KERNEL_DIR}/slurm_logs"
    
    # Create log directory
    mkdir -p "${LOG_DIR}"
    
    # Submit the job and capture job ID
    echo "Submitting ${KERNEL}..."
    JOB_OUTPUT=$(sbatch --partition=short-gpu "${KERNEL_DIR}/run_tests.sh" 2>&1)
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K[0-9]+')
    
    if [[ -z "$JOB_ID" ]]; then
        echo "ERROR: Failed to submit job for ${KERNEL}"
        exit 1
    fi
    
    echo "  -> Job ID: ${JOB_ID}"
    SUBMITTED_JOBS+=("${KERNEL}:${JOB_ID}")
    echo ""
done

echo "==============================================="
echo "All jobs submitted successfully to SHORT QUEUE!"
echo "==============================================="
echo ""
echo "Submitted jobs:"
for JOB_INFO in "${SUBMITTED_JOBS[@]}"; do
    KERNEL="${JOB_INFO%:*}"
    JOB_ID="${JOB_INFO#*:}"
    echo "  ${KERNEL}: ${JOB_ID}"
done
echo ""
echo "==============================================="
echo "Why faster?"
echo "==============================================="
echo "- Using short-gpu partition (lower time-limit = higher priority)"
echo "- Lower fairshare competition on short queue"
echo "- Time limit: 10 min (fits within 4-hour short-gpu limits)"
echo "- Jobs on shorter queues typically start sooner!"
echo ""
echo "==============================================="
echo "Next steps:"
echo "==============================================="
echo "1. Check job status:"
echo "   squeue -u \$USER --partition=short-gpu"
echo ""
echo "2. View job output (replace JOBID with actual ID):"
for JOB_INFO in "${SUBMITTED_JOBS[@]}"; do
    KERNEL="${JOB_INFO%:*}"
    JOB_ID="${JOB_INFO#*:}"
    echo "   tail -f ${BENCHMARK_DIR}/${KERNEL}/slurm_logs/kbench_${KERNEL}_${JOB_ID}.out"
done
echo ""
echo "3. View all log files once complete:"
echo "   ls -lh ${BENCHMARK_DIR}/*/slurm_logs/"
echo ""
