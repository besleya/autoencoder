#!/bin/bash
#SBATCH --job-name=kbench_log_normalize_columns_kernel
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=nvidia_h100_nvl:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/home/besleya/autoencoder/kernel_bench/log_normalize_columns_kernel/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/home/besleya/autoencoder/kernel_bench/log_normalize_columns_kernel/slurm_logs/%x_%j.err

set -euo pipefail

# Set up environment
export PATH=/usr/local/cuda/bin:$PATH

# Navigate to kernel directory
cd /mnt/home/besleya/autoencoder/kernel_bench/log_normalize_columns_kernel

# Step 1: Generate reference test data
echo "==============================================="
echo "Step 1: Generating reference test data"
echo "==============================================="
~/s3/bin/python reference.py
echo ""

# Step 2: Build test_accuracy and bench
echo "==============================================="
echo "Step 2: Building test and benchmark binaries"
echo "==============================================="
make clean
make
echo ""

# Step 3: Run accuracy test
echo "==============================================="
echo "Step 3: Running accuracy test"
echo "==============================================="
if ! ./test_accuracy; then
    echo "ERROR: Accuracy test failed!"
    exit 1
fi
echo ""

# Step 4: Run benchmark
echo "==============================================="
echo "Step 4: Running benchmark"
echo "==============================================="
./bench
echo ""

echo "==============================================="
echo "All tests completed successfully"
echo "==============================================="
