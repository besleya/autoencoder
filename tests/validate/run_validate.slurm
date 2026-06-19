#!/bin/bash
#SBATCH --job-name=validate
#SBATCH --output=/mnt/home/besleya/autoencoder/tests/validate/output_%j.txt
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --partition=short-gpu
#SBATCH --gpus-per-node=nvidia_h100_nvl:1

set -e

echo "Starting validation job $SLURM_JOB_ID"

# Build C++ validator
cd /mnt/home/besleya/autoencoder
echo "Clearing old files..."
make clean
echo "Building C++ validator..."
make validate

# Run C++ validator
echo "Running C++ validator..."
./tests/validate/validate
CPP_EXIT_CODE=$?
echo "C++ validator exited with code: $CPP_EXIT_CODE"

# Run Python comparison
echo "Running Python comparison..."
/mnt/home/besleya/s3/bin/python3 /mnt/home/besleya/autoencoder/tests/validate/compare.py --recompute-cache

echo "Job $SLURM_JOB_ID finished. See /mnt/home/besleya/autoencoder/tests/validate/output_$SLURM_JOB_ID.txt"
