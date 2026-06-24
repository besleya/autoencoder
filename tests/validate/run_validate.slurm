#!/bin/bash
#SBATCH --job-name=validate
#SBATCH --output=/mnt/home/besleya/autoencoder/tests/validate/output_%j_%x.txt
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --partition=debug-gpu
#SBATCH --gpus-per-node=tesla_v100s:1

set -e

echo "Starting validation job $SLURM_JOB_ID"
date

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
/mnt/home/besleya/s3/bin/python3 /mnt/home/besleya/autoencoder/tests/validate/compare.py --use-cache #--recompute-cache

echo "Job $SLURM_JOB_ID finished. See ~/autoencoder/tests/validate/output_${SLURM_JOB_ID}_${SLURM_JOB_NAME}.txt"
