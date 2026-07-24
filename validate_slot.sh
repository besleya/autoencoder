#!/bin/bash
#SBATCH --job-name=validate_slot
#SBATCH --output=/mnt/home/besleya/autoencoder/slurm_logs/validate_slot_%j.out
#SBATCH --partition=short-gpu
#SBATCH --gpus-per-node=nvidia_h100_nvl:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail

echo "Starting validate_slot job $SLURM_JOB_ID"
mkdir -p /mnt/home/besleya/autoencoder/slurm_logs/

cd /mnt/home/besleya/autoencoder
bash tests/slot/run_tests.sh

echo ""
echo "=== Slot validation completed successfully ==="
