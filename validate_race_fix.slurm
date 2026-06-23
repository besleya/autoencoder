#!/bin/bash
#SBATCH --job-name=validate_race_fix
#SBATCH --output=/mnt/home/besleya/autoencoder/slurm_logs/output_%j_%x.txt
#SBATCH --partition=short-gpu
#SBATCH --gpus-per-node=nvidia_h100_nvl:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail

echo "Starting validate_race_fix job $SLURM_JOB_ID"

mkdir -p /mnt/home/besleya/autoencoder/slurm_logs/
cd /mnt/home/besleya/autoencoder

echo "Cleaning old build..."
make clean

echo "Building main_gpu..."
make main_gpu

echo "Generating paths file with 277 .1pz files..."
PATHS_FILE=/tmp/main_gpu_paths_$SLURM_JOB_ID.txt
find ~/quant -mindepth 3 -maxdepth 3 -name '*.1pz' | shuf -n 277 > "$PATHS_FILE"

echo "Running main_gpu with race-fix configuration (4 epochs, batch=256, lr=0.001, seed=42)..."
RUN_LOG=/mnt/home/besleya/autoencoder/slurm_logs/run_${SLURM_JOB_ID}_main_gpu.txt
xargs -a "$PATHS_FILE" ./main_gpu --epochs 4 --batch 256 --lr 0.001 --seed 42 2>&1 | tee "$RUN_LOG"

echo ""
echo "=== Loss trajectory (mean_loss lines) ==="
grep -E 'mean_loss' "$RUN_LOG" || echo "(no mean_loss lines found)"

echo ""
echo "=== First vs last mean_loss ==="
FIRST=$(grep -oE 'mean_loss=[0-9.eE+-]+' "$RUN_LOG" | head -1)
LAST=$(grep -oE 'mean_loss=[0-9.eE+-]+' "$RUN_LOG" | tail -1)
echo "first: $FIRST"
echo "last:  $LAST"
echo "(fix is verified if 'last' < 'first')"

rm -f "$PATHS_FILE"
echo ""
echo "=== Done ==="
