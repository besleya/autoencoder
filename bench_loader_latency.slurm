#!/bin/bash
#SBATCH --job-name=bench_loader_latency
#SBATCH --partition=short-gpu
#SBATCH --gpus-per-node=nvidia_h100_nvl:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/besleya/autoencoder/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/home/besleya/autoencoder/slurm_logs/%x_%j.err

set -euo pipefail

echo "=== Creating logs directory ==="
mkdir -p /mnt/home/besleya/autoencoder/slurm_logs/

echo "=== Changing to work directory ==="
cd /mnt/home/besleya/autoencoder

echo "=== Cleaning previous build ==="
make clean_bench_loader

echo "=== Building bench_loader_latency ==="
make bench_loader_latency

echo "=== Picking sample files ==="
PATHS_FILE=/tmp/bench_loader_latency_paths_$SLURM_JOB_ID.txt
find ~/quant -mindepth 3 -maxdepth 3 -name '*.1pz' | shuf -n 200 > "$PATHS_FILE"
n=$(wc -l < "$PATHS_FILE")
echo "Selected $n files for benchmarking"

echo ""
echo "========================================"
echo "=== Starting benchmark sweep ==="
echo "========================================"
echo ""

SUMMARY_FILE=/mnt/home/besleya/autoencoder/slurm_logs/bench_loader_latency_${SLURM_JOB_ID}_summary.csv
echo "# chunk,ring,dec,min_step_ms,mean_step_ms,trainer_waits" > "$SUMMARY_FILE"

for chunk in 32 64 128; do
  for ring in 2 4 8; do
    for dec in 8 16 32; do
      echo ""
      echo "========================================"
      echo "=== chunk=$chunk ring=$ring dec=$dec ==="
      echo "========================================"
      OUT=$(./bench_loader_latency --paths "$PATHS_FILE" --chunk-size $chunk --batch-size 512 --n-batches 500 --decode-threads $dec --ring-depth $ring --seed 42 2>&1)
      echo "$OUT"
      SUMMARY=$(echo "$OUT" | grep '^# SUMMARY:' | head -1)
      # parse summary line: "# SUMMARY: chunk=64 ring=4 dec=16 min_step_ms=18.40 mean_step_ms=8.20 trainer_waits=0"
      # extract numbers and write CSV row
      min_step=$(echo "$SUMMARY" | grep -oP 'min_step_ms=\K[0-9.]+' || echo "NA")
      mean_step=$(echo "$SUMMARY" | grep -oP 'mean_step_ms=\K[0-9.]+' || echo "NA")
      tw=$(echo "$SUMMARY" | grep -oP 'trainer_waits=\K[0-9]+' || echo "NA")
      echo "$chunk,$ring,$dec,$min_step,$mean_step,$tw" >> "$SUMMARY_FILE"
    done
  done
done

echo ""
echo "=== Aggregate summary CSV ==="
cat "$SUMMARY_FILE"

echo "=== Cleanup ==="
rm -f "$PATHS_FILE"

echo "=== Done ==="
