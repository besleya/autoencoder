#!/bin/bash
#SBATCH --job-name=bench_read_1pz
#SBATCH --partition=short-gpu
#SBATCH --gpus-per-node=nvidia_h100_nvl:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/besleya/autoencoder/slurm_logs/%x_%j.out

set -euo pipefail

echo "=== Creating logs directory ==="
mkdir -p /mnt/home/besleya/autoencoder/slurm_logs/

echo "=== Changing to work directory ==="
cd /mnt/home/besleya/autoencoder

echo "=== Building bench_read_1pz ==="
g++ -O3 -fopenmp -std=c++17 -I/mnt/home/besleya/singlet/include bench_read_1pz.cpp -o bench_read_1pz -lzstd -lstdc++fs

echo "=== Picking sample files ==="
find ~/quant -mindepth 3 -maxdepth 3 -name '*.1pz' | shuf -n 100 > /tmp/bench_files_$SLURM_JOB_ID.txt
num_files=$(wc -l < /tmp/bench_files_$SLURM_JOB_ID.txt)
echo "Found and selected $num_files files"

echo "=== Running benchmark with n_threads=32 (full per-file output) ==="
./bench_read_1pz /tmp/bench_files_$SLURM_JOB_ID.txt 32

echo "=== Quick runs at different thread counts ==="
for threads in 1 4 8 16; do
    echo "=== Quick run with n_threads=$threads ==="
    ./bench_read_1pz /tmp/bench_files_$SLURM_JOB_ID.txt $threads
done

echo "=== Cleaning up ==="
rm -f /tmp/bench_files_$SLURM_JOB_ID.txt

echo "=== Done ==="
