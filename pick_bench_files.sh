#!/usr/bin/env bash
# Usage: ./pick_bench_files.sh [N]   (default 200)
# Finds N random .1pz files from ~/quant and prints them to stdout.

N="${1:-200}"
find ~/quant -mindepth 3 -maxdepth 3 -name '*.1pz' | shuf -n "$N"
