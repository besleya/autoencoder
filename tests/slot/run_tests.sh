#!/bin/bash

set -euo pipefail

# Navigate to test directory
cd "$(dirname "$0")"

# Step 1: Build test binary
echo "Building test_slot binary..."
make clean
make

# Step 2: Run tests
echo "Running slot unit tests..."
./test_slot
