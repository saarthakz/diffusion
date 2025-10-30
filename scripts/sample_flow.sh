#!/bin/bash

# Sample from trained flow matching model
# Usage: bash scripts/sample_flow.sh <model_directory>

# Default model directory (can be overridden by command line argument)
MODEL_DIR="${1:-models/flow_matching_mnist 31-10-2025 10:59:15 UTC}"

python validation/sample_flow.py \
    --model_dir "$MODEL_DIR" \
    --num_samples 64 \
    --sampling_steps 1000 \
    --seed 42 \
    --use_ema

