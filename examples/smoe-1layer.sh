#!/usr/bin/env bash

# Define model args
MODEL_ARGS=(
  --model_config configs/models/tiny-moe-64-emb-1-decoder-8192.yaml
)

# Define data/training args
DATA_ARGS=(
  --training_config configs/training/basic-medium-data.yaml
)

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run training with both sets of args
python pretrain_language.py \
  "${MODEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" | tee "logs/out_${TIMESTAMP}.log"
