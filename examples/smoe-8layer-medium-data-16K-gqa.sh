#!/usr/bin/env bash

# Define model args
MODEL_ARGS=(
  --model_config configs/models/tiny-moe-64-emb-8-decoder-16384-4-1-gqa.yaml
)

# Define data
DATA_ARGS=(
  --training_config configs/training/basic.yaml
)

# Define training args
TRAINING_ARGS=(
  --profile 10
)

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run training with both sets of args
python pretrain_language.py \
  "${MODEL_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${DATA_ARGS[@]}" | tee "logs/out_${TIMESTAMP}.log"