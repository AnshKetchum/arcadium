#!/usr/bin/env bash

# Define model args
MODEL_ARGS=(
  --model_config configs/models/gpt-2/tiny-gpt.yaml
)

# Define data/training args
DATA_ARGS=(
  --training_config configs/training/gpt-2-fineweb.yaml
)

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run training with both sets of args
python pretrain_language.py \
  "${MODEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" | tee "logs/out_${TIMESTAMP}.log"