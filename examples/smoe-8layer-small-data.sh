#!/usr/bin/env bash

# Define model args
MODEL_ARGS=(
  --model_config configs/models/tiny-moe-64-emb-8-decoder-8192.yaml
)

# Define data/training args
DATA_ARGS=(
  --training_config configs/training/basic.yaml
)

# Run training with both sets of args
python pretrain_language.py \
  "${MODEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" | tee out.log
