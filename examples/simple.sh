#!/usr/bin/env bash
# run from the project root:  bash examples/simple.sh

# Define model args
MODEL_ARGS=(
  --model_config configs/models/universal-transformer/tiny-ut.yaml
)

# Define data/training args
DATA_ARGS=(
  --training_config configs/training/basic.yaml
)

# Visualizations per checkpoint
VIS_ARGS=(
  --num-visualize-generations 20
  --loss-viz
  --loss-viz-grid-points 15
  --loss-viz-eval-batches 10
  --loss-viz-interactive 
  --val-batch-size 1
  --spectral-viz
)

# Optional: run lm-eval at every checkpoint (remove to skip)
EVAL_ARGS=(
#   --eval_config configs/eval/basic.yaml
)

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run training with both sets of args
python -m arcadium.tasks.language.pretrain \
  "${MODEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${VIS_ARGS[@]}" \
  "${EVAL_ARGS[@]}" | tee "logs/out_${TIMESTAMP}.log"
  