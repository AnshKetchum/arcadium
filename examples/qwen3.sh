#!/usr/bin/env bash
# run from the project root:  bash examples/simple.sh
MASTER_ADDR=localhost
MASTER_PORT=29500
NNODES=1
NODE_RANK=0

NUM_DP_RANKS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BASE_RUN_DIR="checkpoints"
export TOKENIZERS_PARALLELISM="false"

# Define model args
MODEL_ARGS=(
  --model_config configs/models/qwen3/fineweb-qwen.yaml
)

# Define data/training args
DATA_ARGS=(
  --training_config configs/training/qwen3-fineweb.yaml
)

# Visualizations per checkpoint
VIS_ARGS=(
  --num-visualize-generations 20
  --loss-viz
  --loss-viz-grid-points 15
  --loss-viz-eval-batches 10
  --loss-viz-interactive 
  --val-batch-size 4
  --spectral-viz
)

# Optional: run lm-eval at every checkpoint (remove to skip)
EVAL_ARGS=(
#   --eval_config configs/eval/basic.yaml
)

TORCHRUN_ARGS=(
  --nnodes="${NNODES}"
  --node-rank="${NODE_RANK}"
  --master-addr="${MASTER_ADDR}"
  --master-port="${MASTER_PORT}"
  --nproc-per-node="${NUM_DP_RANKS}"
)

PARALLELISM_ARGS=(
  --num-dp-ranks $NUM_DP_RANKS
)

RUN_DIR_ARGS=(
  --base-run-dir "${BASE_RUN_DIR}"
)

# To resume from a checkpoint, uncomment and set --load to the run directory.
# The latest checkpoint-{N} inside it will be loaded automatically.
# RESUME_ARGS=(
#   --load checkpoints/nanogpt-speedrun-ablations-2026-04-19-01-41-08
#   # --checkpoint checkpoint-2000 # omit to load the latest checkpoint automatically
# )

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run training with both sets of args
echo "Kicking off process"

# python -m arcadium.tasks.language.pretrain \
#   "${MODEL_ARGS[@]}" "${DATA_ARGS[@]}" \
#   "${VIS_ARGS[@]}" "${EVAL_ARGS[@]}" \
#   "${PARALLELISM_ARGS[@]}" "${RUN_DIR_ARGS[@]}" 2>&1 | tee "logs/out_${TIMESTAMP}.log" 

torchrun "${TORCHRUN_ARGS[@]}" \
  -m arcadium.tasks.language.pretrain \
  "${MODEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${VIS_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${PARALLELISM_ARGS[@]}" \
  "${RUN_DIR_ARGS[@]}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee "logs/out_${TIMESTAMP}.log"
  