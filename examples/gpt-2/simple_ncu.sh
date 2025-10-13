#!/usr/bin/env bash

# Define model args
MODEL_ARGS=(
  --model_config configs/models/gpt-2/tiny-gpt.yaml
)

# Define data/training args
DATA_ARGS=(
  --training_config configs/training/gpt-2-medium-data.yaml
)

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Output filenames
NCU_REPORT="logs/ncu_report_$TIMESTAMP.ncu-rep"
CSV_FILE="logs/ncu_stats_$TIMESTAMP.csv"

# Run training with Nsight Compute profiling
ncu --metrics sm__sass_thread_inst_executed_op_fadd.sum,sm__sass_thread_inst_executed_op_fmul.sum,dram__bytes.sum \
    --target-processes all \
    -o "$NCU_REPORT" \
    python pretrain_language.py \
        "${MODEL_ARGS[@]}" \
        "${DATA_ARGS[@]}"

# Export Nsight Compute report to CSV
ncu --import "$NCU_REPORT" --csv --output "$CSV_FILE"

echo "CSV saved to $CSV_FILE"
