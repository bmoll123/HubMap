#!/bin/bash

# Enable CUDA debugging for better error messages
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

CONFIG="all_configs/stage1_2ndplace_data_pt.py"
WORK_DIR="work_dirs/stage1_debug"

mkdir -p "$WORK_DIR"

echo "Starting training with CUDA debugging enabled..."
echo "This will be slower but provide better error messages"

python train.py \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=0

