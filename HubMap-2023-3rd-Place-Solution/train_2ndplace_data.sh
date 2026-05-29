#!/usr/bin/env bash

# 設定環境變數
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/..":$PYTHONPATH

# 設定 GPU ID（方便管理）
GPU_ID=0

# 改到訓練目錄
cd ~/Desktop/HubMap/HubMap-2023-3rd-Place-Solution

echo "=== HubMap 3rd-Place Model Training on 2nd-Place Dataset ==="
echo "Using data from: ~/Desktop/HubMap/data"
echo ""

# 定義你的訓練指令
echo "=== Stage 1: Pretraining for 30 epochs (DS1 3-folds + all DS2) ==="
CMD1="python train.py ./all_configs/stage1_2ndplace_data_pt.py --launcher none --seed 69"

echo "Executing: $CMD1"
echo ""
CUDA_VISIBLE_DEVICES=$GPU_ID $CMD1

# $? 代表上一個指令的結束狀態，0 代表成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Stage 1 completed successfully ==="
    echo ""
    
    # Find the latest checkpoint from Stage 1
    LATEST_CKPT=$(find ./work_dirs/stage1_2ndplace_data_pt -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$LATEST_CKPT" ]; then
        echo "Warning: Could not find Stage 1 checkpoint. Using default assumption."
        LATEST_CKPT="./work_dirs/stage1_2ndplace_data_pt/latest.pth"
    fi
    
    echo "Using checkpoint: $LATEST_CKPT"
    echo ""
    echo "=== Stage 2: Finetuning for 15 epochs (DS1 3-folds only) ==="
    
    CMD2="python train.py ./all_configs/stage2_2ndplace_data_ft.py --launcher none --seed 69 --resume-from $LATEST_CKPT"
    
    echo "Executing: $CMD2"
    echo ""
    CUDA_VISIBLE_DEVICES=$GPU_ID $CMD2
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=== All training stages completed successfully! ==="
    else
        echo ""
        echo "=== Stage 2 training failed ==="
        exit 1
    fi
else
    echo ""
    echo "=== Stage 1 training failed, aborting Stage 2 ==="
    exit 1
fi

echo ""
echo "=== Training pipeline finished ==="
echo "Stage 1 checkpoints: ./work_dirs/stage1_2ndplace_data_pt/"
echo "Stage 2 checkpoints: ./work_dirs/stage2_2ndplace_data_ft/"
