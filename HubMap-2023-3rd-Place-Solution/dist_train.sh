#!/usr/bin/env bash

# 設定環境變數
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/..":$PYTHONPATH

# 設定 GPU ID（方便管理）
GPU_ID=0

# 定義你的訓練指令（已修正末尾換行）
CMD1="python train.py ./all_configs/pretconf/0528_pretexp1_adaplargebeitv2l_htc.py --launcher none --seed 69"

CMD2="python train.py ./all_configs/nops_config_finetune/0528_exp4_adapbeitv2l.py --launcher none --seed 69"

echo "=== 開始執行第一個訓練任務 ==="
CUDA_VISIBLE_DEVICES=$GPU_ID $CMD1

# $? 代表上一個指令的結束狀態，0 代表成功
if [ $? -eq 0 ]; then
    echo "=== 第一個任務成功，準備執行第二個訓練任務 ==="
    CUDA_VISIBLE_DEVICES=$GPU_ID $CMD2
else
    echo "=== 第一個任務失敗，已停止後續動作 ==="
    exit 1
fi

echo "=== 所有訓練任務已完成 ==="