#!/usr/bin/env bash

# 設定環境變數
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/..":$PYTHONPATH

# 設定 GPU ID（方便管理）
GPU_ID=0

# 定義你的訓練指令（已修正末尾換行）
CMD1="python train_cps.py \
  --cfg-vit all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2.py \
  --cfg-cnn all_configs/nops_config_pret/htc_resnext101_cps.py \
  --ckpt-vit results/stage1/best_segm_mAP_epoch_8.pth \
  --ckpt-cnn hubmap-coco-pretrained-models/detec_htcres101x32_pretcoco.pth \
  --work-dir results/cps/stage1"

CMD2="python train_cps.py \
  --cfg-vit all_configs/nops_config_finetune/exp4_adapbeitv2l.py \
  --cfg-cnn all_configs/nops_config_finetune/pretwsiallhtc_resnext101_exp3_augv4_maskloss4.py \
  --ckpt-vit results/cps/stage1/vit_epoch_8.pth \
  --ckpt-cnn results/cps/stage1/cnn_epoch_8.pth \
  --unl-ann-file hubmap-hacking-the-human-vasculature/coco_data/coco/ds2wsiall_coco_1024_train_fold1.json \
  --max-epochs 23 \
  --work-dir results/cps/stage2"

echo "=== 開始執行第一個訓練任務 ==="
CUDA_VISIBLE_DEVICES=$GPU_ID $CMD1

if [ $? -eq 0 ]; then
    echo "=== 第一個任務成功，準備執行第二個訓練任務 ==="
    CUDA_VISIBLE_DEVICES=$GPU_ID $CMD2
else
    echo "=== 第一個任務失敗，已停止後續動作 ==="
    exit 1
fi

echo "=== 所有訓練任務已完成 ==="