#!/usr/bin/env bash

# 設定環境變數
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/..":$PYTHONPATH

# ════════════════════════════════════════════════════════
# 實驗 A（GPU 2+3）：原始 CPS — ViT + ResNeXt-101 (RFP)
#   CNN 使用正確的 pretwsiall-next101htc.pth
# ════════════════════════════════════════════════════════
CUDA_VISIBLE_DEVICES=2,3 python train_cps.py \
  --cfg-vit all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2.py \
  --cfg-cnn all_configs/nops_config_pret/htc_resnext101_cps.py \
  --ckpt-cnn pretrained/pretwsiall-next101htc.pth \
  --ckpt-vit pretrained/pretexp1_adaplargebeitv2l_htc-v2.pth \
  --work-dir results/cps/stage1

CUDA_VISIBLE_DEVICES=2,3 python train_cps.py \
  --cfg-vit all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2.py \
  --cfg-cnn all_configs/nops_config_pret/htc_resnext101_cps.py \
  --resume-vit results/cps/stage1/vit_epoch_1.pth \
  --resume-cnn results/cps/stage1/cnn_epoch_1.pth \
  --work-dir results/cps/stage1_resume

# ════════════════════════════════════════════════════════
# 實驗 B（GPU 0+1）：LNN CPS — ViT(LNNHopfieldFPN) + ResNet-50(LNNHopfieldFPN)
#   CNN 換成較小的 ResNet-50，neck 同樣使用 LNNHopfieldFPN
#   預計 VRAM：ViT ~20GB / R50+LNN ~6-8GB（各自 GPU 均可容納）
# ════════════════════════════════════════════════════════
CUDA_VISIBLE_DEVICES=0,1 python train_cps3.py \
  --cfg-vit  all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2_lnnfpn.py \
  --cfg-cnn1 all_configs/nops_config_pret/htc_resnext101_cps.py \
  --cfg-cnn2 all_configs/nops_config_pret/htc_r50_lnnfpn_cps.py \
  --ckpt-vit  pretrained/pretexp1_adaplargebeitv2l_htc-v2.pth \
  --ckpt-cnn1 pretrained/pretwsiall-next101htc.pth \
  --ckpt-cnn2 pretrained/ds1pretexp1moreaug-htc50.pth \
  --work-dir  results/cps3/stage1

# 定義你的訓練指令（已修正末尾換行）
CMD1="python train_cps.py \
  --cfg-vit all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2.py \
  --cfg-cnn all_configs/nops_config_pret/htc_resnext101_cps.py \
  --ckpt-vit pretrained/pretexp1_adaplargebeitv2l_htc-v2.pth \
  --ckpt-cnn pretrained/pretwsiall-next101htc-exp1-augv4-cv409-best-segm.QHhIJOoW.pth.part \
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