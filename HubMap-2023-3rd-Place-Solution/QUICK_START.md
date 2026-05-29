# Quick Start Guide - 3rd-Place Model on 2nd-Place Data

## 🚀 一鍵訓練（推薦）

```bash
cd ~/Desktop/HubMap/HubMap-2023-3rd-Place-Solution
bash train_2ndplace_data.sh
```

該命令會自動執行：
1. **Stage 1**：30 epochs 預訓練（DS1 3-folds + all DS2）
2. **Stage 2**：15 epochs 微調（DS1 3-folds only）

---

## 📋 訓練配置

### Stage 1 - Pretraining (30 epochs)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./all_configs/stage1_2ndplace_data_pt.py --launcher none --seed 69
```

**配置文件**：`all_configs/stage1_2ndplace_data_pt.py`

**數據使用**：
- Dataset 1 (3 folds): `ds1_wsi1_right.json`, `ds1_wsi2_left.json`, `ds1_wsi2_right.json`
- Dataset 2 (全部): `dtrain_dataset2_dropdup.json`

---

### Stage 2 - Finetune (15 epochs)
```bash
# 自動查找 Stage 1 checkpoint
CKPT=$(find ./work_dirs/stage1_2ndplace_data_pt -name "*.pth" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

CUDA_VISIBLE_DEVICES=0 python train.py ./all_configs/stage2_2ndplace_data_ft.py --launcher none --seed 69 --resume-from $CKPT
```

**配置文件**：`all_configs/stage2_2ndplace_data_ft.py`

**數據使用**：
- Dataset 1 (3 folds only): `ds1_wsi1_right.json`, `ds1_wsi2_left.json`, `ds1_wsi2_right.json`

---

## ✅ 前置檢查清單

在開始訓練前，請確認：

- [ ] 數據存在：`ls ~/Desktop/HubMap/data/hm_1cls/ds1/`
- [ ] 數據存在：`ls ~/Desktop/HubMap/data/dtrain_dataset2_dropdup.json`
- [ ] 數據存在：`ls ~/Desktop/HubMap/data/dval0i.json`
- [ ] 數據存在：`ls ~/Desktop/HubMap/data/train/` （包含 .tif 圖像）
- [ ] 預訓練權重：`ls ~/Desktop/HubMap/HubMap-2023-3rd-Place-Solution/hubmap-coco-pretrained-models/`
- [ ] GPU 可用：`nvidia-smi`

---

## 📊 訓練輸出

最終模型權重：`work_dirs/stage2_2ndplace_data_ft/latest.pth`

---

## ⚠️ 常見問題

### Q1: GPU 記憶體不足
**A**: 編輯配置文件，降低 `samples_per_gpu`

### Q2: 找不到數據
**A**: 確認 `~/Desktop/HubMap/data/` 包含所有必要文件

### Q3: 如何恢復訓練
**A**: 使用 `--resume-from` 參數

---

## 📝 相關文件

| 文件 | 位置 |
|------|------|
| Stage 1 配置 | `all_configs/stage1_2ndplace_data_pt.py` |
| Stage 2 配置 | `all_configs/stage2_2ndplace_data_ft.py` |
| 訓練腳本 | `train_2ndplace_data.sh` |
| 詳細文檔 | `TRAINING_2ND_PLACE_DATA_README.md` |
