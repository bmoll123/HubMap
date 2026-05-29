# Training 3rd-Place Model on 2nd-Place Dataset

本說明文檔指導如何用 **3rd-Place 的模型架構和訓練框架**，在 **2nd-Place 的數據集**上進行訓練。

## 配置文件說明

### 新建的配置文件

1. **`all_configs/stage1_2ndplace_data_pt.py`** - Stage 1 Pretraining
   - 訓練時長：30 epochs
   - 數據組成：
     - Dataset 1 (DS1) 的 3 個 fold
     - Dataset 2 (DS2) 的全部數據
   - 預訓練權重：`./hubmap-coco-pretrained-models/htc++_beitv2_adapter_large_fpn_o365_coco.pth`

2. **`all_configs/stage2_2ndplace_data_ft.py`** - Stage 2 Finetuning
   - 訓練時長：15 epochs
   - 數據組成：只使用 DS1 的 3 個 fold

## 數據路徑配置

所有數據預期位置：`~/Desktop/HubMap/data/`

## 快速開始

### 方式 1：使用提供的訓練腳本（推薦）

```bash
cd ~/Desktop/HubMap/HubMap-2023-3rd-Place-Solution
bash train_2ndplace_data.sh
```

### 方式 2：手動執行訓練命令

**Stage 1:**
```bash
cd ~/Desktop/HubMap/HubMap-2023-3rd-Place-Solution
CUDA_VISIBLE_DEVICES=0 python train.py ./all_configs/stage1_2ndplace_data_pt.py --launcher none --seed 69
```

**Stage 2:**
```bash
CKPT=$(find ./work_dirs/stage1_2ndplace_data_pt -name "*.pth" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
CUDA_VISIBLE_DEVICES=0 python train.py ./all_configs/stage2_2ndplace_data_ft.py --launcher none --seed 69 --resume-from $CKPT
```

## 輸出目錄結構

訓練完成後，模型和日誌將保存在：

```
work_dirs/
├── stage1_2ndplace_data_pt/
│   ├── epoch_1.pth
│   ├── epoch_2.pth
│   └── latest.pth
└── stage2_2ndplace_data_ft/
    ├── epoch_1.pth
    ├── epoch_2.pth
    └── latest.pth  ← 最終模型
```

## 故障排除

### 1. 找不到數據文件
```bash
ls ~/Desktop/HubMap/data/hm_1cls/ds1/
ls ~/Desktop/HubMap/data/dtrain_dataset2_dropdup.json
```

### 2. GPU 記憶體不足
降低配置文件中的 `samples_per_gpu`（從 2 改為 1）

### 3. 預訓練權重不存在
確保下載到：`./hubmap-coco-pretrained-models/`
