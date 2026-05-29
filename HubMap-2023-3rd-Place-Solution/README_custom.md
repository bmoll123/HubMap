# 專案結構

```
專案根目錄
│
├── mmdet/               ← MMDet 2.26.0 原始碼（直接嵌入，非 pip install）
│   └── models/
│       ├── backbones/
│       ├── necks/       ← 標準 FPN 等原版 neck
│       └── ...
│
├── mmdet_custom/        ← 競賽自訂模組（import 時自動注冊到 mmdet registry）
│   └── models/
│       ├── backbones/   → BEiTAdapter、ViT-Adapter（論文新 backbone）
│       ├── necks/       → ExtraAttention、LNNHopfieldFPN（自訂 neck）
│       └── detectors/   → 自訂 detector
│
├── mmcv_custom/         ← mmcv 行為補丁
│   ├── checkpoint.py    → 自訂權重載入（pretrained 格式轉換）
│   └── layer_decay_optimizer_constructor.py → ViT 專用 layer-wise LR decay
│
├── all_configs/         ← 所有實驗設定
├── configs/             ← MMDet 標準 base config（被 all_configs 繼承）
├── ops/                 ← C++ CUDA 自訂運算子
├── split/               ← 資料前處理 Notebook（kfold、pseudo label 生成）
├── reference_module/    ← 參考模組原始碼（不直接使用，僅供對照）
└── train.py             ← 訓練入口點
```

---

# all_configs 說明

| 資料夾 | 模型 | 階段 | Pseudo Label |
|--------|------|------|:---:|
| `pretconf/` | ViT-Adapter BEiT-v2 Large | Stage 1 | ✗ |
| `nops_config_pret/` | ResNeXt-101、CBNetV2-Base | Stage 1 | ✗ |
| `nops_config_finetune/` | 所有模型 | Stage 2 | ✗ |
| `pseudo_config_pret/` | ViT-Adapter BEiT-v2 Large | Stage 1 | ✓ |
| `pseudo_config_finetune/` | 所有模型 | Stage 2 | ✓ |
| `pseudo_conf/` | - | Pseudo Label 推論 | - |
| `cbnetconfv2/` | CBNetV2-Base | ablation | - |
| `nextconf/` | Detectors-ResNeXt101 | ablation | - |
| `hubconf/` `hubconf2/` `hubconf3/` | 各種嘗試 | ablation | - |

---

# 訓練指令

```bash
conda activate mmd
cd /home/cvml_7/Desktop/2026_class/HubMap/HubMap-2023-3rd-Place-Solution
```

### 原版 Stage 1
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2.py \
    --launcher none --seed 69
```

### 原版 Stage 2
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    all_configs/nops_config_finetune/exp4_adapbeitv2l.py \
    --launcher none --seed 69
```

### LNN Stage 1
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2_lnn.py \
    --launcher none --seed 69
```

### LNN Stage 2（需先完成 LNN Stage 1，並更新 load_from）
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    all_configs/nops_config_finetune/exp4_adapbeitv2l_lnn.py \
    --launcher none --seed 69
```

### Resume 訓練
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2.py \
    --launcher none --seed 69 \
    --resume-from ./results/stage1/best_segm_mAP_epoch_8.pth
```

---

# LNN-Hopfield FPN 說明

**位置**：`mmdet_custom/models/necks/lnn_hopfield_fpn.py`

原版 neck 架構：
```
Backbone → ExtraAttention → FPN → RPN / RoI Head
```

LNN 版 neck 架構：
```
Backbone → ExtraAttention → LNNHopfieldFPN → RPN / RoI Head
                                 ↑
                    Phase 2: CfCCell（P6→P2 Coarse-to-Fine 遞歸）
                    Phase 3: ModernHopfieldLayer（per-level 原型記憶修復）
```

**設計目的**：應對 HubMap 不同 WSI 染色差異導致的 OOD 問題，
讓模型在遇到異常染色切片時仍能穩定分割血管。
