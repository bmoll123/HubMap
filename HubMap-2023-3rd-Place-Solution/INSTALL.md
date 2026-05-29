
## 環境建立（Environment Setup）

### 系統需求
- OS: Ubuntu 20.04 / 22.04
- GPU: NVIDIA（CUDA 12.1 以上）
- Miniconda / Anaconda

### Step 1：建立 conda 環境

```bash
conda create -n mmd python=3.10 -y
conda activate mmd
```

### Step 2：安裝 PyTorch（CUDA 12.1）

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 3：安裝 setuptools（提供 pkg_resources，mmcv-full 編譯需要）

```bash
pip install "setuptools==69.5.1"
```

### Step 4：從原始碼編譯 mmcv-full 1.7.2（本步驟約需 10-20 分鐘）

> **說明**：PyTorch 2.3.1 + CUDA 12.1 沒有 mmcv-full 1.x 的預建 wheel，必須從原始碼編譯。

```bash
cd /tmp
git clone -b v1.7.2 https://github.com/open-mmlab/mmcv.git mmcv-1.7.2 --depth=1
cd mmcv-1.7.2
MMCV_WITH_OPS=1 python setup.py bdist_wheel 2>&1 | tee /tmp/mmcv_build.log
pip install dist/mmcv_full-1.7.2-*.whl
cd -
```

### Step 5：安裝其他依賴套件

```bash
pip install \
    timm==1.0.27 \
    einops==0.8.1 \
    segmentation-models-pytorch==0.5.0 \
    ensemble-boxes==1.0.9 \
    pycocotools==2.0.11 \
    numpy==2.2.6 \
    opencv-python==4.13.0.92 \
    scipy==1.15.3 \
    pandas==2.3.3 \
    shapely==2.1.2 \
    openmim
```

### Step 6：驗證安裝

```bash
conda activate mmd
cd /path/to/HubMap-2023-3rd-Place-Solution
python -c "
import warnings; warnings.filterwarnings('ignore')
import torch, mmcv
from mmdet.models import build_detector
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('mmcv:', mmcv.__version__)
print('mmdet: OK')
"
```

預期輸出：
```
torch: 2.3.1+cu121 | CUDA: True
mmcv: 1.7.2
mmdet: OK
```

---

## Data preparation
**COCO 格式資料集**
```bash
mkdir -p hubmap-hacking-the-human-vasculature
cd hubmap-hacking-the-human-vasculature
kaggle datasets download -d nischaydnk/hubmap-coco-datasets
unzip hubmap-coco-datasets.zip
rm hubmap-coco-datasets.zip
cd ../..
```

**競賽原始資料集**（train/test 影像、標注檔等）
放到
```bash
hubmap-hacking-the-human-vasculature‵
```

**③ 下載預訓練模型**
```bash
mkdir -p hubmap-coco-pretrained-models
cd hubmap-coco-pretrained-models
kaggle datasets download -d nischaydnk/hubmap-coco-pretrained-models
unzip hubmap-coco-pretrained-models.zip
rm hubmap-coco-pretrained-models.zip
cd ..
```

解壓後目錄結構：
```
HubMap-2023-3rd-Place-Solution/
├── hubmap-coco-pretrained-models/
│   ├── htc++_beitv2_adapter_large_fpn_o365_coco.pth
│   └── ...（其他 .pth 檔）
└── hubmap-hacking-the-human-vasculature/
    ├── train/
    ├── test/
    ├── polygons.jsonl
    ├── tile_meta.csv
    └── coco_data/          ← hubmap-coco-datasets.zip 解壓結果
```