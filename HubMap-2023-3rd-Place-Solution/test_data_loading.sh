#!/bin/bash
# Quick test to verify data loading works before training

set -e

echo "================================================"
echo "Testing Data Loading Pipeline"
echo "================================================"

conda activate python3.10

echo ""
echo "1. Checking environment..."
python -c "
import torch
import mmcv
from mmdet.datasets.pipelines import StainTransform
print(f'   PyTorch: {torch.__version__}')
print(f'   MMCV: {mmcv.__version__}')
print(f'   ✓ StainTransform: Available')
"

echo ""
echo "2. Building dataset..."
python -c "
import sys
sys.path.insert(0, '.')
from mmdet.datasets import build_dataset
from mmcv import Config

cfg = Config.fromfile('all_configs/stage1_2ndplace_data_pt.py')
print(f'   Config loaded')

# Try to build dataset
try:
    dataset = build_dataset(cfg.data.train)
    print(f'   ✓ Dataset built: {len(dataset)} total samples')
except Exception as e:
    print(f'   ✗ Dataset build failed: {e}')
    sys.exit(1)
"

echo ""
echo "3. Loading sample data..."
python -c "
import sys
sys.path.insert(0, '.')
from mmdet.datasets import build_dataset
from mmcv import Config
import torch

cfg = Config.fromfile('all_configs/stage1_2ndplace_data_pt.py')
dataset = build_dataset(cfg.data.train)

try:
    data = dataset[0]
    img = data['img']
    bboxes = data['gt_bboxes']
    
    print(f'   Image shape: {img.shape}')
    print(f'   Image dtype: {img.dtype}')
    print(f'   Bboxes shape: {bboxes.shape}')
    print(f'   Bboxes dtype: {bboxes.dtype}')
    
    # Check bbox validity
    if isinstance(bboxes, torch.Tensor) and bboxes.numel() > 0:
        if (bboxes < 0).any():
            print(f'   ⚠ WARNING: Negative bbox coordinates found!')
        else:
            print(f'   ✓ All bbox coordinates valid (non-negative)')
            
        # Check bbox bounds
        img_h, img_w = img.shape[1:3] if len(img.shape) == 3 else img.shape
        max_x = bboxes[:, [0, 2]].max()
        max_y = bboxes[:, [1, 3]].max()
        
        if max_x > img_w or max_y > img_h:
            print(f'   ⚠ WARNING: Bbox exceeds image bounds!')
            print(f'     Image: {img_w}x{img_h}, Max bbox: ({max_x:.0f}, {max_y:.0f})')
        else:
            print(f'   ✓ All bboxes within image bounds')
    
    print(f'   ✓ Sample loaded successfully')
    
except Exception as e:
    print(f'   ✗ Failed to load sample: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "================================================"
echo "✓ Data Loading Test Complete!"
echo "================================================"
echo ""
echo "You can now run training with:"
echo "  conda activate python3.10"
echo "  python train.py all_configs/stage1_2ndplace_data_pt.py"
echo ""

