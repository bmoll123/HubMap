# datasets/register_hubmap.py
import os
import random
import numpy as np
import cv2

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import detectron2.data.transforms as DT
from detectron2.config import configurable

import torchvision.transforms as tvT

# ──────────────────────────────────────────────
# 1. 自訂 Transform：RandomRotateScaleCrop
# ──────────────────────────────────────────────
class RandomRotateScaleCropTransform(T.Transform):
    def __init__(self, M, out_h, out_w, border_value=(114, 114, 114)):
        self.M = M
        self.out_h = out_h
        self.out_w = out_w
        self.border_value = border_value

    def apply_image(self, img):
        return cv2.warpAffine(
            img, self.M, (self.out_w, self.out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )

    def apply_coords(self, coords):
        # coords: ndarray (N, 2) in x,y order
        n = len(coords)
        ones = np.ones((n, 1), dtype=np.float32)
        coords_h = np.hstack([coords.astype(np.float32), ones])  # (N, 3)
        return coords_h @ self.M.T  # (N, 2)

    def apply_segmentation(self, segmentation):
        return cv2.warpAffine(
            segmentation.astype(np.uint8), self.M, (self.out_w, self.out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def apply_polygons(self, polygons):
        return [self.apply_coords(np.array(p).reshape(-1, 2)) for p in polygons]


class RandomRotateScaleCrop(T.Augmentation):
    def __init__(
        self,
        img_scale=(768, 768),      # (h, w) 輸出尺寸
        angle_range=(-180, 180),
        scale_range=(0.1, 2.0),
        border_value=(114, 114, 114),
        rotate_prob=0.5,
        scale_prob=1.0,
        hflip_prob=0.5,
        rot90_prob=1.0,
    ):
        self.img_scale = img_scale
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.border_value = border_value
        self.rotate_prob = rotate_prob
        self.scale_prob = scale_prob
        self.hflip_prob = hflip_prob
        self.rot90_prob = rot90_prob

    def get_transform(self, image):
        h, w = image.shape[:2]
        out_h, out_w = self.img_scale
        cx, cy = w / 2.0, h / 2.0

        # 連續旋轉
        angle = 0.0
        if random.random() < self.rotate_prob:
            angle = random.uniform(*self.angle_range)
        # rot90 疊加
        if random.random() < self.rot90_prob:
            angle += random.choice([0, 90, 180, 270])

        # 縮放
        scale = 1.0
        if random.random() < self.scale_prob:
            scale = random.uniform(*self.scale_range)

        # 建立 affine 矩陣（旋轉+縮放，平移讓中心對齊輸出中心）
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[0, 2] += out_w / 2.0 - cx
        M[1, 2] += out_h / 2.0 - cy

        # 水平翻轉疊加進 affine
        if random.random() < self.hflip_prob:
            flip_M = np.float32([[-1, 0, out_w - 1],
                                  [ 0, 1, 0]])
            M3 = np.vstack([M, [0, 0, 1]])
            flip3 = np.vstack([flip_M, [0, 0, 1]])
            M = (flip3 @ M3)[:2]

        return RandomRotateScaleCropTransform(
            M, out_h, out_w, self.border_value
        )


# ──────────────────────────────────────────────
# 2. 自訂 DatasetMapper
# ──────────────────────────────────────────────
class HuBMAPDatasetMapper:
    """
    接管 DI-MaskDINO 的 training mapper，
    把 RandomRotateScaleCrop 注入進去。
    """
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT  # "RGB"
        self.use_instance_mask = cfg.MODEL.MASK_ON

        out_h = cfg.INPUT.HUBMAP_IMG_SIZE   # 自訂 key，見 yaml
        out_w = cfg.INPUT.HUBMAP_IMG_SIZE

        if is_train:
            self.augmentations = T.AugmentationList([
                RandomRotateScaleCrop(
                    img_scale=(out_h, out_w),
                    angle_range=(-180, 180),
                    scale_range=(0.1, 2.0),
                    border_value=(114, 114, 114),
                    rotate_prob=0.5,
                    scale_prob=1.0,
                    hflip_prob=0.5,
                    rot90_prob=1.0,
                ),
                # 病理影像 color jitter（染色差異大）
                T.RandomBrightness(0.6, 1.4),
                T.RandomContrast(0.6, 1.4),
                T.RandomSaturation(0.8, 1.2),
                            ])
        else:
            # val：只 resize 到固定尺寸，不做隨機變換
            self.augmentations = T.AugmentationList([
                T.ResizeShortestEdge(
                    short_edge_length=out_h,
                    max_size=out_w,
                    sample_style="choice",
                ),
            ])

    def __call__(self, dataset_dict):
        import copy
        import torch
        from detectron2.structures import (
            BitMasks, Boxes, BoxMode, Instances, polygons_to_bitmask
        )

        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        h, w = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image.shape[:2]
            )
            for obj in dataset_dict.pop("annotations", [])
            if obj.get("iscrowd", 0) == 0
        ]

        instances = utils.annotations_to_instances(
            annos, image.shape[:2], mask_format="bitmask"
        )
        instances = utils.filter_empty_instances(instances)
        
        # 🟢 修正後：將特殊的 BitMasks 物件轉換為模型預期的標準 torch.Tensor
        if hasattr(instances, "gt_masks"):
            # BitMasks.tensor 儲存的是 bool 型態矩陣 (N, H, W)
            # 轉換成 float32 或適合模型的型態（依你的二進位遮罩需求，一般轉成 float 較安全）
            instances.gt_masks = instances.gt_masks.tensor.to(dtype=torch.float32)
            
        dataset_dict["instances"] = instances
        return dataset_dict


# ──────────────────────────────────────────────
# 3. 註冊 HuBMAP dataset
# ──────────────────────────────────────────────
DATA_ROOT = "/home/cvml-3/yy/114_2/HubMap/data"
IMG_ROOT  = os.path.join(DATA_ROOT, "train")
COCO_ROOT = os.path.join(DATA_ROOT)

_SPLITS = {
    "hubmap_train_fold0": ("dtrain0i.json", IMG_ROOT),
    "hubmap_val_fold0":   ("dval0i.json",   IMG_ROOT),
    "hubmap_train_fold1": ("dtrain1i.json", IMG_ROOT),
    "hubmap_val_fold1":   ("dval1i.json",   IMG_ROOT),
    # Stage 1 noisy pretraining
    "hubmap_dataset2":    ("dtrain_dataset2_dropdup.json", IMG_ROOT),
    # 最終提交用（全量）
    "hubmap_trainval":    ("dtrainval.json", IMG_ROOT),
}

for name, (json_file, img_dir) in _SPLITS.items():
    register_coco_instances(
        name,
        {},  # metadata 從 JSON 裡的 categories 自動讀取
        os.path.join(COCO_ROOT, json_file),
        img_dir,
    )