"""
Stage 1: Pretraining using 3rd-Place model architecture on 2nd-Place dataset
Dataset 1 (3 folds) + Dataset 2 + High-Performance Pathology Augmentation
(WITHOUT StainTransform for debugging CUDA errors)
"""

import os
import glob

NUM_CLASSES = 1
drop_path_rate = 0.3

pretrained = (
    "./hubmap-coco-pretrained-models/htc++_beitv2_adapter_large_fpn_o365_coco.pth"
)

# Model config unchanged (see original file for full model definition)
# Using simplified reference - copy from stage1_2ndplace_data_pt.py if needed
model = dict(
    type="HybridTaskCascade",
    backbone=dict(
        type="BEiTAdapter",
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
    ),
)

dataset_type = "CocoDataset"
data_root = "../data/"
classes = ("blood_vessel",)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

albu_train_transforms = [
    dict(type="RandomRotate90", p=0.5),
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="ElasticTransform", alpha=120, sigma=6.0, alpha_affine=3.6, p=1.0
            ),
            dict(type="GridDistortion", p=1.0),
            dict(type="OpticalDistortion", distort_limit=2, shift_limit=0.5, p=1.0),
        ],
        p=0.5,
    ),
]

# *** KEY CHANGE: Removed StainTransform from pipeline ***
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    # NO StainTransform here!
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type=dataset_type,
                ann_file=data_root + "hm_1cls/ds1/ds1_wsi1_right.json",
                img_prefix=data_root + "train/",
                classes=classes,
                filter_empty_gt=True,
                pipeline=train_pipeline,
            ),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "hm_1cls/ds1/ds1_wsi1_left.json",
        img_prefix=data_root + "train/",
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "hm_1cls/ds1/ds1_wsi1_left.json",
        img_prefix=data_root + "train/",
        classes=classes,
        pipeline=test_pipeline,
    ),
    persistent_workers=False,
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[24, 28]
)
runner = dict(type="EpochBasedRunner", max_epochs=30)

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
custom_hooks = [dict(type="NumClassCheckHook")]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = pretrained
resume_from = None
workflow = [("train", 1)]
evaluation = dict(interval=1, metric=["bbox", "segm"])
