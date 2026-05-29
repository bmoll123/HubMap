"""
Stage 1: Pretraining using 3rd-Place model architecture on 2nd-Place dataset
Dataset 1 (3 folds) + Dataset 2 + Native High-Performance Pathology Augmentation
"""

import os
import glob

NUM_CLASSES = 1
drop_path_rate = 0.3

pretrained = (
    "./hubmap-coco-pretrained-models/htc++_beitv2_adapter_large_fpn_o365_coco.pth"
)
model = dict(
    type="HybridTaskCascade",
    backbone=dict(
        type="BEiTAdapter",
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=drop_path_rate,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        window_attn=[
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
        window_size=[
            14,
            14,
            14,
            14,
            14,
            56,
            14,
            14,
            14,
            14,
            14,
            56,
            14,
            14,
            14,
            14,
            14,
            56,
            14,
            14,
            14,
            14,
            14,
            56,
        ],
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        pretrained=None,
    ),
    neck=[
        dict(
            type="ExtraAttention",
            in_channels=[1024, 1024, 1024, 1024],
            num_head=32,
            with_ffn=True,
            with_cp=True,
            ffn_ratio=4.0,
            drop_path=drop_path_rate,
        ),
        dict(
            type="PAFPN",
            in_channels=[1024, 1024, 1024, 1024],
            norm_cfg=dict(type="GN", num_groups=32),
            out_channels=256,
            num_outs=5,
        ),
    ],
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
    ),
    roi_head=dict(
        type="HybridTaskCascadeRoIHead",
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=[
            dict(
                type="Shared4Conv1FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
            ),
            dict(
                type="Shared4Conv1FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
            ),
            dict(
                type="Shared4Conv1FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
            ),
        ],
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=[
            dict(
                type="HTCMaskHead",
                with_conv_res=False,  # 已經修正為 True 允許特徵串聯
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
            dict(
                type="HTCMaskHead",
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
            dict(
                type="HTCMaskHead",
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
        ],
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                # match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=2,
        ),
        mask_size=28,
        rcnn=[
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
        ],
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.001,  # 原本 0.05
            nms=dict(type="soft_nms", iou_threshold=0.5),  # 原本 nms
            max_per_img=200,  # 原本 100
            mask_thr_binary=0.5,
        ),
    ),
)

# ======================== 核心資料集與 MMDet 原生增強設定 ========================
dataset_type = "CocoDataset"
data_root = "../data/"
classes = ("blood_vessel",)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

stn_aug_root = "../data/stain_9tiles_augs/"
stn_img_ext = ".tif"
margin = 128

# 1. 通用原生高效 Pipeline（徹底移除 Albu，換上 MMDet 原生光學、翻轉與隨機擴展）
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="BboxClipper"),
    dict(
        type="StainTransform",
        img_aug_root=stn_aug_root,
        img_ext=stn_img_ext,
        margin=margin,
        prob=0.5,
    ),
    # 對應 RandomRotate90 + 原本的 RandomFlip
    dict(type="RandomFlip", direction="horizontal", flip_ratio=0.5),
    dict(type="RandomFlip", direction="vertical", flip_ratio=0.5),
    # 對應 RandomBrightnessContrast + HueSaturationValue
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (800, 1333),
                        (832, 1333),
                        (864, 1333),
                        (896, 1333),
                        (928, 1333),
                        (960, 1333),
                        (992, 1333),
                        (1024, 1333),
                        (1056, 1333),
                        (1088, 1333),
                        (1120, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    img_scale=[(900, 1333), (1000, 1333), (1100, 1333)],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(768, 900),
                    allow_negative_crop=False,  # ← 改回 True
                ),
                dict(
                    type="Resize",
                    img_scale=[
                        (800, 1333),
                        (832, 1333),
                        (864, 1333),
                        (896, 1333),
                        (928, 1333),
                        (960, 1333),
                        (992, 1333),
                        (1024, 1333),
                        (1056, 1333),
                        (1088, 1333),
                        (1120, 1333),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]

# 2. 同 WSI 特化高級 Pipeline
train_same_wsi_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="StainTransform",
        img_aug_root=stn_aug_root,
        img_ext=stn_img_ext,
        margin=margin,
        prob=1.0,
    ),
    dict(type="RandomFlip", direction="horizontal", flip_ratio=0.5),
    dict(type="RandomFlip", direction="vertical", flip_ratio=0.5),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (800, 1333),
                        (832, 1333),
                        (864, 1333),
                        (896, 1333),
                        (928, 1333),
                        (960, 1333),
                        (992, 1333),
                        (1024, 1333),
                        (1056, 1333),
                        (1088, 1333),
                        (1120, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    img_scale=[(900, 1333), (1000, 1333), (1100, 1333)],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(768, 900),
                    allow_negative_crop=False,  # 👈 強制關閉
                ),
                dict(
                    type="Resize",
                    img_scale=[
                        (800, 1333),
                        (832, 1333),
                        (864, 1333),
                        (896, 1333),
                        (928, 1333),
                        (960, 1333),
                        (992, 1333),
                        (1024, 1333),
                        (1056, 1333),
                        (1088, 1333),
                        (1120, 1333),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
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

# ======================== 核心 Data 區塊 (加入 min_size 資料洗滌機制) ========================
# ======================== 核心 Data 區塊 (對齊 8 個 WSI 特化資料集) ========================
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,  # 排查階段保持 0 穩定，能跑了再調高
    train=dict(
        type="ConcatDataset",
        datasets=[
            # train_set1: ds1_wsi1_right (使用 Same WSI Pipeline)
            dict(
                type=dataset_type,
                ann_file=data_root
                + "hm_9tiles_crop128_1cls/ds1/ds1_wsi1_right_train.json",
                img_prefix=data_root + "train_9tiles_crop128/",
                classes=classes,
                filter_empty_gt=True,
                pipeline=train_same_wsi_pipeline,
            ),
            # train_set2: ds1_wsi2_left
            dict(
                type=dataset_type,
                ann_file=data_root
                + "hm_9tiles_crop128_1cls/ds1/ds1_wsi2_left_train.json",
                img_prefix=data_root + "train_9tiles_crop128/",
                classes=classes,
                filter_empty_gt=True,
                pipeline=train_pipeline,
            ),
            # train_set3: ds1_wsi2_right
            dict(
                type=dataset_type,
                ann_file=data_root
                + "hm_9tiles_crop128_1cls/ds1/ds1_wsi2_right_train.json",
                img_prefix=data_root + "train_9tiles_crop128/",
                classes=classes,
                filter_empty_gt=True,
                pipeline=train_pipeline,
            ),
            # train_set4: ds1_wsi2_ignore
            dict(
                type=dataset_type,
                ann_file=data_root + "hm_9tiles_crop128_1cls/ds1/ds1_wsi2_ignore.json",
                img_prefix=data_root + "train_9tiles_crop128/",
                classes=classes,
                filter_empty_gt=True,
                pipeline=train_pipeline,
            ),
            # train_set5: ds2_wsi1 (使用 Same WSI Pipeline)
            dict(
                type=dataset_type,
                ann_file=data_root + "hm_9tiles_crop128_1cls/ds2/ds2_wsi1.json",
                img_prefix=data_root + "train_9tiles_crop128/",
                classes=classes,
                filter_empty_gt=True,
                pipeline=train_same_wsi_pipeline,
            ),
            # train_set6: ds2_wsi2
            dict(
                type=dataset_type,
                ann_file=data_root + "hm_9tiles_crop128_1cls/ds2/ds2_wsi2.json",
                img_prefix=data_root + "train_9tiles_crop128/",
                classes=classes,
                filter_empty_gt=True,
                pipeline=train_pipeline,
            ),
            # train_set7: ds2_wsi3
            dict(
                type=dataset_type,
                ann_file=data_root + "hm_9tiles_crop128_1cls/ds2/ds2_wsi3.json",
                img_prefix=data_root + "train_9tiles_crop128/",
                classes=classes,
                filter_empty_gt=True,
                pipeline=train_pipeline,
            ),
            # train_set8: ds2_wsi4
            dict(
                type=dataset_type,
                ann_file=data_root + "hm_9tiles_crop128_1cls/ds2/ds2_wsi4.json",
                img_prefix=data_root + "train_9tiles_crop128/",
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

# 優化與調度器設定
# [FIX] lr 0.03 -> 0.02，配合真正退火到 1e-4
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    type="GradientCumulativeOptimizerHook",
    cumulative_iters=6,
    grad_clip=dict(max_norm=35, norm_type=2),
)
lr_config = dict(
    policy="CosineAnnealing",
    by_epoch=False,
    warmup="linear",
    warmup_iters=250,
    warmup_ratio=0.001,
    min_lr=1e-4,
)
runner = dict(type="EpochBasedRunner", max_epochs=12)

# 雜項系統參數
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
custom_hooks = [dict(type="NumClassCheckHook")]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = pretrained
resume_from = None
workflow = [("train", 1)]
evaluation = dict(interval=1, metric=["bbox", "segm"])
work_dir = "./results/0529/stage1"
