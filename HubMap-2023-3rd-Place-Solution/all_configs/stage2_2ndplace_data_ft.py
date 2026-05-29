"""
Stage 2: Finetuning using 3rd-Place model architecture (HybridTaskCascade + BEiTAdapter)
Dataset: 2nd-Place 9tiles_crop128 dataset (4 train sets + val/test)
Augmentation: MMDet-native pipeline (no Albu), with StainTransform + PhotoMetricDistortion
"""

NUM_CLASSES = 1
drop_path_rate = 0.2  # Lower drop_path for finetune

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
        window_attn=[True] * 24,
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
            drop_path=0.1,
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
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=NUM_CLASSES,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
            dict(
                type="HTCMaskHead",
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=NUM_CLASSES,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
            dict(
                type="HTCMaskHead",
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=NUM_CLASSES,
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
                match_low_quality=True,
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
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
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
            score_thr=0.001,  # 調低門檻捕捉微弱特徵
            nms=dict(
                type="soft_nms", iou_threshold=0.4
            ),  # 使用 soft_nms 降低密集物件互相壓制的問題
            max_per_img=200,  # 放寬單張上限避免密集區域被截斷
            mask_thr_binary=0.5,
        ),
    ),
)

# ======================== Dataset & Augmentation ========================
dataset_type = "CocoDataset"
data_root = "datasets/"
classes = ("blood_vessel",)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

stn_aug_root = "datasets/stain_9tiles_augs/"
stn_img_ext = ".tif"
margin = 128

# ---------------------------------------------------------------------------
# General train pipeline (MMDet-native, no Albu)
# - StainTransform prob=0.5
# - H+V flip, PhotoMetricDistortion, AutoAugment multi-scale
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="StainTransform",
        img_aug_root=stn_aug_root,
        img_ext=stn_img_ext,
        margin=margin,
        prob=0.5,
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
                    allow_negative_crop=True,
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

# ---------------------------------------------------------------------------
# Same-WSI train pipeline: StainTransform prob=1.0（強制 domain shift）
# ---------------------------------------------------------------------------
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
                    allow_negative_crop=True,
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

# ======================== Data ========================
# train_set1 (ds1_wsi1_right): same-WSI pipeline, StainTransform=1.0
# train_set2~4: general pipeline, StainTransform=0.5
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + "hm_9tiles_crop128_1cls/ds1/ds1_wsi1_right_train.json",
            img_prefix=data_root + "train_9tiles_crop128/",
            classes=classes,
            filter_empty_gt=True,
            pipeline=train_same_wsi_pipeline,
        ),
        dict(
            type=dataset_type,
            ann_file=data_root + "hm_9tiles_crop128_1cls/ds1/ds1_wsi2_left_train.json",
            img_prefix=data_root + "train_9tiles_crop128/",
            classes=classes,
            filter_empty_gt=True,
            pipeline=train_pipeline,
        ),
        dict(
            type=dataset_type,
            ann_file=data_root + "hm_9tiles_crop128_1cls/ds1/ds1_wsi2_right_train.json",
            img_prefix=data_root + "train_9tiles_crop128/",
            classes=classes,
            filter_empty_gt=True,
            pipeline=train_pipeline,
        ),
        dict(
            type=dataset_type,
            ann_file=data_root + "hm_9tiles_crop128_1cls/ds1/ds1_wsi2_ignore.json",
            img_prefix=data_root + "train_9tiles_crop128/",
            classes=classes,
            filter_empty_gt=True,
            pipeline=train_pipeline,
        ),
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + "hm_9tiles_crop128_1cls/ds1/ds1_wsi1_left_val.json",
        img_prefix=data_root + "train_9tiles_crop128/",
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "hm_9tiles_crop128_1cls/ds1/ds1_wsi1_left_val.json",
        img_prefix=data_root + "train_9tiles_crop128/",
        classes=classes,
        pipeline=test_pipeline,
    ),
    persistent_workers=False,
)

# ======================== Optimizer & Schedule ========================
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[12, 14],
)

runner = dict(type="EpochBasedRunner", max_epochs=15)

# ======================== Misc ========================
checkpoint_config = dict(
    interval=1, save_last=True, max_keep_ckpts=5, save_optimizer=False
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
custom_hooks = [dict(type="NumClassCheckHook")]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = "/home/yuyun/Desktop/HubMap/HubMap-2023-3rd-Place-Solution/results/0529/stage1/epoch_12.pth"
resume_from = None
workflow = [("train", 1)]
evaluation = dict(save_best="auto", rule="greater", interval=1, metric=["bbox", "segm"])
work_dir = "./results/stage2_finetune_9tiles"
