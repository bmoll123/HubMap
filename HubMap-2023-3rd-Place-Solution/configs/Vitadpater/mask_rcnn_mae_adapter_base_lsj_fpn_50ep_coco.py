# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    "../_base_/models/mask_rcnn_r50_fpn.py",
    "../_base_/datasets/coco_instance.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
# pretrained = 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth'
pretrained = "pretrained/mae_pretrain_vit_base.pth"
model = dict(
    backbone=dict(
        _delete_=True,
        type="ViTAdapter",
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.2,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        use_extra_extractor=False,
        layer_scale=False,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
        ],
        window_size=[14, 14, None, 14, 14, None, 14, 14, None, 14, 14, None],
        pretrained=pretrained,
    ),
    neck=dict(
        type="FPN",
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type="MMSyncBN", requires_grad=True),
    ),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type="Shared4Conv1FCBBoxHead",
            norm_cfg=dict(type="MMSyncBN", requires_grad=True),
        ),
        mask_head=dict(norm_cfg=dict(type="MMSyncBN", requires_grad=True)),
    ),
)
# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="Resize",
        img_scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        multiscale_mode="range",
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_type="absolute_range",
        crop_size=(1024, 1024),
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=(1024, 1024)),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]


data_root = "/home/yuyun/Desktop/HubMap/data"  # 請依你的實際 datasets 路徑調整

data = dict(
    samples_per_gpu=2,  # 如果 RTX 4090/5090 顯存夠大(且有開bfloat16)，可以試著調到 4 或 8
    workers_per_gpu=4,
    train=dict(
        type="CocoDataset",
        classes=("blood_vessel",),
        ann_file=data_root
        + "hm_1cls/annotations/train_all.json",  # ds1 + ds2 的完整大 JSON
        img_prefix=data_root + "stain_augs/",  # 染色增強圖資料夾
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CocoDataset",
        classes=("blood_vessel",),
        ann_file=data_root + "hm_1cls/annotations/val_fold0.json",  # 先拿 fold0 當驗證
        img_prefix=data_root + "train/",  # 驗證用原始圖片
        pipeline=dict(  # 簡單定義測試/驗證的 pipeline
            type="MultiScaleFlipAug",
            img_scale=(1024, 1024),
            fn_tuples=[
                dict(type="Resize", keep_ratio=True),
                dict(type="RandomFlip"),
                dict(type="Normalize", **img_norm_cfg),
                dict(type="Pad", size_divisor=32),
                dict(type="ImageToTensor", keys=["img"]),
                dict(type="Collect", keys=["img"]),
            ],
        ),
    ),
)


lr_config = dict(
    _delete_=True,
    policy="CosineAnnealing",
    min_lr_ratio=0.01,
    warmup="linear",
    warmup_iters=2000,
    warmup_ratio=0.001,
)
runner = dict(type="EpochBasedRunner", max_epochs=30)
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "level_embed": dict(decay_mult=0.0),
            "pos_embed": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "bias": dict(decay_mult=0.0),
        }
    ),
)
optimizer_config = dict(grad_clip=None)
fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)
