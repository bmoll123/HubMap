import os
import glob
NUM_CLASSES = 1
drop_path_rate = 0.2
model = dict(
    type='HybridTaskCascade',
    backbone=dict(
        type='BEiTAdapter',
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-06,
        drop_path_rate=0.2,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        window_attn=[
            True, True, True, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True, True, True, True, True,
            True, True
        ],
        window_size=[
            14, 14, 14, 14, 14, 56, 14, 14, 14, 14, 14, 56, 14, 14, 14, 14, 14,
            56, 14, 14, 14, 14, 14, 56
        ],
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        pretrained=None),
    neck=[
        dict(
            type='ExtraAttention',
            in_channels=[1024, 1024, 1024, 1024],
            num_head=32,
            with_ffn=True,
            with_cp=True,
            ffn_ratio=4.0,
            drop_path=0.2),
        dict(
            type='FPN',
            in_channels=[1024, 1024, 1024, 1024],
            norm_cfg=dict(type='GN', num_groups=32),
            out_channels=256,
            num_outs=5)
    ],
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='HybridTaskCascadeRoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            # [FIX] soft_nms iou_threshold 0.5 -> 0.4，降低對密集血管的壓制
            nms=dict(type='soft_nms', iou_threshold=0.4),
            # [FIX] max_per_img 100 -> 200，避免 dense tile 截斷
            max_per_img=200,
            mask_thr_binary=0.5)))

data_root = '/home/cvml-3/yy/114_2/HubMap/HubMap-2023-3rd-Place-Solution/hubmap-hacking-the-human-vasculature'
metainfo = dict(classes=('blood_vessels', ), palette=[(220, 20, 60)])
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = 1400

# [FIX] 加入 stain augmentation 到 Stage 2 的 Albu transforms
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.15,
        rotate_limit=15,
        p=0.4),
    dict(type='RandomRotate90', p=0.4),
    # [FIX] stain augmentation
    dict(type='HueSaturationValue',
         hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
    dict(type='CLAHE', clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
    dict(type='GaussianBlur', blur_limit=(3, 5), p=0.2),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=[(1400, 1400)], keep_ratio=True),
    dict(
        type='RandomFlip',
        direction=['horizontal', 'vertical'],
        flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Shear',
            'prob': 0.4,
            'level': 0
        }], [{
            'type': 'Translate',
            'prob': 0.4,
            'level': 5
        }],
                  [{
                      'type': 'PhotoMetricDistortion',
                      'brightness_delta': 32,
                      'contrast_range': (0.5, 1.5),
                      'hue_delta': 18
                  }],
                  [{
                      'type': 'MinIoURandomCrop',
                      'min_ious': (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                      'min_crop_size': 0.3
                  }],
                  [{
                      'type':
                      'CutOut',
                      'n_holes': (5, 10),
                      'cutout_shape': [(4, 4), (4, 8), (8, 4), (8, 8),
                                       (16, 32), (32, 16), (32, 32), (32, 48),
                                       (48, 32), (48, 48)]
                  }],
                  [{
                      'type': 'BrightnessTransform',
                      'prob': 0.6,
                      'level': 4
                  }, {
                      'type': 'ContrastTransform',
                      'prob': 0.6,
                      'level': 6
                  }, {
                      'type': 'Rotate',
                      'prob': 0.6,
                      'level': 10
                  }],
                  [{
                      'type': 'ColorTransform',
                      'prob': 1.0,
                      'level': 6
                  }, {
                      'type': 'EqualizeTransform'
                  }]]),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes', gt_masks='masks'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1400, 1400)],
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


# 1. 定義資料集的相對路徑（相對於執行訓練腳本的專案根目錄）
data_root = 'hubmap-hacking-the-human-vasculature'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    pin_memory=True,
    drop_last=False,
    train=dict(
        type='CocoDataset',
        data_root=data_root,
        classes=('blood_vessels', ),
        ann_file='coco_data/coco/ds1_coco_1024_train_all_fold1.json',
        img_prefix='train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', img_scale=[(1400, 1400)], keep_ratio=True),
            dict(
                type='RandomFlip',
                direction=['horizontal', 'vertical'],
                flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type': 'Shear',
                    'prob': 0.4,
                    'level': 0
                }], [{
                    'type': 'Translate',
                    'prob': 0.4,
                    'level': 5
                }],
                          [{
                              'type': 'PhotoMetricDistortion',
                              'brightness_delta': 32,
                              'contrast_range': (0.5, 1.5),
                              'hue_delta': 18
                          }],
                          [{
                              'type': 'MinIoURandomCrop',
                              'min_ious': (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                              'min_crop_size': 0.3
                          }],
                          [{
                              'type':
                              'CutOut',
                              'n_holes': (5, 10),
                              'cutout_shape': [(4, 4), (4, 8), (8, 4), (8, 8),
                                               (16, 32), (32, 16), (32, 32),
                                               (32, 48), (48, 32), (48, 48)]
                          }],
                          [{
                              'type': 'BrightnessTransform',
                              'prob': 0.6,
                              'level': 4
                          }, {
                              'type': 'ContrastTransform',
                              'prob': 0.6,
                              'level': 6
                          }, {
                              'type': 'Rotate',
                              'prob': 0.6,
                              'level': 10
                          }],
                          [{
                              'type': 'ColorTransform',
                              'prob': 1.0,
                              'level': 6
                          }, {
                              'type': 'EqualizeTransform'
                          }]]),
            dict(
                type='Albu',
                # [FIX] 使用含 stain aug 的 albu_train_transforms
                transforms=albu_train_transforms,
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_bboxes='bboxes', gt_masks='masks'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='CocoDataset',
        data_root=data_root,
        classes=('blood_vessels', ),
        ann_file='coco_data/coco/ds1_coco_1024_valid_all_fold1.json',
        img_prefix='train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1400, 1400)],
                flip=True,
                flip_direction=['horizontal', 'vertical'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        data_root=data_root,
        classes=('blood_vessels', ),
        ann_file='coco_data/coco/ds12_coco_1024_valid_all_fold1.json',
        img_prefix='train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1400, 1400)],
                flip=True,
                flip_direction=['horizontal', 'vertical'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

optimizer = dict(type='SGD', lr=0.0125, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=6,
    grad_clip=dict(max_norm=35, norm_type=2)
)
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    min_lr=1e-08)
evaluation = dict(interval=1, metric=['segm'], save_best='segm_mAP')
runner = dict(type='EpochBasedRunner', max_epochs=23)
checkpoint_config = dict(interval=-1, filename_tmpl='detectors_epoch_{}.pth')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
fp16 = None
gpu_ids = range(0, 1)
seed = 69
dist_params = dict(backend='nccl')
log_level = 'INFO'

# 2. [FIX] 透過 Python 萬用字元動態自動尋找最新的 stage1 權重檔案
stage1_ckpts = sorted(glob.glob('results/stage1/best_segm_mAP_epoch_*.pth'))
if stage1_ckpts:
    load_from = stage1_ckpts[-1]  # 自動指派排序最後一個（通常是 Epoch 號碼最大的）
else:
    # 如果找不到任何萬用匹配，就留預設相對路徑
    load_from = 'results/stage1/best_segm_mAP_epoch_8.pth'

work_dir = './results/stage2'
workflow = [('train', 1)]
auto_resume = False
resume_from = None
launcher = 'none'
device = 'cuda'