# =========================================================================
# 🌟 Kaggle 離線提交專用：RTMDet-x (CSPNeXt) + MultiEMA Inference 配置
# =========================================================================

norm_cfg = dict(type="BN")

# --- 內部核心 Detector 配置 ---
detector_cfg = dict(
    type="RTMDetWithMaskHead",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        batch_augments=None,
    ),
    mask_head=dict(
        type="FCNMaskHead",
        num_convs=7,
        in_channels=320,
        conv_out_channels=256,
        num_classes=1,
    ),
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=1.33,
        widen_factor=1.25,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    neck=dict(
        type="CSPNeXtPAFPN",
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    bbox_head=dict(
        type="RTMDetSepBNHead",
        num_classes=3,
        in_channels=320,
        stacked_convs=2,
        feat_channels=320,
        anchor_generator=dict(type="MlvlPointGenerator", offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type="DistancePointBBoxCoder"),
        loss_cls=dict(
            type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    train_cfg=dict(
        mask_pos_mode="weighted_sum",
        mask_roi_size=28,
        assigner=dict(type="IgnoreMaskDynamicSoftLabelAssigner", topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        hflip_tta=False,
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type="nms", iou_threshold=0.65),
        max_per_img=300,
    ),
    init_cfg=dict(
        type="Pretrained",
        checkpoint=(
            "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
        ),
    ),
)

# --- 外層 Multi-EMA 包裝 ---
model = dict(
    type="MultiEMADetector",
    momentums=[0.001, 0.0005, 0.00025],
    detector=detector_cfg,
)

# =========================================================================
# 🎯 Kaggle 離線提交專用：資料集與測試載入器配置
# =========================================================================
dataset_type = "CocoDataset"
# 1. 將資料根目錄改為工作區，也就是你在 Kaggle 生成假 JSON 的地方
data_root = "/kaggle/working/"
# 2. 🟢 關鍵修正：將測試圖片路徑精確指向 Kaggle 官方的真實離線測試集影像資料夾
img_prefix = "/kaggle/input/hubmap-hacking-the-human-vasculature/test/"
metainfo = dict(classes=("blood_vessel", "glomerulus", "unsure"))
backend_args = None

img_scale = (768, 768)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# 3. 🌟 徹底解耦！重寫專屬於測試的 test_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="data/dval0i.json",  # 指向你在 /kaggle/working/data/ 下建立的假標籤
        data_prefix=dict(img=img_prefix),  # 完美指向 /kaggle/input/.../test/
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

# 4. 評估器路徑對齊
test_evaluator = [
    dict(
        type="FastCocoMetric",
        ann_file=test_dataloader["dataset"]["data_root"]
        + test_dataloader["dataset"]["ann_file"],
        metric=["bbox", "segm"],
        classwise=True,
        format_only=False,
        backend_args=backend_args,
    ),
]

# --- 系統排程與迴圈設定，切換為純 Test 推理模式 ---
train_cfg = None
train_dataloader = None
val_cfg = None
val_dataloader = None
val_evaluator = None

test_cfg = dict(type="TestLoop")

# 訓練相關優化器設為 None
optim_wrapper = None
auto_scale_lr = dict(enable=True, base_batch_size=16)
param_scheduler = None

default_scope = "mmdet"

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=1008, save_optimizer=False
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

custom_hooks = [dict(type="MultiEMAHook", skip_buffers=False, interval=1)]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)

log_level = "INFO"
resume = False

# 5. 確保這裡寫的是你自定義模組上傳時的正確名稱
custom_imports = dict(imports=["custom_modules"], allow_failed_imports=False)
