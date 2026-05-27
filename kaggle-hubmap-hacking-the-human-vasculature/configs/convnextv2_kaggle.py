# =========================================================================
# 🌟 優化新架構：MultiEMA + RTMDetWithMaskHead (ConvNeXt-V2-Base) 設定檔
# =========================================================================

norm_cfg = dict(type="BN")

# --- 內部核心 Detector 配置 ---
detector_cfg = dict(
    type="RTMDetWithMaskHead",  # 🟢 完美沿用前人開發的「多尺度遮罩輔助監督」偵測器外殼
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        batch_augments=None,
    ),
    # 【遮罩頭通道對齊】配合 Neck 輸出修改為 256
    mask_head=dict(
        type="FCNMaskHead",
        num_convs=7,
        in_channels=256,  # 由 320 修正為 256
        conv_out_channels=256,
        num_classes=1,
    ),
    # 【骨幹網路現代化】
    backbone=dict(
        type="mmpretrain.ConvNeXt",
        arch="base",
        out_indices=(1, 2, 3),
        drop_path_rate=0.4,
        use_grn=True,
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained",
            # 🚨 替換成這個 100% 正確的官方下載點：
            checkpoint="https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-pre_3rdparty_in1k_20230104-00a70fa4.pth",
            prefix="backbone.",
        ),
    ),
    # 【特徵頸部通道對齊】無縫接收並融合 ConvNeXtV2-Base 的特徵矩陣
    neck=dict(
        type="CSPNeXtPAFPN",
        in_channels=[
            256,
            512,
            1024,
        ],  # 🚀 精確對齊 ConvNeXtV2-Base 的輸出特徵通道 (256, 512, 1024)
        out_channels=256,  # 將輸出通道統一收窄為 256，精簡偵測頭計算負擔
        num_csp_blocks=4,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    # 【偵測頭通道對齊】
    bbox_head=dict(
        type="RTMDetSepBNHead",
        num_classes=3,
        in_channels=256,  # 由 320 修正為 256
        stacked_convs=2,
        feat_channels=256,  # 由 320 修正為 256
        anchor_generator=dict(type="MlvlPointGenerator", offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type="DistancePointBBoxCoder"),
        loss_cls=dict(
            type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0
        ),
        # 🎯【針對性提升 AP60 的殺手鐧】將原先的 GIoULoss 升級為 CIoULoss
        # CIoULoss 同時對重疊面積、中心點距離、長寬比施加約束，能逼出極度貼合微血管邊緣的精準邊框
        loss_bbox=dict(type="CIoULoss", loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="SiLU", inplace=True),
    ),
    train_cfg=dict(
        mask_pos_mode="weighted_sum",  # 🟢 完美保留前人的特徵加權求和智慧邏輯！
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
)

# --- 外層 Multi-EMA 包裝 ---
model = dict(
    type="MultiEMADetector",  # 🟢 完美保留前人的 3 通道移動平均權重平滑機制
    momentums=[0.001, 0.0005, 0.00025],
    detector=detector_cfg,  # 注入升級後的新偵測器
)

# =========================================================================
# 🎯 Kaggle 離線提交專用：資料集與測試載入器配置（徹底對齊 m0i.py 智慧）
# =========================================================================
dataset_type = "CocoDataset"
# 1. 將資料根目錄改為工作區，這也是我們剛剛生成假 dval0i.json 的地方
data_root = "/kaggle/working/"
# 2. 🟢 關鍵修正：將測試圖片路徑精確指向 Kaggle 官方的真實離線測試集影像資料夾
img_prefix = "/kaggle/input/hubmap-hacking-the-human-vasculature/test/"
metainfo = dict(classes=("blood_vessel", "glomerulus", "unsure"))
backend_args = None

img_scale = (768, 768)

# 測試集的資料增強流水線（維持你的 768x768 縮放比例）
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# 3. 🌟 徹底解耦！重寫專屬於測試的 test_dataloader，不再借用 val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="data/dval0i.json",  # 指向我們在 /kaggle/working/data/ 下建立的假標籤
        data_prefix=dict(img=img_prefix),  # 完美指向 /kaggle/input/.../test/
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

# 4. 評估器對齊修正
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

# --- 以下為系統排程與迴圈設定，維持不變 ---
train_cfg = dict(type="IterBasedTrainLoop", max_iters=22400, val_interval=1008)
val_cfg = dict(type="MultiEMAValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(type="AdamW", lr=5e-4, weight_decay=0.01),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

auto_scale_lr = dict(enable=True, base_batch_size=16)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=50)
]
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

# 5. 🚨 確保這裡寫的是你 Dataset 上傳時的正確名稱（如果是 hubmap_modules 就要寫對）
custom_imports = dict(
    imports=["custom_modules", "mmpretrain.models"], allow_failed_imports=False
)
