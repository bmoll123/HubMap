"""train_cps.py — Online Cross-Pseudo Supervision (雙卡版)

架構：
  GPU 0: ViT BEiT-v2-Large HTC   (~21 GB)
  GPU 1: ResNeXt-101 DetectoRS HTC (~4 GB)

原理：
  每個 iteration：
    1. 兩個模型各自對 labeled batch 計算 supervised loss
    2. GPU0 ViT 做 no_grad inference → 生成 CNN 的 pseudo GT
    3. GPU1 CNN 做 no_grad inference → 生成 ViT 的 pseudo GT
    4. 兩模型各自加上 CPS loss，backward + step

用法：
  python train_cps.py \\
    --cfg-vit  all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2.py \\
    --cfg-cnn  all_configs/nops_config_pret/htc_resnext101_cps.py \\
    --ckpt-vit results/stage1/best_segm_mAP_epoch_8.pth \\
    --ckpt-cnn hubmap-coco-pretrained-models/detec_htcres101x32_pretcoco.pth \\
    --work-dir results/cps/stage1

注意：需要 2 × RTX 4090 (各 24 GB)。
"""

import argparse
import copy
import logging
import math
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning,
                        message="On January 1, 2023, MMCV will release v2.0.0*")

import mmcv
import mmcv_custom
import mmdet_custom
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import build_optimizer, load_checkpoint
from mmdet.apis import set_random_seed
from mmdet.core import BitmapMasks, encode_mask_results
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

# ── 超參數（可在 CLI 覆蓋） ────────────────────────────────────────────────────
CPS_WEIGHT_MAX    = 1.0   # CPS loss 最大權重（λ_max）
CPS_RAMPUP_EPOCHS = 2     # 前 N epoch 線性 ramp-up（避免初期爛 pseudo label 主導）
PSEUDO_CONF_THR   = 0.50  # pseudo label confidence 門檻
GRAD_CLIP_NORM    = 35.0  # gradient clip max_norm
LOG_INTERVAL      = 20    # 每隔多少 iter 印 log
CKPT_INTERVAL     = 1     # 每隔多少 epoch 存 checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('Online CPS Training (雙 GPU)')
    p.add_argument('--cfg-vit', required=True,
                   help='ViT config 路徑 (cuda:0)')
    p.add_argument('--cfg-cnn', required=True,
                   help='CNN config 路徑 (cuda:1)')
    p.add_argument('--ckpt-vit', default=None,
                   help='ViT 初始化 checkpoint')
    p.add_argument('--ckpt-cnn', default=None,
                   help='CNN 初始化 checkpoint')
    p.add_argument('--work-dir', default='results/cps',
                   help='checkpoint 與 log 輸出目錄')
    p.add_argument('--max-epochs', type=int, default=8)
    p.add_argument('--seed', type=int, default=69)
    p.add_argument('--resume-vit', default=None,
                   help='從指定 checkpoint 恢復 ViT')
    p.add_argument('--resume-cnn', default=None,
                   help='從指定 checkpoint 恢復 CNN')
    p.add_argument('--unl-ann-file', default=None,
                   help='unlabeled COCO JSON 路徑（預設使用 ViT config 的 train ann_file）')
    p.add_argument('--val-only', action='store_true',
                   help='只跑一次 validation 後結束，不做訓練（需搭配 --ckpt-vit / --ckpt-cnn）')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

def cps_rampup(cur_epoch: int, rampup_epochs: int, max_w: float) -> float:
    """線性 ramp-up CPS loss 權重。epoch 0 → 0，epoch >= rampup_epochs → max_w。"""
    if rampup_epochs <= 0:
        return max_w
    return max_w * min(1.0, cur_epoch / rampup_epochs)


def parse_losses(losses: dict) -> torch.Tensor:
    """把 MMDet forward_train 回傳的 loss dict 加總成單一 scalar。
    跳過沒有 'loss' 字串的 key（例如 acc、iou 等 metric）。
    """
    total = None
    for key, val in losses.items():
        if 'loss' not in key:
            continue
        if isinstance(val, (list, tuple)):
            val = sum(v.mean() for v in val if isinstance(v, torch.Tensor))
        elif isinstance(val, torch.Tensor):
            val = val.mean()
        else:
            continue
        total = val if total is None else total + val
    if total is None:
        raise RuntimeError('loss dict 中找不到任何 loss tensor')
    return total


def inference_to_pseudo_gt(
    result,
    img_metas: list,
    conf_thr: float,
    device: str,
):
    """把 HTC simple_test(rescale=False) 的輸出轉成 forward_train 所需的 GT 格式。

    simple_test 回傳（單張影像）：
        result[0] = (bbox_result, segm_result)
        bbox_result[0]: np.ndarray [N, 5]  x1,y1,x2,y2,score  (img_shape 座標)
        segm_result[0]: list of N  np.ndarray bool  (img_h, img_w)  (img_shape 座標)

    使用 rescale=False 讓座標系與 forward_train 的 gt_bboxes 一致
    （都在 resize 後、pad 前的 img_shape 空間）。

    Returns:
        gt_bboxes: Tensor [N, 4]   on `device`
        gt_labels: Tensor [N]      on `device`
        gt_masks:  BitmapMasks     (numpy-based, 不需要搬到 device)
    """
    bbox_res, segm_res = result

    # 只有一個類別 (blood_vessel)，取 index 0
    bboxes_np  = bbox_res[0]    # [N, 5] or empty [0, 5]
    masks_list = segm_res[0]    # list of N bool arrays

    meta    = img_metas[0]
    H_img   = meta['img_shape'][0]
    W_img   = meta['img_shape'][1]

    # ── empty guard ──────────────────────────────────────────────────────────
    def _empty():
        return (
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0,),   dtype=torch.long,    device=device),
            BitmapMasks(np.zeros((0, H_img, W_img), dtype=np.uint8), H_img, W_img),
        )

    if len(bboxes_np) == 0:
        return _empty()

    # ── confidence 過濾 ───────────────────────────────────────────────────────
    scores = bboxes_np[:, 4]
    keep   = scores >= conf_thr
    bboxes_np  = bboxes_np[keep]
    masks_list = [m for m, k in zip(masks_list, keep.tolist()) if k]

    if len(bboxes_np) == 0:
        return _empty()

    # ── gt_bboxes：取 x1y1x2y2，clip 到 img_shape ────────────────────────────
    boxes = bboxes_np[:, :4].copy()
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W_img)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H_img)
    gt_bboxes = torch.tensor(boxes, dtype=torch.float32, device=device)

    # ── gt_labels：全 0（只有 blood_vessel 一類） ─────────────────────────────
    gt_labels = torch.zeros(len(bboxes_np), dtype=torch.long, device=device)

    # ── gt_masks：BitmapMasks (img_shape 大小) ────────────────────────────────
    resized_masks = []
    for m in masks_list:
        m_bin = m.astype(np.uint8)
        if m_bin.shape[:2] != (H_img, W_img):
            # 少數情況：mask 尺寸不符，用最近鄰 resize
            m_bin = mmcv.imresize(m_bin, (W_img, H_img), interpolation='nearest')
        resized_masks.append(m_bin)

    mask_arr = np.stack(resized_masks, axis=0)   # [N, H_img, W_img]
    gt_masks = BitmapMasks(mask_arr, H_img, W_img)

    return gt_bboxes, gt_labels, gt_masks


def build_unlabeled_loader(cfg_vit: Config, unl_ann_file: str, workers: int = 2):
    """建立 unlabeled tiles 的 DataLoader。

    使用簡化版 pipeline（不載入 GT、不做資料增強）：
      LoadImageFromFile → Resize → Normalize → Pad → DefaultFormatBundle → Collect
    輸出格式與 training pipeline 的 img/img_metas 相同，
    可直接傳入 model.simple_test(rescale=False)。
    """
    img_norm = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
    img_size = getattr(cfg_vit, 'img_size', 1400)

    unl_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),   # 不翻轉，保持座標系一致
        dict(type='Normalize', **img_norm),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img']),
    ]

    unl_cfg = copy.deepcopy(cfg_vit.data.train)
    unl_cfg.pipeline = unl_pipeline

    # Keep data_root/img_prefix semantics consistent with CocoDataset.
    data_root = unl_cfg.get('data_root', None)
    if data_root is None:
        data_root = cfg_vit.get('data_root', None)

    if data_root is not None:
        unl_cfg.data_root = data_root

        root_norm = os.path.normpath(str(data_root))
        root_name = os.path.basename(root_norm)

        # Convert ann_file to a path relative to data_root when possible.
        if os.path.isabs(unl_ann_file):
            ann_abs = os.path.abspath(unl_ann_file)
            root_abs = os.path.abspath(str(data_root))
            if os.path.commonpath([root_abs, ann_abs]) == root_abs:
                unl_cfg.ann_file = os.path.relpath(ann_abs, root_abs)
            else:
                unl_cfg.ann_file = ann_abs
        else:
            ann_rel = os.path.normpath(unl_ann_file)
            if ann_rel == root_name or ann_rel.startswith(root_name + os.sep):
                ann_rel = os.path.relpath(ann_rel, root_name)
            unl_cfg.ann_file = ann_rel

        # Normalize img_prefix in the same way to avoid data_root duplication.
        if 'img_prefix' in unl_cfg and isinstance(unl_cfg.img_prefix, str):
            if os.path.isabs(unl_cfg.img_prefix):
                img_abs = os.path.abspath(unl_cfg.img_prefix)
                root_abs = os.path.abspath(str(data_root))
                if os.path.commonpath([root_abs, img_abs]) == root_abs:
                    unl_cfg.img_prefix = os.path.relpath(img_abs, root_abs)
                else:
                    unl_cfg.img_prefix = img_abs
            else:
                img_rel = os.path.normpath(unl_cfg.img_prefix)
                if img_rel == root_name or img_rel.startswith(root_name + os.sep):
                    img_rel = os.path.relpath(img_rel, root_name)
                unl_cfg.img_prefix = img_rel
    else:
        # Fallback to absolute ann_file and absolute img_prefix if no data_root exists.
        unl_cfg.ann_file = os.path.abspath(unl_ann_file)
        if 'img_prefix' in unl_cfg and not os.path.isabs(unl_cfg.img_prefix):
            unl_cfg.img_prefix = os.path.abspath(unl_cfg.img_prefix)

    unl_dataset = build_dataset(unl_cfg)
    unl_loader  = build_dataloader(
        unl_dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers,
        dist=False,
        shuffle=True,
        drop_last=True,
    )
    return unl_loader


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('cps_train')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s',
                                          datefmt='%m-%d %H:%M:%S'))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def save_ckpt(model, optimizer, scaler, path: str, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scaler': scaler.state_dict() if scaler is not None else None,
            'meta': meta,
        },
        path,
    )


def _build_iter_lr_scheduler(optimizer, base_lrs, lr_cfg, max_iters: int):
    """Create a closure that updates optimizer lr at each iter.

    Supports commonly used MMDet style configs in this repo:
      - policy='CosineAnnealing'
      - warmup='linear' with warmup_iters / warmup_ratio
      - min_lr
    """
    if lr_cfg is None:
        def _noop(_iter_idx: int):
            return
        return _noop

    policy = lr_cfg.get('policy', 'CosineAnnealing')
    warmup = lr_cfg.get('warmup', None)
    warmup_iters = int(lr_cfg.get('warmup_iters', 0))
    warmup_ratio = float(lr_cfg.get('warmup_ratio', 0.1))
    min_lr = float(lr_cfg.get('min_lr', 0.0))

    def _set_lr(scale_fn):
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg['lr'] = scale_fn(base_lr)

    def _step(iter_idx: int):
        # iter_idx: 0-based global iter
        if warmup == 'linear' and warmup_iters > 0 and iter_idx < warmup_iters:
            alpha = float(iter_idx + 1) / float(warmup_iters)
            scale = warmup_ratio + (1.0 - warmup_ratio) * alpha
            _set_lr(lambda b: b * scale)
            return

        if policy == 'CosineAnnealing':
            if max_iters <= 1:
                progress = 1.0
            else:
                progress = min(max(iter_idx, 0), max_iters - 1) / float(max_iters - 1)

            _set_lr(lambda b: min_lr + 0.5 * (b - min_lr) * (1.0 + math.cos(math.pi * progress)))
            return

        # Fallback: keep constant lr if policy is unsupported.
        _set_lr(lambda b: b)

    return _step


def _pick_metric(eval_dict: dict, metric: str, iou: str):
    """Fetch a metric value from MMDet evaluate dict with robust key fallback."""
    keys = [
        f'{metric}_mAP_{iou}',
        f'{metric}_mAP',
    ]
    for k in keys:
        if k in eval_dict:
            return eval_dict[k]
    return None


def _evaluate_one_model(model, val_loader, device: str, logger: logging.Logger, tag: str):
    """Run single-model validation and return bbox/segm AP summaries."""
    dataset = val_loader.dataset
    results = []

    was_training = model.training
    model.eval()

    device_id = int(device.split(':')[1])

    with torch.no_grad():
        for data in val_loader:
            # scatter moves all DataContainers to the target device properly
            batch = scatter(data, [device_id])[0]
            out = model(return_loss=False, **batch)
            # binary mask → RLE（與 single_gpu_test 相同處理）
            if isinstance(out[0], tuple):
                out = [(bbox_r, encode_mask_results(mask_r))
                       for bbox_r, mask_r in out]
            results.extend(out)

    # mAP50:95 (COCO default)
    eval_50_95 = dataset.evaluate(results, metric=['bbox', 'segm'])
    # mAP50 and mAP60
    eval_50 = dataset.evaluate(results, metric=['bbox', 'segm'], iou_thrs=np.array([0.5]))
    eval_60 = dataset.evaluate(results, metric=['bbox', 'segm'], iou_thrs=np.array([0.6]))

    if was_training:
        model.train()

    bbox_50 = _pick_metric(eval_50, 'bbox', '50')
    bbox_60 = _pick_metric(eval_60, 'bbox', '60')
    bbox_50_95 = eval_50_95.get('bbox_mAP')

    segm_50 = _pick_metric(eval_50, 'segm', '50')
    segm_60 = _pick_metric(eval_60, 'segm', '60')
    segm_50_95 = eval_50_95.get('segm_mAP')

    logger.info(
        f'[val][{tag}] vessel bbox mAP50={bbox_50:.4f} mAP60={bbox_60:.4f} mAP50:95={bbox_50_95:.4f} | '
        f'segm mAP50={segm_50:.4f} mAP60={segm_60:.4f} mAP50:95={segm_50_95:.4f}'
    )

    return {
        'bbox_mAP50': bbox_50,
        'bbox_mAP60': bbox_60,
        'bbox_mAP50_95': bbox_50_95,
        'segm_mAP50': segm_50,
        'segm_mAP60': segm_60,
        'segm_mAP50_95': segm_50_95,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主訓練迴圈
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=False)
    os.makedirs(args.work_dir, exist_ok=True)

    logger = setup_logger(os.path.join(args.work_dir, 'train_cps.log'))
    logger.info(f'work_dir = {args.work_dir}')
    logger.info(f'cfg_vit  = {args.cfg_vit}')
    logger.info(f'cfg_cnn  = {args.cfg_cnn}')

    # ── 載入 config ──────────────────────────────────────────────────────────
    cfg_vit = Config.fromfile(args.cfg_vit)
    cfg_cnn = Config.fromfile(args.cfg_cnn)

    # pretrained 路徑已由 ckpt-vit / ckpt-cnn 參數處理，清除 config 中的 pretrained
    cfg_vit.model.pretrained = None
    cfg_cnn.model.pretrained = None

    # ── 建立模型 ──────────────────────────────────────────────────────────────
    logger.info('[init] 建立模型 …')
    model_vit = build_detector(cfg_vit.model,
                               train_cfg=cfg_vit.get('train_cfg'),
                               test_cfg=cfg_vit.get('test_cfg'))
    model_cnn = build_detector(cfg_cnn.model,
                               train_cfg=cfg_cnn.get('train_cfg'),
                               test_cfg=cfg_cnn.get('test_cfg'))

    # --ckpt-* 優先；resume 時跳過 load_from（weights 會由 resume 載入）；
    # 全新訓練且未指定 --ckpt-* 時，才退回 config 的 load_from
    ckpt_vit = args.ckpt_vit or (None if args.resume_vit else cfg_vit.get('load_from'))
    ckpt_cnn = args.ckpt_cnn or (None if args.resume_cnn else cfg_cnn.get('load_from'))
    if ckpt_vit:
        load_checkpoint(model_vit, ckpt_vit, map_location='cpu',
                        revise_keys=[(r'^module\.', '')])
        logger.info(f'[init] ViT 載入 {ckpt_vit}')
    if ckpt_cnn:
        load_checkpoint(model_cnn, ckpt_cnn, map_location='cpu',
                        revise_keys=[(r'^module\.', '')])
        logger.info(f'[init] CNN 載入 {ckpt_cnn}')

    # AMP（與原始 config 的 fp16 開關對齊）
    use_amp_vit = cfg_vit.get('fp16') is not None
    use_amp_cnn = cfg_cnn.get('fp16') is not None
    scaler_vit = torch.cuda.amp.GradScaler(enabled=use_amp_vit)
    scaler_cnn = torch.cuda.amp.GradScaler(enabled=use_amp_cnn)

    model_vit = model_vit.to('cuda:0').train()
    model_cnn = model_cnn.to('cuda:1').train()
    logger.info('[init] ViT → cuda:0 | CNN → cuda:1')

    # 讓 model.simple_test 知道自己的 test_cfg（不用 MMDataParallel 時需手動設）
    model_vit.cfg = cfg_vit
    model_cnn.cfg = cfg_cnn

    # ── 建立 Optimizer ────────────────────────────────────────────────────────
    opt_vit = build_optimizer(model_vit, cfg_vit.optimizer)
    opt_cnn = build_optimizer(model_cnn, cfg_cnn.optimizer)

    # ── 建立 DataLoader ───────────────────────────────────────────────────────
    logger.info('[data] 建立 datasets …')

    # labeled train set（取自 ViT config 的 data.train）
    lab_dataset = build_dataset(cfg_vit.data.train)
    lab_loader  = build_dataloader(
        lab_dataset,
        samples_per_gpu=cfg_vit.data.samples_per_gpu,
        workers_per_gpu=cfg_vit.data.workers_per_gpu,
        dist=False,
        shuffle=True,
        drop_last=True,
    )

    # val set（每個 epoch 後評估 ViT / CNN）
    val_dataset = build_dataset(cfg_vit.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg_vit.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        drop_last=False,
    )

    # ── 建立 LR scheduler（iter-based）────────────────────────────────────────
    total_train_iters = args.max_epochs * len(lab_loader)
    base_lrs_vit = [pg['lr'] for pg in opt_vit.param_groups]
    base_lrs_cnn = [pg['lr'] for pg in opt_cnn.param_groups]
    step_lr_vit = _build_iter_lr_scheduler(opt_vit, base_lrs_vit, cfg_vit.get('lr_config', None), total_train_iters)
    step_lr_cnn = _build_iter_lr_scheduler(opt_cnn, base_lrs_cnn, cfg_cnn.get('lr_config', None), total_train_iters)

    # unlabeled set
    train_data_root = cfg_vit.data.train.get('data_root', None)
    if train_data_root is None:
        train_data_root = cfg_vit.get('data_root', None)

    if args.unl_ann_file:
        unl_ann = args.unl_ann_file
    else:
        if train_data_root is not None:
            unl_ann = os.path.join(train_data_root, cfg_vit.data.train.ann_file)
        else:
            unl_ann = cfg_vit.data.train.ann_file
    logger.info(f'[data] unlabeled ann_file = {unl_ann}')
    unl_loader = build_unlabeled_loader(
        cfg_vit, unl_ann,
        workers=cfg_vit.data.workers_per_gpu,
    )

    logger.info(f'[data] labeled={len(lab_dataset)} | unlabeled={len(unl_loader.dataset)}')

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    total_iters = 0
    resume_epoch_vit = None
    resume_epoch_cnn = None

    if args.resume_vit and os.path.exists(args.resume_vit):
        ck = torch.load(args.resume_vit, map_location='cpu')
        model_vit.load_state_dict(ck['state_dict'], strict=False)
        if ck.get('optimizer') is not None:
            opt_vit.load_state_dict(ck['optimizer'])
        if ck.get('scaler') is not None:
            scaler_vit.load_state_dict(ck['scaler'])
        resume_epoch_vit = ck.get('meta', {}).get('epoch', 0)
        total_iters = max(total_iters, ck.get('meta', {}).get('iter', 0))
        logger.info(f'[resume] ViT 恢復，checkpoint epoch = {resume_epoch_vit}')

    if args.resume_cnn and os.path.exists(args.resume_cnn):
        ck = torch.load(args.resume_cnn, map_location='cpu')
        model_cnn.load_state_dict(ck['state_dict'], strict=False)
        if ck.get('optimizer') is not None:
            opt_cnn.load_state_dict(ck['optimizer'])
        if ck.get('scaler') is not None:
            scaler_cnn.load_state_dict(ck['scaler'])
        resume_epoch_cnn = ck.get('meta', {}).get('epoch', 0)
        total_iters = max(total_iters, ck.get('meta', {}).get('iter', 0))
        logger.info(f'[resume] CNN 恢復，checkpoint epoch = {resume_epoch_cnn}')

    if resume_epoch_vit is not None or resume_epoch_cnn is not None:
        if resume_epoch_vit is None:
            start_epoch = int(resume_epoch_cnn)
        elif resume_epoch_cnn is None:
            start_epoch = int(resume_epoch_vit)
        else:
            if int(resume_epoch_vit) != int(resume_epoch_cnn):
                logger.warning(
                    '[resume] ViT/CNN epoch 不一致：vit=%s, cnn=%s，使用較小值以避免越界',
                    resume_epoch_vit,
                    resume_epoch_cnn,
                )
            start_epoch = min(int(resume_epoch_vit), int(resume_epoch_cnn))

        logger.info(f'[resume] 共同起始 epoch = {start_epoch}, iter = {total_iters}')

    # ── val-only 模式：跑完驗證就結束 ──────────────────────────────────────────
    if args.val_only:
        logger.info('[val-only] 開始驗證，不執行訓練')
        _evaluate_one_model(model_vit, val_loader, 'cuda:0', logger, tag='ViT')
        _evaluate_one_model(model_cnn, val_loader, 'cuda:1', logger, tag='CNN')
        logger.info('[val-only] 完成')
        return

    # ── 訓練主迴圈 ────────────────────────────────────────────────────────────
    best_metric = 0.0   # 留給未來整合 evaluation hook

    for epoch in range(start_epoch, args.max_epochs):
        model_vit.train()
        model_cnn.train()

        lam = cps_rampup(epoch, CPS_RAMPUP_EPOCHS, CPS_WEIGHT_MAX)
        logger.info(f'━━ epoch {epoch + 1}/{args.max_epochs}  CPS λ = {lam:.3f} ━━')

        unl_iter = iter(unl_loader)

        for i, lab_data in enumerate(lab_loader):

            # ── 取一批 unlabeled 資料 ─────────────────────────────────────────
            try:
                unl_data = next(unl_iter)
            except StopIteration:
                unl_iter = iter(unl_loader)
                unl_data = next(unl_iter)

            # ── 解包 labeled 資料（DataContainer → tensor / list） ────────────
            img_lab   = lab_data['img'].data[0]           # [B, C, H, W]
            metas_lab = lab_data['img_metas'].data[0]     # list of B dicts
            gt_bboxes = lab_data['gt_bboxes'].data[0]     # list of B tensors
            gt_labels = lab_data['gt_labels'].data[0]     # list of B tensors
            gt_masks  = lab_data['gt_masks'].data[0]      # list of B BitmapMasks

            # ── 解包 unlabeled 資料 ───────────────────────────────────────────
            img_unl   = unl_data['img'].data[0]           # [1, C, H, W]
            metas_unl = unl_data['img_metas'].data[0]     # list of 1 dict

            # ==================================================================
            # Part A：更新 ViT（cuda:0）
            #   supervised loss  +  λ × CPS loss（from CNN pseudo）
            # ==================================================================
            # iter-based lr scheduling
            step_lr_vit(total_iters)
            step_lr_cnn(total_iters)

            opt_vit.zero_grad(set_to_none=True)

            # A1. Supervised loss
            with torch.cuda.amp.autocast(enabled=use_amp_vit):
                losses_vit_sup = model_vit(
                    return_loss=True,
                    img=img_lab.to('cuda:0'),
                    img_metas=metas_lab,
                    gt_bboxes=[b.to('cuda:0') for b in gt_bboxes],
                    gt_labels=[l.to('cuda:0') for l in gt_labels],
                    gt_masks=gt_masks,
                )
                loss_vit_sup = parse_losses(losses_vit_sup)

            # A2. CNN pseudo labels（no_grad on cuda:1）
            if lam > 0:
                model_cnn.eval()
                with torch.no_grad():
                    result_cnn = model_cnn.simple_test(
                        img_unl.to('cuda:1'),
                        metas_unl,
                        rescale=False,   # 保持 img_shape 座標系
                    )
                model_cnn.train()

                pb, pl, pm = inference_to_pseudo_gt(
                    result_cnn[0], metas_unl, PSEUDO_CONF_THR, 'cuda:0')

                if len(pb) > 0:
                    with torch.cuda.amp.autocast(enabled=use_amp_vit):
                        losses_vit_cps = model_vit(
                            return_loss=True,
                            img=img_unl.to('cuda:0'),
                            img_metas=metas_unl,
                            gt_bboxes=[pb],
                            gt_labels=[pl],
                            gt_masks=[pm],
                        )
                        loss_vit_cps = parse_losses(losses_vit_cps)
                else:
                    loss_vit_cps = torch.tensor(0.0, device='cuda:0')
            else:
                loss_vit_cps = torch.tensor(0.0, device='cuda:0')

            loss_vit_total = loss_vit_sup + lam * loss_vit_cps
            scaler_vit.scale(loss_vit_total).backward()
            scaler_vit.unscale_(opt_vit)
            torch.nn.utils.clip_grad_norm_(model_vit.parameters(), GRAD_CLIP_NORM)
            scaler_vit.step(opt_vit)
            scaler_vit.update()

            # ==================================================================
            # Part B：更新 CNN（cuda:1）
            #   supervised loss  +  λ × CPS loss（from ViT pseudo）
            # ==================================================================
            opt_cnn.zero_grad(set_to_none=True)

            # B1. Supervised loss
            with torch.cuda.amp.autocast(enabled=use_amp_cnn):
                losses_cnn_sup = model_cnn(
                    return_loss=True,
                    img=img_lab.to('cuda:1'),
                    img_metas=metas_lab,
                    gt_bboxes=[b.to('cuda:1') for b in gt_bboxes],
                    gt_labels=[l.to('cuda:1') for l in gt_labels],
                    gt_masks=gt_masks,
                )
                loss_cnn_sup = parse_losses(losses_cnn_sup)

            # B2. ViT pseudo labels（no_grad on cuda:0）
            if lam > 0:
                model_vit.eval()
                with torch.no_grad():
                    result_vit = model_vit.simple_test(
                        img_unl.to('cuda:0'),
                        metas_unl,
                        rescale=False,
                    )
                model_vit.train()

                pb, pl, pm = inference_to_pseudo_gt(
                    result_vit[0], metas_unl, PSEUDO_CONF_THR, 'cuda:1')

                if len(pb) > 0:
                    with torch.cuda.amp.autocast(enabled=use_amp_cnn):
                        losses_cnn_cps = model_cnn(
                            return_loss=True,
                            img=img_unl.to('cuda:1'),
                            img_metas=metas_unl,
                            gt_bboxes=[pb],
                            gt_labels=[pl],
                            gt_masks=[pm],
                        )
                        loss_cnn_cps = parse_losses(losses_cnn_cps)
                else:
                    loss_cnn_cps = torch.tensor(0.0, device='cuda:1')
            else:
                loss_cnn_cps = torch.tensor(0.0, device='cuda:1')

            loss_cnn_total = loss_cnn_sup + lam * loss_cnn_cps
            scaler_cnn.scale(loss_cnn_total).backward()
            scaler_cnn.unscale_(opt_cnn)
            torch.nn.utils.clip_grad_norm_(model_cnn.parameters(), GRAD_CLIP_NORM)
            scaler_cnn.step(opt_cnn)
            scaler_cnn.update()

            total_iters += 1

            # ── Log ───────────────────────────────────────────────────────────
            if total_iters % LOG_INTERVAL == 0:
                logger.info(
                    f'epoch {epoch+1} | iter {i+1:4d} | '
                    f'vit_sup={loss_vit_sup.item():.3f} '
                    f'vit_cps={loss_vit_cps.item():.3f} | '
                    f'cnn_sup={loss_cnn_sup.item():.3f} '
                    f'cnn_cps={loss_cnn_cps.item():.3f} | '
                    f'λ={lam:.3f} | '
                    f'lr_vit={opt_vit.param_groups[0]["lr"]:.6g} '
                    f'lr_cnn={opt_cnn.param_groups[0]["lr"]:.6g}'
                )

        # ── 存 checkpoint ─────────────────────────────────────────────────────
        if (epoch + 1) % CKPT_INTERVAL == 0:
            meta = {'epoch': epoch + 1, 'iter': total_iters}
            save_ckpt(model_vit,
                      opt_vit,
                      scaler_vit,
                      os.path.join(args.work_dir, f'vit_epoch_{epoch+1}.pth'),
                      meta)
            save_ckpt(model_cnn,
                      opt_cnn,
                      scaler_cnn,
                      os.path.join(args.work_dir, f'cnn_epoch_{epoch+1}.pth'),
                      meta)
            logger.info(f'[ckpt] saved epoch {epoch+1}')

            # ── 每個 epoch 做 validation（vessel bbox/segm mAP50, mAP60, mAP50:95） ──
            _evaluate_one_model(model_vit, val_loader, 'cuda:0', logger, tag='ViT')
            _evaluate_one_model(model_cnn, val_loader, 'cuda:1', logger, tag='CNN')

    logger.info('[done] CPS 訓練完成')


if __name__ == '__main__':
    main()
