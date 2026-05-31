"""train_cps3.py — 3-branch Cross-Pseudo Supervision (三卡版)

架構：
  GPU 0: ViT  BEiT-v2-Large HTC              (~21 GB)
  GPU 1: CNN1 ResNeXt-101 DetectoRS HTC  (~4-5 GB)
         CNN2 R50 + LNNHopfieldFPN HTC   (~4-6 GB)  ← 與 CNN1 共用，合計 ~9-12 GB

CPS 策略（非對稱）：
  ViT  ← nms_merge(CNN1_pseudo, CNN2_pseudo)  # 兩個 CNN 合體投票，NMS 去重
  CNN1 ← ViT_pseudo                           # 最強模型直接指導
  CNN2 ← ViT_pseudo                           # 最強模型直接指導

  設計邏輯：
    - ViT 品質最高，單一 ViT 偽標籤已足夠指導兩個 CNN
    - 單個 CNN 品質較低，需要兩個 CNN 的投票結果（NMS 去重後）才能
      抗衡 ViT 的高召回率，給 ViT 提供互補的 pseudo label

OOM 緩解 — split backward：
  每個 model 分兩次獨立 backward（supervised 先、CPS 後）再 step，
  不同時持有兩次 forward 的 activation。ViT 峰值 VRAM ~21 GB（非 ~42 GB）。

  若仍 OOM 的退路：
    1. 降低 img_size（1024 → 800）
    2. 改用循環式 pseudo（ViT←CNN2, CNN1←ViT, CNN2←CNN1）

用法：
  CUDA_VISIBLE_DEVICES=0,1 python train_cps3.py \\
    --cfg-vit  all_configs/pretconf/pretexp1_adaplargebeitv2l_htc-v2_lnnfpn.py \\
    --cfg-cnn1 all_configs/nops_config_pret/htc_resnext101_cps.py \\
    --cfg-cnn2 all_configs/nops_config_pret/htc_r50_lnnfpn_cps.py \\
    --ckpt-vit  pretrained/pretexp1_adaplargebeitv2l_htc-v2.pth \\
    --ckpt-cnn1 pretrained/pretwsiall-next101htc.pth \\
    --work-dir  results/cps3/stage1
    # --ckpt-cnn2 省略時使用 config 的 load_from (COCO HTC-R50 URL)

注意：需要 3 × RTX 4090 (各 24 GB)。
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
import mmcv_custom   # noqa: F401  — 觸發 mmcv_custom 中的 register
import mmdet_custom  # noqa: F401  — 觸發 mmdet_custom 中的 register（含 LNNHopfieldFPN）
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
CPS_RAMPUP_EPOCHS = 2     # 前 N epoch 線性 ramp-up
PSEUDO_CONF_THR   = 0.50  # pseudo label confidence 門檻
NMS_IOU_THR       = 0.50  # NMS IoU 閾值（CNN1+CNN2 合體去重時使用）
GRAD_CLIP_NORM    = 35.0  # gradient clip max_norm
LOG_INTERVAL      = 20    # 每隔多少 iter 印 log
CKPT_INTERVAL     = 1     # 每隔多少 epoch 存 checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('Online 3-branch CPS Training (三 GPU)')
    p.add_argument('--cfg-vit',  required=True,  help='ViT config  (cuda:0)')
    p.add_argument('--cfg-cnn1', required=True,  help='CNN1 config (cuda:1)')
    p.add_argument('--cfg-cnn2', required=True,  help='CNN2 config (cuda:1, 與 CNN1 共用)')
    p.add_argument('--ckpt-vit',  default=None,  help='ViT  初始化 checkpoint')
    p.add_argument('--ckpt-cnn1', default=None,  help='CNN1 初始化 checkpoint')
    p.add_argument('--ckpt-cnn2', default=None,  help='CNN2 初始化 checkpoint（省略則用 config load_from）')
    p.add_argument('--resume-vit',  default=None, help='從指定 ckpt 恢復 ViT')
    p.add_argument('--resume-cnn1', default=None, help='從指定 ckpt 恢復 CNN1')
    p.add_argument('--resume-cnn2', default=None, help='從指定 ckpt 恢復 CNN2')
    p.add_argument('--work-dir', default='results/cps3', help='checkpoint 與 log 輸出目錄')
    p.add_argument('--max-epochs', type=int, default=8)
    p.add_argument('--seed', type=int, default=69)
    p.add_argument('--unl-ann-file', default=None,
                   help='unlabeled COCO JSON 路徑（預設使用 ViT config 的 train ann_file）')
    p.add_argument('--val-only', action='store_true',
                   help='只跑一次 validation 後結束')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 工具函式（與 train_cps.py 共用邏輯）
# ─────────────────────────────────────────────────────────────────────────────

def cps_rampup(cur_epoch: int, rampup_epochs: int, max_w: float) -> float:
    if rampup_epochs <= 0:
        return max_w
    return max_w * min(1.0, cur_epoch / rampup_epochs)


def parse_losses(losses: dict) -> torch.Tensor:
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


def inference_to_pseudo_gt(result, img_metas: list, conf_thr: float, device: str):
    """HTC simple_test 輸出 → forward_train 所需的 GT 格式。

    回傳 (gt_bboxes, gt_labels, gt_masks) 三個元素的 tuple。
    所有 tensor 移動到 `device`；BitmapMasks 維持 numpy-based。
    """
    bbox_res, segm_res = result
    bboxes_np  = bbox_res[0]    # [N, 5] or [0, 5]
    masks_list = segm_res[0]    # list of N bool arrays

    meta  = img_metas[0]
    H_img = meta['img_shape'][0]
    W_img = meta['img_shape'][1]

    def _empty():
        return (
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0,),   dtype=torch.long,    device=device),
            BitmapMasks(np.zeros((0, H_img, W_img), dtype=np.uint8), H_img, W_img),
        )

    if len(bboxes_np) == 0:
        return _empty()

    scores = bboxes_np[:, 4]
    keep   = scores >= conf_thr
    bboxes_np  = bboxes_np[keep]
    masks_list = [m for m, k in zip(masks_list, keep.tolist()) if k]

    if len(bboxes_np) == 0:
        return _empty()

    boxes = bboxes_np[:, :4].copy()
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W_img)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H_img)
    gt_bboxes = torch.tensor(boxes, dtype=torch.float32, device=device)
    gt_labels = torch.zeros(len(bboxes_np), dtype=torch.long, device=device)

    resized_masks = []
    for m in masks_list:
        m_bin = m.astype(np.uint8)
        if m_bin.shape[:2] != (H_img, W_img):
            m_bin = mmcv.imresize(m_bin, (W_img, H_img), interpolation='nearest')
        resized_masks.append(m_bin)

    mask_arr = np.stack(resized_masks, axis=0)
    gt_masks = BitmapMasks(mask_arr, H_img, W_img)
    return gt_bboxes, gt_labels, gt_masks


def nms_merge_raw_pseudo(result1, result2, img_metas, conf_thr, device,
                         nms_iou_thr=NMS_IOU_THR):
    """CNN1 + CNN2 的原始推論結果 concat 後執行 NMS，供 ViT 的 CPS pseudo GT 使用。

    Args:
        result1, result2 : model.simple_test()[0]，即單張影像的 (bbox_res, segm_res)
        img_metas        : unlabeled 影像 metas（list of 1 dict）
        conf_thr         : confidence 過濾門檻
        device           : 輸出 tensor 的目標 device（ViT 在 cuda:0）
        nms_iou_thr      : NMS IoU 閾值，去除兩個 CNN 間的重複框

    Returns:
        (gt_bboxes Tensor[N,4], gt_labels Tensor[N], gt_masks BitmapMasks)
    """
    from mmcv.ops import nms as mmcv_nms  # mmcv-full 才有 CUDA NMS，lazy import

    meta  = img_metas[0]
    H_img = meta['img_shape'][0]
    W_img = meta['img_shape'][1]

    def _empty():
        return (
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0,),   dtype=torch.long,    device=device),
            BitmapMasks(np.zeros((0, H_img, W_img), dtype=np.uint8), H_img, W_img),
        )

    def _extract(result, thr):
        """從單張 result 提取 bboxes[N,5] 與 masks_list，並按 conf_thr 過濾。"""
        bbox_res, segm_res = result
        bboxes_np  = bbox_res[0]   # class 0，shape [N, 5]
        masks_list = segm_res[0]   # list of N bool arrays
        if len(bboxes_np) == 0:
            return bboxes_np, []
        keep = bboxes_np[:, 4] >= thr
        return bboxes_np[keep], [m for m, k in zip(masks_list, keep.tolist()) if k]

    bboxes1, masks1 = _extract(result1, conf_thr)
    bboxes2, masks2 = _extract(result2, conf_thr)

    if len(bboxes1) == 0 and len(bboxes2) == 0:
        return _empty()

    # Concatenate（兩者都非空才 concat）
    if len(bboxes1) > 0 and len(bboxes2) > 0:
        all_bboxes = np.concatenate([bboxes1, bboxes2], axis=0)
        all_masks  = list(masks1) + list(masks2)
    elif len(bboxes1) > 0:
        all_bboxes, all_masks = bboxes1, list(masks1)
    else:
        all_bboxes, all_masks = bboxes2, list(masks2)

    # NMS（CPU 執行，boxes 為 img_shape 座標系）
    boxes_t  = torch.tensor(all_bboxes[:, :4], dtype=torch.float32)
    scores_t = torch.tensor(all_bboxes[:, 4],  dtype=torch.float32)
    _, keep  = mmcv_nms(boxes_t, scores_t, nms_iou_thr)
    keep_idx = keep.cpu().numpy().tolist()

    all_bboxes = all_bboxes[keep_idx]
    all_masks  = [all_masks[k] for k in keep_idx]

    if len(all_bboxes) == 0:
        return _empty()

    # 與 inference_to_pseudo_gt 相同的後處理
    boxes = all_bboxes[:, :4].copy()
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W_img)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H_img)
    gt_bboxes = torch.tensor(boxes, dtype=torch.float32, device=device)
    gt_labels = torch.zeros(len(all_bboxes), dtype=torch.long, device=device)

    resized_masks = []
    for m in all_masks:
        m_bin = m.astype(np.uint8)
        if m_bin.shape[:2] != (H_img, W_img):
            m_bin = mmcv.imresize(m_bin, (W_img, H_img), interpolation='nearest')
        resized_masks.append(m_bin)

    mask_arr = np.stack(resized_masks, axis=0)
    gt_masks = BitmapMasks(mask_arr, H_img, W_img)
    return gt_bboxes, gt_labels, gt_masks


def build_unlabeled_loader(cfg_vit: Config, unl_ann_file: str, workers: int = 2):
    img_norm = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
    img_size = getattr(cfg_vit, 'img_size', 1400)

    unl_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img']),
    ]

    unl_cfg = copy.deepcopy(cfg_vit.data.train)
    unl_cfg.pipeline = unl_pipeline

    data_root = unl_cfg.get('data_root', None)
    if data_root is None:
        data_root = cfg_vit.get('data_root', None)

    if data_root is not None:
        unl_cfg.data_root = data_root
        root_norm = os.path.normpath(str(data_root))
        root_name = os.path.basename(root_norm)

        if os.path.isabs(unl_ann_file):
            ann_abs  = os.path.abspath(unl_ann_file)
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

        if 'img_prefix' in unl_cfg and isinstance(unl_cfg.img_prefix, str):
            if os.path.isabs(unl_cfg.img_prefix):
                img_abs  = os.path.abspath(unl_cfg.img_prefix)
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
    logger = logging.getLogger('cps3_train')
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
            'optimizer':  optimizer.state_dict() if optimizer is not None else None,
            'scaler':     scaler.state_dict()    if scaler    is not None else None,
            'meta':       meta,
        },
        path,
    )


def _build_iter_lr_scheduler(optimizer, base_lrs, lr_cfg, max_iters: int):
    if lr_cfg is None:
        def _noop(_iter_idx: int):
            return
        return _noop

    policy       = lr_cfg.get('policy',       'CosineAnnealing')
    warmup       = lr_cfg.get('warmup',       None)
    warmup_iters = int(lr_cfg.get('warmup_iters', 0))
    warmup_ratio = float(lr_cfg.get('warmup_ratio', 0.1))
    min_lr       = float(lr_cfg.get('min_lr',       0.0))

    def _set_lr(scale_fn):
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg['lr'] = scale_fn(base_lr)

    def _step(iter_idx: int):
        if warmup == 'linear' and warmup_iters > 0 and iter_idx < warmup_iters:
            alpha = float(iter_idx + 1) / float(warmup_iters)
            scale = warmup_ratio + (1.0 - warmup_ratio) * alpha
            _set_lr(lambda b: b * scale)
            return

        if policy == 'CosineAnnealing':
            progress = min(max(iter_idx, 0), max_iters - 1) / float(max(max_iters - 1, 1))
            _set_lr(lambda b: min_lr + 0.5 * (b - min_lr) * (1.0 + math.cos(math.pi * progress)))
            return

        _set_lr(lambda b: b)

    return _step


def _pick_metric(eval_dict: dict, metric: str, iou: str):
    for k in (f'{metric}_mAP_{iou}', f'{metric}_mAP'):
        if k in eval_dict:
            return eval_dict[k]
    return None


def _evaluate_one_model(model, val_loader, device: str,
                        logger: logging.Logger, tag: str):
    dataset   = val_loader.dataset
    results   = []
    was_train = model.training
    model.eval()

    device_id = int(device.split(':')[1])

    with torch.no_grad():
        for data in val_loader:
            batch = scatter(data, [device_id])[0]
            out   = model(return_loss=False, **batch)
            if isinstance(out[0], tuple):
                out = [(b, encode_mask_results(m)) for b, m in out]
            results.extend(out)

    eval_50_95 = dataset.evaluate(results, metric=['bbox', 'segm'])
    eval_50    = dataset.evaluate(results, metric=['bbox', 'segm'], iou_thrs=np.array([0.5]))
    eval_60    = dataset.evaluate(results, metric=['bbox', 'segm'], iou_thrs=np.array([0.6]))

    if was_train:
        model.train()

    bbox_50    = _pick_metric(eval_50,    'bbox', '50') or 0.0
    bbox_60    = _pick_metric(eval_60,    'bbox', '60') or 0.0
    bbox_50_95 = eval_50_95.get('bbox_mAP') or 0.0
    segm_50    = _pick_metric(eval_50,    'segm', '50') or 0.0
    segm_60    = _pick_metric(eval_60,    'segm', '60') or 0.0
    segm_50_95 = eval_50_95.get('segm_mAP') or 0.0

    logger.info(
        f'[val][{tag}] vessel bbox mAP50={bbox_50:.4f} mAP60={bbox_60:.4f} '
        f'mAP50:95={bbox_50_95:.4f} | '
        f'segm mAP50={segm_50:.4f} mAP60={segm_60:.4f} '
        f'segm_mAP50:95={segm_50_95:.4f}'
    )
    return {
        'bbox_mAP50': bbox_50, 'bbox_mAP60': bbox_60, 'bbox_mAP50_95': bbox_50_95,
        'segm_mAP50': segm_50, 'segm_mAP60': segm_60, 'segm_mAP50_95': segm_50_95,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 核心 helper：對單一模型做 split backward 更新
# ─────────────────────────────────────────────────────────────────────────────

def _update_model(
    *,
    tag: str,
    model,
    opt,
    scaler,
    use_amp: bool,
    device: str,
    # labeled data (already on CPU, will be moved)
    img_lab,
    metas_lab,
    gt_bboxes,
    gt_labels,
    gt_masks,
    # unlabeled image (already on CPU)
    img_unl,
    metas_unl,
    # merged pseudo GT for this model (already on `device`)
    pg_merged,           # (bboxes, labels, masks) or None when lam==0
    lam: float,
    logger: logging.Logger,
    iter_idx: int,
):
    """Split-backward model update.

    Step 1: supervised forward → backward  (activation 立刻釋放)
    Step 2: CPS forward → backward          (activation 立刻釋放)
    Step 3: clip + optimizer step

    回傳 (loss_sup_val, loss_cps_val) float 純量。
    """
    pb, pl, pm = pg_merged

    opt.zero_grad(set_to_none=True)

    # ── Step 1: Supervised ────────────────────────────────────────────────────
    with torch.cuda.amp.autocast(enabled=use_amp):
        losses_sup = model(
            return_loss=True,
            img=img_lab.to(device),
            img_metas=metas_lab,
            gt_bboxes=[b.to(device) for b in gt_bboxes],
            gt_labels=[lb.to(device) for lb in gt_labels],
            gt_masks=gt_masks,
        )
        loss_sup = parse_losses(losses_sup)

    if torch.isfinite(loss_sup):
        scaler.scale(loss_sup).backward()
    else:
        logger.warning(f'[{tag}] NaN/Inf sup loss at iter {iter_idx}, skip sup backward')
        opt.zero_grad(set_to_none=True)

    loss_sup_val = loss_sup.item() if torch.isfinite(loss_sup) else float('nan')

    # ── Step 2: CPS ───────────────────────────────────────────────────────────
    loss_cps_val = 0.0
    if lam > 0 and len(pb) > 0:
        with torch.cuda.amp.autocast(enabled=use_amp):
            losses_cps = model(
                return_loss=True,
                img=img_unl.to(device),
                img_metas=metas_unl,
                gt_bboxes=[pb],
                gt_labels=[pl],
                gt_masks=[pm],
            )
            loss_cps = parse_losses(losses_cps)

        if torch.isfinite(loss_cps):
            scaler.scale(lam * loss_cps).backward()
            loss_cps_val = loss_cps.item()
        else:
            logger.warning(f'[{tag}] NaN/Inf CPS loss at iter {iter_idx}, skip CPS backward')

    # ── Step 3: Clip + Step ───────────────────────────────────────────────────
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    scaler.step(opt)
    scaler.update()

    return loss_sup_val, loss_cps_val


# ─────────────────────────────────────────────────────────────────────────────
# 主訓練迴圈
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=False)
    os.makedirs(args.work_dir, exist_ok=True)

    logger = setup_logger(os.path.join(args.work_dir, 'train_cps3.log'))
    logger.info(f'work_dir  = {args.work_dir}')
    logger.info(f'cfg_vit   = {args.cfg_vit}')
    logger.info(f'cfg_cnn1  = {args.cfg_cnn1}')
    logger.info(f'cfg_cnn2  = {args.cfg_cnn2}')

    # ── 載入 config ───────────────────────────────────────────────────────────
    cfg_vit  = Config.fromfile(args.cfg_vit)
    cfg_cnn1 = Config.fromfile(args.cfg_cnn1)
    cfg_cnn2 = Config.fromfile(args.cfg_cnn2)

    cfg_vit.model.pretrained  = None
    cfg_cnn1.model.pretrained = None
    cfg_cnn2.model.pretrained = None

    # ── 建立模型 ──────────────────────────────────────────────────────────────
    logger.info('[init] 建立模型 …')
    model_vit  = build_detector(cfg_vit.model,
                                train_cfg=cfg_vit.get('train_cfg'),
                                test_cfg=cfg_vit.get('test_cfg'))
    model_cnn1 = build_detector(cfg_cnn1.model,
                                train_cfg=cfg_cnn1.get('train_cfg'),
                                test_cfg=cfg_cnn1.get('test_cfg'))
    model_cnn2 = build_detector(cfg_cnn2.model,
                                train_cfg=cfg_cnn2.get('train_cfg'),
                                test_cfg=cfg_cnn2.get('test_cfg'))

    # checkpoint：--ckpt-* 優先；resume 時跳過 load_from；否則退回 config load_from
    ckpt_vit  = args.ckpt_vit  or (None if args.resume_vit  else cfg_vit.get('load_from'))
    ckpt_cnn1 = args.ckpt_cnn1 or (None if args.resume_cnn1 else cfg_cnn1.get('load_from'))
    ckpt_cnn2 = args.ckpt_cnn2 or (None if args.resume_cnn2 else cfg_cnn2.get('load_from'))

    for ckpt, model, tag in [
        (ckpt_vit,  model_vit,  'ViT'),
        (ckpt_cnn1, model_cnn1, 'CNN1'),
        (ckpt_cnn2, model_cnn2, 'CNN2'),
    ]:
        if ckpt:
            load_checkpoint(model, ckpt, map_location='cpu',
                            revise_keys=[(r'^module\.', '')])
            logger.info(f'[init] {tag} 載入 {ckpt}')

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp_vit  = cfg_vit.get('fp16')  is not None
    use_amp_cnn1 = cfg_cnn1.get('fp16') is not None
    use_amp_cnn2 = cfg_cnn2.get('fp16') is not None
    scaler_vit   = torch.cuda.amp.GradScaler(enabled=use_amp_vit)
    scaler_cnn1  = torch.cuda.amp.GradScaler(enabled=use_amp_cnn1)
    scaler_cnn2  = torch.cuda.amp.GradScaler(enabled=use_amp_cnn2)

    # ── GPU 分配 ──────────────────────────────────────────────────────────────
    model_vit  = model_vit.to('cuda:0').train()
    model_cnn1 = model_cnn1.to('cuda:1').train()
    model_cnn2 = model_cnn2.to('cuda:1').train()
    logger.info('[init] ViT → cuda:0 | CNN1+CNN2 → cuda:1')

    model_vit.cfg  = cfg_vit
    model_cnn1.cfg = cfg_cnn1
    model_cnn2.cfg = cfg_cnn2

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt_vit  = build_optimizer(model_vit,  cfg_vit.optimizer)
    opt_cnn1 = build_optimizer(model_cnn1, cfg_cnn1.optimizer)
    opt_cnn2 = build_optimizer(model_cnn2, cfg_cnn2.optimizer)

    # ── DataLoader ────────────────────────────────────────────────────────────
    logger.info('[data] 建立 datasets …')

    lab_dataset = build_dataset(cfg_vit.data.train)
    lab_loader  = build_dataloader(
        lab_dataset,
        samples_per_gpu=cfg_vit.data.samples_per_gpu,
        workers_per_gpu=cfg_vit.data.workers_per_gpu,
        dist=False,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = build_dataset(cfg_vit.data.val)
    val_loader  = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg_vit.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        drop_last=False,
    )

    # unlabeled loader
    train_data_root = cfg_vit.data.train.get('data_root', None) or cfg_vit.get('data_root', None)
    if args.unl_ann_file:
        unl_ann = args.unl_ann_file
    elif train_data_root is not None:
        unl_ann = os.path.join(train_data_root, cfg_vit.data.train.ann_file)
    else:
        unl_ann = cfg_vit.data.train.ann_file

    logger.info(f'[data] unlabeled ann_file = {unl_ann}')
    unl_loader = build_unlabeled_loader(cfg_vit, unl_ann, workers=cfg_vit.data.workers_per_gpu)
    logger.info(f'[data] labeled={len(lab_dataset)} | unlabeled={len(unl_loader.dataset)}')

    # ── LR scheduler ──────────────────────────────────────────────────────────
    total_train_iters = args.max_epochs * len(lab_loader)
    step_lr_vit  = _build_iter_lr_scheduler(opt_vit,  [pg['lr'] for pg in opt_vit.param_groups],  cfg_vit.get('lr_config'),  total_train_iters)
    step_lr_cnn1 = _build_iter_lr_scheduler(opt_cnn1, [pg['lr'] for pg in opt_cnn1.param_groups], cfg_cnn1.get('lr_config'), total_train_iters)
    step_lr_cnn2 = _build_iter_lr_scheduler(opt_cnn2, [pg['lr'] for pg in opt_cnn2.param_groups], cfg_cnn2.get('lr_config'), total_train_iters)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 0
    total_iters  = 0

    resume_epochs = []
    for resume_path, model, opt, scaler, tag in [
        (args.resume_vit,  model_vit,  opt_vit,  scaler_vit,  'ViT'),
        (args.resume_cnn1, model_cnn1, opt_cnn1, scaler_cnn1, 'CNN1'),
        (args.resume_cnn2, model_cnn2, opt_cnn2, scaler_cnn2, 'CNN2'),
    ]:
        if resume_path and os.path.exists(resume_path):
            ck = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(ck['state_dict'], strict=False)
            if ck.get('optimizer') is not None:
                opt.load_state_dict(ck['optimizer'])
            if ck.get('scaler') is not None:
                scaler.load_state_dict(ck['scaler'])
            ep = ck.get('meta', {}).get('epoch', 0)
            it = ck.get('meta', {}).get('iter',  0)
            total_iters = max(total_iters, it)
            resume_epochs.append(int(ep))
            logger.info(f'[resume] {tag} 恢復，checkpoint epoch = {ep}')

    if resume_epochs:
        start_epoch = min(resume_epochs)
        logger.info(f'[resume] 共同起始 epoch = {start_epoch}, iter = {total_iters}')

    # ── val-only ──────────────────────────────────────────────────────────────
    if args.val_only:
        logger.info('[val-only] 開始驗證，不執行訓練')
        _evaluate_one_model(model_vit,  val_loader, 'cuda:0', logger, tag='ViT')
        _evaluate_one_model(model_cnn1, val_loader, 'cuda:1', logger, tag='CNN1')
        _evaluate_one_model(model_cnn2, val_loader, 'cuda:1', logger, tag='CNN2')
        logger.info('[val-only] 完成')
        return

    # ── 訓練主迴圈 ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.max_epochs):
        model_vit.train()
        model_cnn1.train()
        model_cnn2.train()

        lam = cps_rampup(epoch, CPS_RAMPUP_EPOCHS, CPS_WEIGHT_MAX)
        logger.info(f'━━ epoch {epoch+1}/{args.max_epochs}  CPS λ = {lam:.3f} ━━')

        unl_iter = iter(unl_loader)

        for i, lab_data in enumerate(lab_loader):

            # ── 取 unlabeled batch ──────────────────────────────────────────
            try:
                unl_data = next(unl_iter)
            except StopIteration:
                unl_iter  = iter(unl_loader)
                unl_data  = next(unl_iter)

            # ── 解包 labeled data ───────────────────────────────────────────
            img_lab   = lab_data['img'].data[0]
            metas_lab = lab_data['img_metas'].data[0]
            gt_bboxes = lab_data['gt_bboxes'].data[0]
            gt_labels = lab_data['gt_labels'].data[0]
            gt_masks  = lab_data['gt_masks'].data[0]

            # ── 解包 unlabeled data ─────────────────────────────────────────
            img_unl   = unl_data['img'].data[0]
            metas_unl = unl_data['img_metas'].data[0]

            # ── LR scheduling ───────────────────────────────────────────────
            step_lr_vit(total_iters)
            step_lr_cnn1(total_iters)
            step_lr_cnn2(total_iters)

            # ==================================================================
            # Phase 1: Inference（所有模型，no_grad，取得 pseudo labels）
            # 在各自的 GPU 上推論，結果 (bbox_res, segm_res) 依然是 numpy-based，
            # 後續 inference_to_pseudo_gt 搬到目標 device。
            # ==================================================================
            raw_pseudo_vit  = None
            raw_pseudo_cnn1 = None
            raw_pseudo_cnn2 = None

            if lam > 0:
                # ViT inference（cuda:0）
                model_vit.eval()
                with torch.no_grad():
                    raw_pseudo_vit = model_vit.simple_test(
                        img_unl.to('cuda:0'), metas_unl, rescale=False)
                model_vit.train()

                # CNN1 inference（cuda:1）
                model_cnn1.eval()
                with torch.no_grad():
                    raw_pseudo_cnn1 = model_cnn1.simple_test(
                        img_unl.to('cuda:1'), metas_unl, rescale=False)
                model_cnn1.train()

                # CNN2 inference（cuda:1，與 CNN1 共用）
                model_cnn2.eval()
                with torch.no_grad():
                    raw_pseudo_cnn2 = model_cnn2.simple_test(
                        img_unl.to('cuda:1'), metas_unl, rescale=False)
                model_cnn2.train()

            # 轉換為 pseudo GT（搬至各 student 的 device）並合併
            def _get_pg(raw, target_device):
                """raw pseudo（來自任意 GPU）→ pseudo GT tensors on target_device。"""
                if raw is None:
                    H = metas_unl[0]['img_shape'][0]
                    W = metas_unl[0]['img_shape'][1]
                    return (
                        torch.zeros((0, 4), dtype=torch.float32, device=target_device),
                        torch.zeros((0,),   dtype=torch.long,    device=target_device),
                        BitmapMasks(np.zeros((0, H, W), dtype=np.uint8), H, W),
                    )
                return inference_to_pseudo_gt(raw[0], metas_unl, PSEUDO_CONF_THR, target_device)

            # ViT ← NMS_merge(CNN1, CNN2)：兩個 CNN 合體投票後去重（target: cuda:0）
            if raw_pseudo_cnn1 is not None and raw_pseudo_cnn2 is not None:
                pg_for_vit = nms_merge_raw_pseudo(
                    raw_pseudo_cnn1[0], raw_pseudo_cnn2[0],
                    metas_unl, PSEUDO_CONF_THR, 'cuda:0')
            else:
                # lam==0 時兩者均為 None，回傳空 pseudo
                pg_for_vit = _get_pg(raw_pseudo_cnn1 or raw_pseudo_cnn2, 'cuda:0')

            # CNN1 ← ViT pseudo 直接使用（最強模型單向指導）
            pg_for_cnn1 = _get_pg(raw_pseudo_vit, 'cuda:1')

            # CNN2 ← ViT pseudo 直接使用（最強模型單向指導）
            pg_for_cnn2 = _get_pg(raw_pseudo_vit, 'cuda:1')

            # ==================================================================
            # Phase 2: 更新各模型（split backward）
            # 注意：三個模型在不同 GPU，可以視為獨立進行。
            # 這裡用串行執行（不同 GPU 的 backward 實際上有一定程度的異步）。
            # ==================================================================

            # ── 更新 ViT（cuda:0）────────────────────────────────────────────
            loss_vit_sup, loss_vit_cps = _update_model(
                tag='ViT',
                model=model_vit, opt=opt_vit, scaler=scaler_vit,
                use_amp=use_amp_vit, device='cuda:0',
                img_lab=img_lab, metas_lab=metas_lab,
                gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks,
                img_unl=img_unl, metas_unl=metas_unl,
                pg_merged=pg_for_vit, lam=lam,
                logger=logger, iter_idx=total_iters + 1,
            )

            # ── 更新 CNN1（cuda:1）───────────────────────────────────────────
            loss_cnn1_sup, loss_cnn1_cps = _update_model(
                tag='CNN1',
                model=model_cnn1, opt=opt_cnn1, scaler=scaler_cnn1,
                use_amp=use_amp_cnn1, device='cuda:1',
                img_lab=img_lab, metas_lab=metas_lab,
                gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks,
                img_unl=img_unl, metas_unl=metas_unl,
                pg_merged=pg_for_cnn1, lam=lam,
                logger=logger, iter_idx=total_iters + 1,
            )

            # ── 更新 CNN2（cuda:1）───────────────────────────────────────────
            loss_cnn2_sup, loss_cnn2_cps = _update_model(
                tag='CNN2',
                model=model_cnn2, opt=opt_cnn2, scaler=scaler_cnn2,
                use_amp=use_amp_cnn2, device='cuda:1',
                img_lab=img_lab, metas_lab=metas_lab,
                gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks,
                img_unl=img_unl, metas_unl=metas_unl,
                pg_merged=pg_for_cnn2, lam=lam,
                logger=logger, iter_idx=total_iters + 1,
            )

            total_iters += 1

            # ── Log ─────────────────────────────────────────────────────────
            if total_iters % LOG_INTERVAL == 0:
                logger.info(
                    f'epoch {epoch+1} | iter {i+1:4d} | '
                    f'vit={loss_vit_sup:.3f}/{loss_vit_cps:.3f} '
                    f'cnn1={loss_cnn1_sup:.3f}/{loss_cnn1_cps:.3f} '
                    f'cnn2={loss_cnn2_sup:.3f}/{loss_cnn2_cps:.3f} '
                    f'(sup/cps) | λ={lam:.3f} | '
                    f'lr_vit={opt_vit.param_groups[0]["lr"]:.5g} '
                    f'lr_cnn1={opt_cnn1.param_groups[0]["lr"]:.5g} '
                    f'lr_cnn2={opt_cnn2.param_groups[0]["lr"]:.5g}'
                )

        # ── Checkpoint ───────────────────────────────────────────────────────
        if (epoch + 1) % CKPT_INTERVAL == 0:
            meta = {'epoch': epoch + 1, 'iter': total_iters}
            for model, opt, scaler, name in [
                (model_vit,  opt_vit,  scaler_vit,  'vit'),
                (model_cnn1, opt_cnn1, scaler_cnn1, 'cnn1'),
                (model_cnn2, opt_cnn2, scaler_cnn2, 'cnn2'),
            ]:
                save_ckpt(model, opt, scaler,
                          os.path.join(args.work_dir, f'{name}_epoch_{epoch+1}.pth'),
                          meta)
            logger.info(f'[ckpt] saved epoch {epoch+1}')

            # ── Validation ───────────────────────────────────────────────────
            _evaluate_one_model(model_vit,  val_loader, 'cuda:0', logger, tag='ViT')
            _evaluate_one_model(model_cnn1, val_loader, 'cuda:1', logger, tag='CNN1')
            _evaluate_one_model(model_cnn2, val_loader, 'cuda:1', logger, tag='CNN2')

    logger.info('[done] 3-branch CPS 訓練完成')


if __name__ == '__main__':
    main()
