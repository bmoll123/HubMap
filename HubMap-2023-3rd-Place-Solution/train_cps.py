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
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning,
                        message="On January 1, 2023, MMCV will release v2.0.0*")

import mmcv
import mmcv_custom   # noqa: F401 — 注冊自訂 mmcv 修補
import mmdet_custom  # noqa: F401 — 注冊自訂模組 (BEiTAdapter, LNNHopfieldFPN 等)
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import build_optimizer, load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.core import BitmapMasks
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

    unl_cfg            = copy.deepcopy(cfg_vit.data.train)
    unl_cfg.ann_file   = unl_ann_file
    unl_cfg.pipeline   = unl_pipeline
    unl_cfg.pop('data_root', None)  # ann_file 已是完整路徑

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
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s',
                                          datefmt='%m-%d %H:%M:%S'))
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def save_ckpt(model, path: str, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, path)


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

    if args.ckpt_vit:
        load_checkpoint(model_vit, args.ckpt_vit, map_location='cpu',
                        revise_keys=[(r'^module\.', '')])
        logger.info(f'[init] ViT 載入 {args.ckpt_vit}')
    if args.ckpt_cnn:
        load_checkpoint(model_cnn, args.ckpt_cnn, map_location='cpu',
                        revise_keys=[(r'^module\.', '')])
        logger.info(f'[init] CNN 載入 {args.ckpt_cnn}')

    # FP16（與原始 config fp16 設定一致）
    if cfg_vit.get('fp16') is not None:
        wrap_fp16_model(model_vit)
    if cfg_cnn.get('fp16') is not None:
        wrap_fp16_model(model_cnn)

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

    # unlabeled set
    unl_ann = (args.unl_ann_file
               or os.path.join(cfg_vit.data_root,
                               cfg_vit.data.train.ann_file))
    logger.info(f'[data] unlabeled ann_file = {unl_ann}')
    unl_loader = build_unlabeled_loader(
        cfg_vit, unl_ann,
        workers=cfg_vit.data.workers_per_gpu,
    )

    # val set（用 ViT 的 val set 做 epoch 結束時的快速參考）
    val_dataset = build_dataset(cfg_vit.data.val, dict(test_mode=True))
    val_loader  = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg_vit.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    logger.info(f'[data] labeled={len(lab_dataset)} | unlabeled={len(unl_loader.dataset)} | val={len(val_dataset)}')

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume_vit and os.path.exists(args.resume_vit):
        ck = torch.load(args.resume_vit, map_location='cpu')
        model_vit.load_state_dict(ck['state_dict'], strict=False)
        start_epoch = ck.get('meta', {}).get('epoch', 0)
        logger.info(f'[resume] ViT 恢復，起始 epoch = {start_epoch}')
    if args.resume_cnn and os.path.exists(args.resume_cnn):
        ck = torch.load(args.resume_cnn, map_location='cpu')
        model_cnn.load_state_dict(ck['state_dict'], strict=False)
        logger.info(f'[resume] CNN 恢復')

    # ── 訓練主迴圈 ────────────────────────────────────────────────────────────
    total_iters = 0
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
            opt_vit.zero_grad()

            # A1. Supervised loss
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
            loss_vit_total.backward()
            torch.nn.utils.clip_grad_norm_(model_vit.parameters(), GRAD_CLIP_NORM)
            opt_vit.step()

            # ==================================================================
            # Part B：更新 CNN（cuda:1）
            #   supervised loss  +  λ × CPS loss（from ViT pseudo）
            # ==================================================================
            opt_cnn.zero_grad()

            # B1. Supervised loss
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
            loss_cnn_total.backward()
            torch.nn.utils.clip_grad_norm_(model_cnn.parameters(), GRAD_CLIP_NORM)
            opt_cnn.step()

            total_iters += 1

            # ── Log ───────────────────────────────────────────────────────────
            if total_iters % LOG_INTERVAL == 0:
                logger.info(
                    f'epoch {epoch+1} | iter {i+1:4d} | '
                    f'vit_sup={loss_vit_sup.item():.3f} '
                    f'vit_cps={loss_vit_cps.item():.3f} | '
                    f'cnn_sup={loss_cnn_sup.item():.3f} '
                    f'cnn_cps={loss_cnn_cps.item():.3f} | '
                    f'λ={lam:.3f}'
                )

        # ── 存 checkpoint ─────────────────────────────────────────────────────
        if (epoch + 1) % CKPT_INTERVAL == 0:
            meta = {'epoch': epoch + 1, 'iter': total_iters}
            save_ckpt(model_vit,
                      os.path.join(args.work_dir, f'vit_epoch_{epoch+1}.pth'),
                      meta)
            save_ckpt(model_cnn,
                      os.path.join(args.work_dir, f'cnn_epoch_{epoch+1}.pth'),
                      meta)
            logger.info(f'[ckpt] saved epoch {epoch+1}')

    logger.info('[done] CPS 訓練完成')


if __name__ == '__main__':
    main()
