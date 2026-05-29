import torch
import torch.nn.functional as F

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.htc import HybridTaskCascade


@DETECTORS.register_module()
class DualScaleHTC(HybridTaskCascade):
    """HTC with dual-scale consistency loss.

    Branch A 使用原始輸入影像；Branch B 使用縮小 scale_factor 倍的影像。
    兩條 branch 共用完全相同的 backbone + neck（weight sharing）。
    Consistency loss（MSE）計算於 FPN P2~P(num_fpn_levels+1) 的每個 level，
    強制模型學習對影像尺度不變的特徵表達。
    推論階段（test）只使用 Branch A，與原版 HTC 完全相同。

    Args:
        scale_factor (float): Branch B 輸入縮小比例，預設 0.5（半解析度）。
        consistency_loss_weight (float): consistency loss 的權重，預設 0.1。
        num_fpn_levels (int): 計算 consistency loss 的 FPN level 數（從 P2 開始），
            預設 4（即 P2, P3, P4, P5）。
    """

    def __init__(self,
                 scale_factor=0.5,
                 consistency_loss_weight=0.1,
                 num_fpn_levels=4,
                 **kwargs):
        super(DualScaleHTC, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.consistency_loss_weight = consistency_loss_weight
        self.num_fpn_levels = num_fpn_levels

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Branch A：原始影像 → backbone+neck → FPN features → RPN + RoI Head（偵測）
        Branch B：縮小影像 → backbone+neck（共用權重）→ FPN features（僅算 consistency loss）
        """
        # ── Branch A：原始尺度特徵（用於偵測） ──────────────────────────────
        x_a = self.extract_feat(img)

        # ── Branch B：縮小尺度特徵（共用 backbone+neck 權重） ─────────────
        img_b = F.interpolate(
            img,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)
        x_b = self.extract_feat(img_b)

        # ── Consistency Loss（FPN P2~P(num_fpn_levels+1) 各 level MSE） ──
        num_levels = min(self.num_fpn_levels, len(x_a), len(x_b))
        loss_consistency = img.new_zeros(1)[0]
        for i in range(num_levels):
            fa = x_a[i]                                             # (B, C, H,   W  )
            fb_up = F.interpolate(                                  # (B, C, H,   W  )  ← 上取樣對齊
                x_b[i], size=fa.shape[-2:],
                mode='bilinear', align_corners=False)
            loss_consistency = loss_consistency + F.mse_loss(fb_up, fa)
        loss_consistency = (loss_consistency / num_levels) * self.consistency_loss_weight

        losses = dict(loss_consistency=loss_consistency)

        # ── RPN（使用 Branch A 特徵） ─────────────────────────────────────
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_a,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # ── RoI Head（使用 Branch A 特徵） ───────────────────────────────
        roi_losses = self.roi_head.forward_train(
            x_a,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs)
        losses.update(roi_losses)

        return losses
