# Copyright (c) 2026. All rights reserved.
"""
LNN-Hopfield FPN
================
將論文 "Out-of-Distribution Nuclei Segmentation in Histology Imaging via
Liquid Neural Networks with Modern Hopfield Layer" 的核心方法整合進
Feature Pyramid Network，用以應對病理影像不同染色導致的 OOD 問題。

架構流程
--------
  Backbone → [標準 FPN top-down 路徑]
           → CfC Feature Encoder  (Coarse-to-Fine 時間序列融合)
           → Modern Hopfield Layer (特徵聯想修復)
           → FPN 輸出

模組說明
--------
CfCCell
  - 模擬 Closed-form Continuous-time (CfC) RNN 的決策機制。
  - 隱藏狀態 h 從最粗糙的 FPN 層（P5/P6）傳遞到最精細層（P2）。
  - 包含兩個特徵處理器：
      g(·): 全局觀（Global view） — 在 OOD 情況下提供穩定的背景資訊
      h(·): 局部觀（Local view）  — 在影像品質佳時描繪精細細節
  - 動態決策網路 f(·) 根據當前切片品質自動決定 g/h 的混合比例。

ModernHopfieldLayer
  - 儲存可學習的「健康細胞核原型（Prototypes）」。
  - 將 LNN 融合的特徵作為 Query，計算與原型的相似度（Softmax Attention）。
  - 將檢索出的原型加權後作為通道注意力，自動修復被異常染色污染的特徵。

LNNHopfieldFPN
  - 繼承標準 FPN，在其輸出後依序套用 CfCCell 和 ModernHopfieldLayer。
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from torch import Tensor

from ..builder import NECKS
from .fpn import FPN


# ---------------------------------------------------------------------------
# 1. CfC Cell — 動態特徵融合決策大腦
# ---------------------------------------------------------------------------

class CfCCell(BaseModule):
    """Closed-form Continuous-time Cell for spatial feature maps.

    將多尺度 FPN 特徵視為「由粗到細（Coarse-to-Fine）」的時間序列，
    透過 CfC 動態決策機制在每個 step 自動決定要信任精細局部特徵或全局背景。

    核心方程式（簡化版）：
        combined = [x_gap ; h_prev]   # 當前特徵摘要 + 前一隱藏狀態
        g  = G(combined)              # Global processor（看全局）
        h_ = H(combined)              # Local  processor（看局部細節）
        f  = F(combined)  ∈ 無界實數  # Dynamic gate（決策推桿）

        h_new = σ(-f·t) ⊙ g + (1 - σ(-f·t)) ⊙ h_  # CfC 輸出：動態混合
        out   = x ⊙ σ(h_new) + Conv(x)              # 作用回空間特徵圖

    當影像 OOD（局部特徵雜訊高），f → 0，模型退回依賴 g（全局穩定視角）。
    當影像清晰，f → 1，模型讓 h 主導，描繪精細細胞邊界。

    Args:
        channels (int): FPN 特徵通道數（= FPN out_channels，預設 256）。
        hidden_ratio (float): 中間層擴展比例，預設 1.0（不擴展）。
    """

    def __init__(self, channels, hidden_ratio=1.0, init_cfg=None):
        super(CfCCell, self).__init__(init_cfg=init_cfg)
        self.channels = channels
        hidden = int(channels * hidden_ratio)

        # Global processor g: 捕捉整體背景脈絡（粗粒度、穩定）
        self.g_net = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, channels),
        )

        # Local processor h: 捕捉精細局部細節（細粒度、對 OOD 敏感）
        self.h_net = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, channels),
        )

        # Dynamic gate f: 輸出無界實數（unconstrained）
        # gate = σ(-f(x,I;θ_f) · t)，sigmoid 在 forward 中與 t 一起套用
        self.f_net = nn.Sequential(
            nn.Linear(channels * 2, channels),
        )

        # 殘差空間特徵調制：將混合後的通道向量投射回空間特徵圖
        self.spatial_mod = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, h_prev, t=1.0):
        """前向傳播。

        Args:
            x (Tensor): 當前 FPN level 的特徵圖，shape (B, C, H, W)。
            h_prev (Tensor): 前一步隱藏狀態（來自更粗糙 level），shape (B, C)。
            t (float): 時間步（level index，由粗到細遞增）。

        Returns:
            Tuple[Tensor, Tensor]:
                - h_new: 更新後的隱藏狀態 (B, C)
                - out:   調制後的特徵圖 (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Step 1: 取得當前 level 的全局通道摘要
        x_gap = F.adaptive_avg_pool2d(x, 1).reshape(B, C)  # (B, C)

        # Step 2: 拼接當前摘要與前一隱藏狀態
        combined = torch.cat([x_gap, h_prev], dim=1)  # (B, 2C)

        # Step 3: CfC 三路計算
        g = self.g_net(combined)        # (B, C)
        h_local = self.h_net(combined)  # (B, C)
        f_val = self.f_net(combined)    # (B, C) 無界實數

        # Step 4: 套用原始 CfC 公式
        # gate = σ(-f·t)；t 大（精細 level）→ gate 小 → h 主導
        gate = torch.sigmoid(-f_val * t)              # (B, C)
        h_new = gate * g + (1.0 - gate) * h_local    # (B, C)

        # Step 5: 將 h_new 作為通道注意力調制空間特徵圖
        scale = torch.sigmoid(h_new).view(B, C, 1, 1)  # (B, C, 1, 1)
        out = x * scale + self.spatial_mod(x)           # residual

        return h_new, out


# ---------------------------------------------------------------------------
# 2. Modern Hopfield Layer — 特徵聯想修復記憶圖鑑
# ---------------------------------------------------------------------------

class ModernHopfieldLayer(BaseModule):
    """Modern Hopfield Layer for feature association and auto-repair.

    在 Hopfield 記憶層中儲存大量「標準、健康細胞核特徵原型（Prototypes）」。
    當輸入特徵受到異常染色污染時，Hopfield Attention 會將其拉向最近的健康原型，
    達到自動「調音（Auto-tune）」效果。

    運作機制（類 Attention）：
        q      = Q_proj(x_gap)         # Query：輸入特徵摘要
        energy = β · q @ P^T           # Hopfield energy（pure dot product）
        attn   = softmax(energy)        # (B, num_proto)
        recall = attn @ P               # 從記憶中檢索 (B, C)
        mod    = σ(OutProj(recall + x_gap))  # 通道注意力
        out    = x ⊙ mod               # 修復後的特徵圖

    Args:
        channels (int): 特徵通道數。
        num_prototypes (int): 記憶庫中原型的數量，預設 128。
        beta (float): Hopfield 溫度參數（越大，Attention 越尖銳），預設 1.0。
    """

    def __init__(self, channels, num_prototypes=128, beta=1.0, init_cfg=None):
        super(ModernHopfieldLayer, self).__init__(init_cfg=init_cfg)
        self.channels = channels
        self.beta = beta

        # 可學習原型記憶庫 P ∈ ℝ^{num_proto × C}
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, channels))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0)).squeeze(0)

        # Query 投影
        self.query_proj = nn.Linear(channels, channels)

        # Output 投影
        self.out_proj = nn.Linear(channels, channels)

        # LayerNorm 穩定訓練
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """前向傳播。

        Args:
            x (Tensor): 輸入特徵圖，shape (B, C, H, W)。

        Returns:
            Tensor: 修復後的特徵圖，shape (B, C, H, W)。
        """
        B, C, H, W = x.shape

        # Step 1: 全局摘要（節省計算，避免逐像素 attention）
        x_gap = F.adaptive_avg_pool2d(x, 1).reshape(B, C)  # (B, C)

        # Step 2: 計算 Query
        q = self.query_proj(x_gap)  # (B, C)

        # Step 3: Hopfield Attention
        # energy = β⟨q, pⱼ⟩（純 dot product，僅 β 縮放）
        energy = self.beta * (q @ self.prototypes.T)  # (B, num_proto)
        attn = F.softmax(energy, dim=-1)              # (B, num_proto)

        # Step 4: 從記憶庫中檢索
        recalled = attn @ self.prototypes  # (B, C)

        # Step 5: 產生通道注意力調制信號
        mod_signal = self.norm(self.out_proj(recalled + x_gap))  # (B, C)
        mod = torch.sigmoid(mod_signal).view(B, C, 1, 1)         # (B, C, 1, 1)

        # Step 6: 調制原始特徵圖
        out = x * mod

        return out


# ---------------------------------------------------------------------------
# 3. LNNHopfieldFPN — 完整整合 Neck 模組
# ---------------------------------------------------------------------------

@NECKS.register_module()
class LNNHopfieldFPN(FPN):
    """LNN-Hopfield FPN: OOD-Robust Feature Pyramid Network.

    在標準 FPN 之後，依序套用：
      1. CfC Feature Encoder: 將 FPN 多尺度特徵以 Coarse-to-Fine 時間順序
         餵給 CfCCell，利用動態 gate 自適應融合全局/局部視角，抵抗 OOD 衝擊。
      2. Modern Hopfield Layer: 以可學習原型記憶庫修復被染色異常污染的特徵。

    使用方式（config）::

        neck=[
            dict(type='ExtraAttention', ...),
            dict(
                type='LNNHopfieldFPN',
                in_channels=[1024, 1024, 1024, 1024],
                out_channels=256,
                num_outs=5,
                norm_cfg=dict(type='GN', num_groups=32),
                num_prototypes=128,
                hopfield_beta=1.0,
                cfc_hidden_ratio=1.0,
            ),
        ]

    Args:
        num_prototypes (int): Hopfield 記憶庫原型數量，預設 128。
        hopfield_beta (float): Hopfield 溫度參數，預設 1.0。
        cfc_hidden_ratio (float): CfC Cell 中間層擴展比例，預設 1.0。
        其他參數繼承自 FPN。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 # 新增參數
                 num_prototypes=128,
                 hopfield_beta=1.0,
                 cfc_hidden_ratio=1.0):
        super(LNNHopfieldFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg,
        )

        # CfC Cell（共享權重，適用於所有 FPN levels）
        self.cfc_cell = CfCCell(out_channels, hidden_ratio=cfc_hidden_ratio)

        # Modern Hopfield Layer（每個 level 獨立，因尺度語意不同）
        self.hopfield_layers = nn.ModuleList([
            ModernHopfieldLayer(
                channels=out_channels,
                num_prototypes=num_prototypes,
                beta=hopfield_beta,
            )
            for _ in range(num_outs)
        ])

    @auto_fp16()
    def forward(self, inputs):
        """前向傳播。

        執行順序：
          1. 標準 FPN top-down 特徵金字塔（繼承自父類）
          2. CfC Encoder：Coarse-to-Fine 時間序列融合（P6→P2）
          3. Hopfield Layer：per-level 特徵修復

        Args:
            inputs (tuple[Tensor]): Backbone 各 stage 的特徵圖。

        Returns:
            tuple: OOD 強化後的多尺度特徵金字塔。
        """
        # Phase 1: 標準 FPN
        fpn_outs = list(super(LNNHopfieldFPN, self).forward(inputs))
        num_levels = len(fpn_outs)
        B = fpn_outs[0].shape[0]
        device = fpn_outs[0].device

        # Phase 2: CfC Feature Encoder（Coarse-to-Fine）
        # 從最粗糙 level（高 index，大 stride）往最精細 level（低 index）傳遞隱藏狀態
        # 例：[P2, P3, P4, P5, P6] → P6(t=1) → P5(t=2) → P4(t=3) → P3(t=4) → P2(t=5)
        h = torch.zeros(B, self.out_channels, device=device)
        cfc_outs = [None] * num_levels

        for step, lvl in enumerate(reversed(range(num_levels))):
            t = float(step + 1)  # t 從 1 開始，避免 t=0 使 gate 恆為 0.5
            h, cfc_outs[lvl] = self.cfc_cell(fpn_outs[lvl], h, t=t)

        # Phase 3: Modern Hopfield Layer（特徵聯想修復）
        final_outs = []
        for lvl in range(num_levels):
            hopfield_out = self.hopfield_layers[lvl](cfc_outs[lvl])
            final_outs.append(hopfield_out)

        return tuple(final_outs)
