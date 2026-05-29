# Copyright (c) 2026. All rights reserved.
"""
LNN-Hopfield FPN (mmdet 2.x 相容版)
=====================================
將 reference_module/lnn_hopfield_fpn.py 適配為 mmdet 2.x / mmcv 1.x API：
  - mmengine.model.BaseModule  → mmcv.runner.BaseModule
  - mmdet.registry.MODELS      → mmdet.models.builder.NECKS
  - mmdet.utils ConfigType     → Optional[dict]
  - from .fpn import FPN       → from mmdet.models.necks.fpn import FPN

使用方式（config）::

    neck=dict(
        type='LNNHopfieldFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        num_prototypes=128,
        hopfield_beta=1.0,
        cfc_hidden_ratio=1.0,
    )
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models.builder import NECKS
from mmdet.models.necks.fpn import FPN
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. CfC Cell
# ---------------------------------------------------------------------------

class CfCCell(BaseModule):
    """Closed-form Continuous-time Cell for spatial feature maps."""

    def __init__(self, channels: int, hidden_ratio: float = 1.0,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        hidden = int(channels * hidden_ratio)

        self.g_net = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, channels),
        )
        self.h_net = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, channels),
        )
        self.f_net = nn.Sequential(
            nn.Linear(channels * 2, channels),
        )
        self.spatial_mod = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor, h_prev: Tensor,
                t: float = 1.0) -> Tuple[Tensor, Tensor]:
        B, C, H, W = x.shape
        x_gap = F.adaptive_avg_pool2d(x, 1).reshape(B, C)
        combined = torch.cat([x_gap, h_prev], dim=1)

        g = self.g_net(combined)
        h_local = self.h_net(combined)
        f_val = self.f_net(combined)

        gate = torch.sigmoid(-f_val * t)
        h_new = gate * g + (1.0 - gate) * h_local

        scale = torch.sigmoid(h_new).view(B, C, 1, 1)
        out = x * scale + self.spatial_mod(x)

        return h_new, out


# ---------------------------------------------------------------------------
# 2. Modern Hopfield Layer
# ---------------------------------------------------------------------------

class ModernHopfieldLayer(BaseModule):
    """Modern Hopfield Layer for feature association and auto-repair."""

    def __init__(self, channels: int, num_prototypes: int = 128,
                 beta: float = 1.0,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.beta = beta

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, channels))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0)).squeeze(0)

        self.query_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_gap = F.adaptive_avg_pool2d(x, 1).reshape(B, C)

        q = self.query_proj(x_gap)
        energy = self.beta * (q @ self.prototypes.T)
        attn = F.softmax(energy, dim=-1)
        recalled = attn @ self.prototypes

        mod_signal = self.norm(self.out_proj(recalled + x_gap))
        mod = torch.sigmoid(mod_signal).view(B, C, 1, 1)

        return x * mod


# ---------------------------------------------------------------------------
# 3. LNNHopfieldFPN
# ---------------------------------------------------------------------------

@NECKS.register_module(force=True)
class LNNHopfieldFPN(FPN):
    """LNN-Hopfield FPN: OOD-Robust Feature Pyramid Network (mmdet 2.x)."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        upsample_cfg: dict = dict(mode='nearest'),
        init_cfg: Optional[dict] = dict(
            type='Xavier', layer='Conv2d', distribution='uniform'),
        num_prototypes: int = 128,
        hopfield_beta: float = 1.0,
        cfc_hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__(
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

        self.cfc_cell = CfCCell(out_channels, hidden_ratio=cfc_hidden_ratio)
        self.hopfield_layers = nn.ModuleList([
            ModernHopfieldLayer(
                channels=out_channels,
                num_prototypes=num_prototypes,
                beta=hopfield_beta,
            )
            for _ in range(num_outs)
        ])

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        fpn_outs = list(super().forward(inputs))
        num_levels = len(fpn_outs)
        B = fpn_outs[0].shape[0]
        device = fpn_outs[0].device

        h = torch.zeros(B, self.out_channels, device=device)
        cfc_outs = [None] * num_levels

        for step, lvl in enumerate(reversed(range(num_levels))):
            t = float(step + 1)
            h, cfc_outs[lvl] = self.cfc_cell(fpn_outs[lvl], h, t=t)

        final_outs = []
        for lvl in range(num_levels):
            hopfield_out = self.hopfield_layers[lvl](cfc_outs[lvl])
            final_outs.append(hopfield_out)

        return tuple(final_outs)
