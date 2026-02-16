"""
Physical-constraint-driven multi-term loss for RNA 3D prediction.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.geometry import fape_loss

BACKBONE_DIST = 5.9
BACKBONE_TOL = 1.0
WC_PAIR_DIST = 10.4
WC_PAIR_TOL = 1.5
CLASH_THRESHOLD = 3.0


class RNA3DLoss(nn.Module):
    def __init__(
        self,
        w_fape: float = 1.0,
        w_dist: float = 0.3,
        w_ss: float = 0.5,
        w_violation: float = 0.02,
        w_clash: float = 0.1,
        w_chain_break: float = 0.05,
        w_pairing_geom: float = 0.2,
        fape_clamp: float = 10.0,
    ):
        super().__init__()
        self.w_fape = w_fape
        self.w_dist = w_dist
        self.w_ss = w_ss
        self.w_violation = w_violation
        self.w_clash = w_clash
        self.w_chain_break = w_chain_break
        self.w_pairing_geom = w_pairing_geom
        self.fape_clamp = fape_clamp

    def _build_target_frames(self, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple identity frame at each C1' for target (no rotation from coords)."""
        B, L, _ = coord.shape
        device = coord.device
        R = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
        return R, coord

    def forward(
        self,
        pred_coord: torch.Tensor,
        pred_frames: Tuple[torch.Tensor, torch.Tensor],
        target_coord: torch.Tensor,
        mask: torch.Tensor,
        pred_ss: Optional[torch.Tensor] = None,
        target_bp: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        pred_coord, target_coord: (B, L, 3)
        mask: (B, L)
        pred_ss: (B, L, L) pairing probability
        target_bp: (B, L, L) 0/1 base pair label
        """
        B, L, _ = pred_coord.shape
        device = pred_coord.device
        target_R, target_t = self._build_target_frames(target_coord)

        losses = {}

        if self.w_fape > 0:
            l_fape = fape_loss(
                pred_frames,
                (target_R, target_t),
                pred_coord,
                target_coord,
                mask=mask,
                clamp=self.fape_clamp,
                d_clamp=self.fape_clamp,
            )
            losses["fape"] = l_fape

        if self.w_dist > 0:
            d_pred = torch.sqrt(((pred_coord.unsqueeze(2) - pred_coord.unsqueeze(1)) ** 2).sum(-1) + 1e-8)
            d_tgt = torch.sqrt(((target_coord.unsqueeze(2) - target_coord.unsqueeze(1)) ** 2).sum(-1) + 1e-8)
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            l_dist = (F.smooth_l1_loss(d_pred, d_tgt, reduction="none") * pair_mask).sum() / (pair_mask.sum() + 1e-8)
            losses["dist"] = l_dist

        if self.w_ss > 0 and pred_ss is not None and target_bp is not None:
            l_ss = F.binary_cross_entropy(pred_ss.clamp(1e-5, 1 - 1e-5), target_bp, reduction="none")
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            l_ss = (l_ss * pair_mask).sum() / (pair_mask.sum() + 1e-8)
            losses["ss"] = l_ss

        if self.w_chain_break > 0:
            d_adj = torch.sqrt(((pred_coord[:, 1:] - pred_coord[:, :-1]) ** 2).sum(-1) + 1e-8)
            target_adj = BACKBONE_DIST
            m = mask[:, 1:] * mask[:, :-1]
            l_chain = (F.smooth_l1_loss(d_adj, d_adj.new_full(d_adj.shape, target_adj), reduction="none") * m).sum() / (m.sum() + 1e-8)
            losses["chain_break"] = l_chain

        if self.w_clash > 0:
            d = torch.sqrt(((pred_coord.unsqueeze(2) - pred_coord.unsqueeze(1)) ** 2).sum(-1) + 1e-8)
            clash_mask = torch.ones(B, L, L, device=device)
            for i in range(L - 1):
                clash_mask[:, i, i + 1] = 0
                clash_mask[:, i + 1, i] = 0
            clash_mask = clash_mask * mask.unsqueeze(2) * mask.unsqueeze(1)
            clash = (d < CLASH_THRESHOLD).float() * clash_mask
            l_clash = clash.sum() / (clash_mask.sum() + 1e-8)
            losses["clash"] = l_clash

        if self.w_pairing_geom > 0 and pred_ss is not None:
            d = torch.sqrt(((pred_coord.unsqueeze(2) - pred_coord.unsqueeze(1)) ** 2).sum(-1) + 1e-8)
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            pred_paired = (pred_ss > 0.5).float() * pair_mask
            if pred_paired.sum() > 0:
                target_d = WC_PAIR_DIST
                l_pair = (pred_paired * (d - target_d) ** 2).sum() / (pred_paired.sum() + 1e-8)
                losses["pairing_geom"] = l_pair
            else:
                losses["pairing_geom"] = pred_coord.new_tensor(0.0)

        if self.w_violation > 0:
            d_adj = torch.sqrt(((pred_coord[:, 1:] - pred_coord[:, :-1]) ** 2).sum(-1) + 1e-8)
            m = mask[:, 1:] * mask[:, :-1]
            violation = ((d_adj < BACKBONE_DIST - BACKBONE_TOL) | (d_adj > BACKBONE_DIST + BACKBONE_TOL)).float() * m
            l_viol = violation.sum() / (m.sum() + 1e-8)
            losses["violation"] = l_viol

        total = (
            self.w_fape * losses.get("fape", 0)
            + self.w_dist * losses.get("dist", 0)
            + self.w_ss * losses.get("ss", 0)
            + self.w_violation * losses.get("violation", 0)
            + self.w_clash * losses.get("clash", 0)
            + self.w_chain_break * losses.get("chain_break", 0)
            + self.w_pairing_geom * losses.get("pairing_geom", 0)
        )
        losses["total"] = total
        return losses
