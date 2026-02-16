"""
SO(3) rotation, frame operations, and FAPE utilities for RNA structure.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_matrix_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    """q: (..., 4) w,x,y,z. Returns (..., 3, 3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def quaternion_from_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
    """R: (..., 3, 3). Returns (..., 4) w,x,y,z."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = torch.sqrt(torch.clamp(1 + trace, min=1e-8)) / 2
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w + 1e-8)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w + 1e-8)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w + 1e-8)
    return torch.stack([w, x, y, z], dim=-1)


def invert_rigid(T: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert rigid transform (R, t). R (..., 3,3), t (..., 3)."""
    R, t = T
    R_inv = R.transpose(-2, -1)
    t_inv = -torch.einsum("...ij,...j->...i", R_inv, t)
    return R_inv, t_inv


def apply_rigid_to_point(R: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply (R,t) to points x. R (...,3,3), t (...,3), x (...,3)."""
    return torch.einsum("...ij,...j->...i", R, x) + t


def fape_loss(
    pred_frames: Tuple[torch.Tensor, torch.Tensor],
    target_frames: Tuple[torch.Tensor, torch.Tensor],
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp: float = 10.0,
    d_clamp: float = 10.0,
) -> torch.Tensor:
    """
    Frame Aligned Point Error.
    pred/target_frames: (R, t) each R (B,N,3,3), t (B,N,3).
    pred/target_positions: (B,N,3) C1' positions.
    mask: (B,N) 1 = valid.
    """
    R_pred, t_pred = pred_frames
    R_tgt, t_tgt = target_frames
    B, N, _ = pred_positions.shape

    if mask is None:
        mask = torch.ones(B, N, device=pred_positions.device, dtype=torch.float32)

    R_pred_inv = R_pred.transpose(-2, -1)
    R_tgt_inv = R_tgt.transpose(-2, -1)

    # Local frame coordinates: p_i in frame j
    # d_ij = R_j^{-1} (p_i - t_j)
    pred_local = torch.einsum("bnij,bnj->bni", R_pred_inv, pred_positions - t_pred)  # (B,N,N,3) but we need (B,N,N,3)
    # Actually: for each pair (i,j) we want R_j^{-1}(p_i - t_j)
    pred_pos_expand = pred_positions.unsqueeze(2)  # (B,N,1,3)
    t_pred_expand = t_pred.unsqueeze(1)  # (B,1,N,3)
    R_pred_inv_expand = R_pred_inv.unsqueeze(1)  # (B,1,N,3,3)
    pred_diff = pred_pos_expand - t_pred_expand  # (B,N,N,3)
    pred_local = torch.einsum("bijnk,bijk->bijn", R_pred_inv_expand, pred_diff)  # (B,N,N,3)

    tgt_pos_expand = target_positions.unsqueeze(2)
    t_tgt_expand = t_tgt.unsqueeze(1)
    R_tgt_inv_expand = R_tgt_inv.unsqueeze(1)
    tgt_diff = tgt_pos_expand - t_tgt_expand
    tgt_local = torch.einsum("bijnk,bijk->bijn", R_tgt_inv_expand, tgt_diff)

    d = torch.sqrt(((pred_local - tgt_local) ** 2).sum(dim=-1) + 1e-8)
    d = torch.clamp(d, max=d_clamp)

    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # (B,N,N)
    loss = (d * pair_mask).sum() / (pair_mask.sum() + 1e-8)
    return loss


def rmsd(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """pred, target (N,3). mask (N,). Returns scalar."""
    if mask is not None:
        pred = pred[mask > 0]
        target = target[mask > 0]
    if pred.numel() == 0:
        return pred.new_tensor(0.0)
    diff = pred - target
    return torch.sqrt((diff ** 2).sum() / pred.shape[0] + 1e-8)


def kabsch_align(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kabsch alignment: find R,t such that R@pred + t best matches target.
    Returns (R, t) and aligned pred = R @ pred + t.
    pred, target: (N, 3).
    """
    if mask is not None:
        pred = pred[mask > 0]
        target = target[mask > 0]
    pred_centroid = pred.mean(dim=0)
    tgt_centroid = target.mean(dim=0)
    P = pred - pred_centroid
    Q = target - tgt_centroid
    H = P.T @ Q
    U, _, Vh = torch.linalg.svd(H)
    R = Vh.T @ U.T
    if R.det() < 0:
        Vh = Vh.clone()
        Vh[-1] *= -1
        R = Vh.T @ U.T
    t = tgt_centroid - R @ pred_centroid
    return R, t
