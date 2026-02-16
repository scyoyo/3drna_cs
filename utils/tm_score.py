"""
Local TM-score computation for RNA C1' coordinates.
TM-score = max over alignments of (1/L_ref) * sum_i 1 / (1 + (d_i/d0)^2)
d0 = 0.6 * (L_ref - 0.5)^0.5 - 2.5 for L_ref >= 30; else tabulated.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def _d0(L: int) -> float:
    if L >= 30:
        return 0.6 * (L - 0.5) ** 0.5 - 2.5
    if L < 12:
        return 0.3
    if L < 16:
        return 0.4
    if L < 20:
        return 0.5
    if L < 24:
        return 0.6
    return 0.7


def _kabsch_np(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """P, Q (N,3). Returns R (3,3), t (3,)."""
    p_cent = P.mean(axis=0)
    q_cent = Q.mean(axis=0)
    Pc = P - p_cent
    Qc = Q - q_cent
    H = Pc.T @ Qc
    U, S, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[-1] *= -1
        R = Vh.T @ U.T
    t = q_cent - R @ p_cent
    return R, t


def tm_score(
    pred: np.ndarray,
    target: np.ndarray,
    L_ref: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    TM-score for C1' coordinates. Sequence-dependent (1-to-1 alignment).
    pred, target: (N, 3) in Angstroms.
    L_ref: reference length (default N).
    mask: (N,) 1 = include in score.
    """
    if mask is not None:
        pred = pred[mask > 0]
        target = target[mask > 0]
    N = pred.shape[0]
    if N == 0:
        return 0.0
    if L_ref is None:
        L_ref = N

    R, t = _kabsch_np(pred.astype(np.float64), target.astype(np.float64))
    pred_aligned = (pred @ R.T) + t
    d = np.sqrt(((pred_aligned - target) ** 2).sum(axis=1))
    d0 = _d0(L_ref)
    s = 1.0 / (1.0 + (d / d0) ** 2)
    return float(np.mean(s))


def tm_score_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    L_ref: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Torch version; pred, target (N,3) on same device."""
    if mask is not None:
        pred = pred[mask > 0]
        target = target[mask > 0]
    return pred.new_tensor(
        tm_score(
            pred.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
            L_ref=L_ref,
            mask=mask.cpu().numpy() if mask is not None else None,
        )
    )
