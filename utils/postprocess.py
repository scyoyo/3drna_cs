"""
Coordinate post-processing: geometry validation and clash fixing.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

# C1'-C1' backbone distance ~5.9 A; WC pair ~10.4 A
BACKBONE_DIST_MIN = 4.5
BACKBONE_DIST_MAX = 7.5
CLASH_THRESHOLD = 3.0
WC_PAIR_DIST = 10.4
WC_PAIR_TOL = 1.5


def detect_clashes(coords: np.ndarray, threshold: float = CLASH_THRESHOLD) -> np.ndarray:
    """coords (N,3). Returns (N,N) boolean matrix of clashes (excluding adjacent)."""
    N = coords.shape[0]
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    mask = np.ones((N, N), dtype=bool)
    for i in range(N - 1):
        mask[i, i + 1] = False
        mask[i + 1, i] = False
    mask[np.diag_indices(N)] = False
    return (d < threshold) & mask


def backbone_distances(coords: np.ndarray) -> np.ndarray:
    """(N,3) -> (N-1,) consecutive C1'-C1' distances."""
    return np.linalg.norm(np.diff(coords, axis=0), axis=1)


def violation_mask_backbone(coords: np.ndarray) -> np.ndarray:
    """Returns (N-1,) bool: True = violation (outside [BACKBONE_DIST_MIN, BACKBONE_DIST_MAX])."""
    d = backbone_distances(coords)
    return (d < BACKBONE_DIST_MIN) | (d > BACKBONE_DIST_MAX)


def clamp_coordinates(coords: np.ndarray, low: float = -999.999, high: float = 9999.999) -> np.ndarray:
    """Clamp to submission range (legacy PDB format)."""
    return np.clip(coords, low, high).astype(np.float64)


def center_and_align(
    pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Kabsch align pred to target; return aligned pred."""
    if mask is not None:
        pred = pred[mask > 0]
        target = target[mask > 0]
    p_cent = pred.mean(axis=0)
    q_cent = target.mean(axis=0)
    Pc = pred - p_cent
    Qc = target - q_cent
    H = Pc.T @ Qc
    U, _, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh = Vh.copy()
        Vh[-1] *= -1
        R = Vh.T @ U.T
    t = q_cent - R @ p_cent
    return (pred @ R.T) + t
