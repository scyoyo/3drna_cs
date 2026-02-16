"""
Diversity-based selection: from N candidates choose 5 that maximize structural diversity.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def pairwise_rmsd_matrix(coords_list: List[np.ndarray], align: bool = True) -> np.ndarray:
    """
    coords_list: list of (L, 3) arrays (same L).
    Returns (N, N) RMSD matrix. If align=True, Kabsch-align each pair before RMSD.
    """
    N = len(coords_list)
    L = coords_list[0].shape[0]
    rmsd_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            a, b = coords_list[i], coords_list[j]
            if align:
                R, t = _kabsch(a, b)
                a = (a @ R.T) + t
            d = np.sqrt(((a - b) ** 2).sum(axis=1).mean() + 1e-10)
            rmsd_mat[i, j] = rmsd_mat[j, i] = d
    return rmsd_mat


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple:
    """Kabsch: R, t so that R@P + t best matches Q."""
    p_cent = P.mean(axis=0)
    q_cent = Q.mean(axis=0)
    Pc = P - p_cent
    Qc = Q - q_cent
    H = Pc.T @ Qc
    U, _, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh = Vh.copy()
        Vh[-1] *= -1
        R = Vh.T @ U.T
    t = q_cent - R @ p_cent
    return R, t


def select_diverse_top5(
    candidates: List[np.ndarray],
    scores: Optional[List[float]] = None,
    n: int = 5,
) -> List[int]:
    """
    From N candidates, select n indices that maximize diversity (greedy).
    candidates: list of (L, 3) C1' coordinate arrays.
    scores: optional confidence per candidate (higher = better); first pick is max score.
    Returns list of n indices into candidates.
    """
    N = len(candidates)
    if N <= n:
        return list(range(N))
    rmsd_mat = pairwise_rmsd_matrix(candidates, align=True)
    if scores is not None:
        selected = [int(np.argmax(scores))]
    else:
        selected = [0]
    while len(selected) < n:
        min_rmsd_to_set = rmsd_mat[:, selected].min(axis=1).copy()
        for idx in selected:
            min_rmsd_to_set[idx] = -1
        best_next = np.argmax(min_rmsd_to_set)
        selected.append(int(best_next))
    return selected[:n]
