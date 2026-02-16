"""
MSA parsing, subsampling, and covariation feature computation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_msa(path: str | Path) -> Tuple[List[str], List[str]]:
    """
    Load MSA from FASTA. Returns (headers, sequences).
    For multi-chain MSAs, each row may have placeholders (-) for other chains.
    """
    path = Path(path)
    headers = []
    sequences = []
    current_header = None
    current_seq = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append("".join(current_seq))
                current_header = line[1:].strip()
                current_seq = []
                headers.append(current_header)
            else:
                current_seq.append(line.upper().replace("T", "U"))
        if current_header is not None:
            sequences.append("".join(current_seq))

    return headers, sequences


def subsample_msa(
    headers: List[str],
    sequences: List[str],
    n_max: int = 512,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Subsample MSA to at most n_max sequences (random or first n_max)."""
    if len(sequences) <= n_max:
        return headers, sequences
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sequences), size=n_max, replace=False)
    idx = np.sort(idx)
    return [headers[i] for i in idx], [sequences[i] for i in idx]


def _letter_to_idx(c: str) -> int:
    return {"A": 0, "U": 1, "G": 2, "C": 3, "-": 4, "N": 4}.get(c.upper(), 4)


def msa_to_array(sequences: List[str], L: Optional[int] = None) -> np.ndarray:
    """(S, L) int array, 0=A, 1=U, 2=G, 3=C, 4=gap/N."""
    if L is None:
        L = max(len(s) for s in sequences)
    S = len(sequences)
    arr = np.full((S, L), 4, dtype=np.int64)
    for i, s in enumerate(sequences):
        for j, c in enumerate(s):
            if j >= L:
                break
            arr[i, j] = _letter_to_idx(c)
    return arr


def compute_mi(msa: np.ndarray) -> np.ndarray:
    """Mutual information matrix (L, L) from MSA (S, L). Uses 5-letter alphabet A,U,G,C,gap."""
    S, L = msa.shape
    f_i = np.zeros((L, 5))
    f_ij = np.zeros((L, L, 5, 5))
    for i in range(L):
        for a in range(5):
            f_i[i, a] = (msa[:, i] == a).sum() / S
    for i in range(L):
        for j in range(L):
            for a in range(5):
                for b in range(5):
                    f_ij[i, j, a, b] = ((msa[:, i] == a) & (msa[:, j] == b)).sum() / S

    mi = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            for a in range(5):
                for b in range(5):
                    p_ij = f_ij[i, j, a, b]
                    p_i = f_i[i, a]
                    p_j = f_i[j, b]
                    if p_ij > 1e-10 and p_i > 1e-10 and p_j > 1e-10:
                        mi[i, j] += p_ij * np.log(p_ij / (p_i * p_j + 1e-10) + 1e-10)
    return mi


def apc_correct(mi: np.ndarray) -> np.ndarray:
    """Average Product Correction: MI_ij - (MI_i. * MI_.j) / MI_.."""
    mi_apc = mi.copy()
    row = mi.mean(axis=1)
    col = mi.mean(axis=0)
    global_mean = mi.mean()
    for i in range(mi.shape[0]):
        for j in range(mi.shape[1]):
            mi_apc[i, j] = mi[i, j] - (row[i] * col[j]) / (global_mean + 1e-10)
    return np.maximum(mi_apc, 0)


def compute_covariation(sequences: List[str], use_apc: bool = True) -> np.ndarray:
    """
    Covariation (APC-corrected MI) matrix (L, L) from MSA sequences.
    High values suggest evolutionary coupling (e.g. base pairing).
    """
    msa = msa_to_array(sequences)
    mi = compute_mi(msa)
    if use_apc:
        return apc_correct(mi)
    return np.maximum(mi, 0)


def effective_sequence_count(msa: np.ndarray, identity_threshold: float = 0.8) -> float:
    """Neff: effective number of non-redundant sequences (simplified)."""
    S = msa.shape[0]
    if S <= 1:
        return float(S)
    # Simple estimate: weight by sequence identity to first sequence
    first = msa[0]
    identity = (msa == first).mean(axis=1)
    weights = 1.0 / (identity * (identity >= identity_threshold) + 1e-8)
    weights = np.clip(weights, 0, 1)
    return float(weights.sum())


def get_msa_features(
    msa_path: str | Path,
    n_max: int = 512,
    compute_covar: bool = True,
    seed: Optional[int] = None,
) -> dict:
    """
    Load MSA and compute features. Returns dict with:
    - msa_array: (S, L) int
    - covariation: (L, L) float if compute_covar
    - neff: float
    - headers, sequences (after subsample)
    """
    headers, sequences = load_msa(msa_path)
    headers, sequences = subsample_msa(headers, sequences, n_max=n_max, seed=seed)
    if not sequences:
        return {"msa_array": None, "covariation": None, "neff": 0.0, "headers": [], "sequences": []}

    msa_arr = msa_to_array(sequences)
    neff = effective_sequence_count(msa_arr)

    out = {"msa_array": msa_arr, "neff": neff, "headers": headers, "sequences": sequences}
    if compute_covar and msa_arr.shape[0] >= 2:
        out["covariation"] = compute_covariation(sequences, use_apc=True)
    else:
        out["covariation"] = np.zeros((msa_arr.shape[1], msa_arr.shape[1]))
    return out
