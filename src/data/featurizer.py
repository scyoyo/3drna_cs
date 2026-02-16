"""
Sequence and structure feature encoding for RNA.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

# A=0, U=1, G=2, C=3, pad/gap=4
NUCLEOTIDE_TO_ID = {"A": 0, "U": 1, "G": 2, "C": 3, "-": 4, "N": 4}
ID_TO_NUCLEOTIDE = "AUGCN"


def encode_sequence(sequence: str, max_len: Optional[int] = None, pad_id: int = 4) -> np.ndarray:
    """Encode RNA sequence to int array. Returns (L,) or (max_len,) with padding."""
    seq = sequence.upper().replace("T", "U")
    ids = np.array([NUCLEOTIDE_TO_ID.get(c, 4) for c in seq], dtype=np.int64)
    if max_len is not None:
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = np.pad(ids, (0, max_len - len(ids)), constant_values=pad_id)
    return ids


def encode_sequence_torch(sequence: str, max_len: Optional[int] = None, device=None) -> torch.Tensor:
    """Encode to torch long tensor."""
    arr = encode_sequence(sequence, max_len=max_len)
    t = torch.from_numpy(arr).long()
    if device is not None:
        t = t.to(device)
    return t


def encode_msa(sequences: List[str], max_len: Optional[int] = None, pad_id: int = 4) -> np.ndarray:
    """Encode MSA to (S, L) int array."""
    L = max(len(s) for s in sequences) if sequences else 0
    if max_len is not None:
        L = min(L, max_len)
    S = len(sequences)
    arr = np.full((S, L), pad_id, dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = seq.upper().replace("T", "U")
        for j, c in enumerate(seq):
            if j >= L:
                break
            arr[i, j] = NUCLEOTIDE_TO_ID.get(c, 4)
    return arr


def position_bins(max_len: int, num_bins: int = 32) -> np.ndarray:
    """Relative position encoding bins for pair (i,j). Returns (max_len, max_len, num_bins)."""
    bins = np.zeros((max_len, max_len, num_bins), dtype=np.float32)
    for i in range(max_len):
        for j in range(max_len):
            d = j - i
            bin_idx = min(abs(d) % num_bins, num_bins - 1)
            bins[i, j, bin_idx] = 1.0
    return bins


def distance_bins(d_min: float = 2.0, d_max: float = 22.0, num_bins: int = 36) -> tuple:
    """Returns (bin_edges, step). For discretizing C1'-C1' distances."""

    step = (d_max - d_min) / num_bins
    edges = np.linspace(d_min, d_max, num_bins + 1)
    return edges, step


def distance_to_bin(d: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Continuous distance to bin index (0 .. num_bins-1). Values outside range clipped."""
    return np.clip(np.searchsorted(edges, d, side="right") - 1, 0, len(edges) - 2).astype(np.int64)
