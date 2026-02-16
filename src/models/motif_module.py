"""
3D motif awareness: soft constraints for known RNA motifs (K-turn, GNRA, etc.).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MotifModule(nn.Module):
    """
    Lightweight motif module: from secondary structure + pair repr, predict motif type logits
    per residue or per loop. Used as auxiliary loss or to bias structure module.
    """

    def __init__(self, pair_dim: int = 128, num_motif_types: int = 16):
        super().__init__()
        self.pair_dim = pair_dim
        self.num_motif_types = num_motif_types
        self.proj = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, num_motif_types),
        )

    def forward(
        self,
        pair: torch.Tensor,
        ss_logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pair: (B, L, L, pair_dim)
        Returns: (B, L, num_motif_types) logits for motif type at each position (simplified: from diagonal of pair).
        """
        B, L, _, D = pair.shape
        diag = torch.diagonal(pair, dim1=1, dim2=2)
        return self.proj(diag)
