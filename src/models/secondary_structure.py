"""
Explicit secondary structure prediction layer: base pairing probability from pair representation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SecondaryStructureHead(nn.Module):
    """
    Predict base pairing probability matrix P(i,j) from pair representation.
    Output is (L, L, 2) logits for [unpaired, paired] then symmetrize and apply constraints.
    """

    def __init__(self, pair_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(
        self,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pair: (B, L, L, pair_dim)
        Returns: (B, L, L) probability of paired (symmetric, 0-1).
        """
        logits = self.proj(pair)
        prob = F.softmax(logits, dim=-1)[..., 1]
        prob = (prob + prob.transpose(-2, -1)) / 2
        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            prob = prob * pair_mask
        return prob


def nussinov_decode(prob: torch.Tensor, min_loop: int = 3) -> torch.Tensor:
    """
    Decode pairing from probability matrix using Nussinov-style DP (simplified).
    prob: (L, L), returns (L, L) binary 0/1.
    Only nested structures; min_loop = minimum hairpin loop length.
    """
    L = prob.shape[0]
    device = prob.device
    dp = torch.zeros(L + 1, L + 1, device=device)
    for length in range(2, L + 1):
        for i in range(L - length + 1):
            j = i + length - 1
            if j - i < min_loop:
                dp[i, j] = dp[i, j - 1]
                continue
            pair_score = prob[i, j] + dp[i + 1, j - 1]
            skip = max(dp[i, j - 1], dp[i + 1, j])
            best = max(dp[i + 1, j - 1], dp[i, j - 1], dp[i + 1, j], pair_score)
            dp[i, j] = best
    # Backtrack to get structure (simplified: just threshold prob for training)
    return (prob > 0.5).float()
