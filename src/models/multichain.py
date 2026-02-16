"""
Multi-chain assembly: chain-aware pair representation and symmetry.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class MultiChainEmbedding(nn.Module):
    """Embed chain id and copy number for each residue."""

    def __init__(self, max_chains: int = 16, max_copies: int = 8, embed_dim: int = 64):
        super().__init__()
        self.chain_embed = nn.Embedding(max_chains + 1, embed_dim, padding_idx=0)
        self.copy_embed = nn.Embedding(max_copies + 1, embed_dim, padding_idx=0)

    def forward(
        self,
        chain_ids: torch.Tensor,
        copy_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.chain_embed(chain_ids) + self.copy_embed(copy_ids)


class ChainPairBias(nn.Module):
    """Bias pair representation by whether (i,j) are same chain / same copy."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 2, num_heads)

    def forward(
        self,
        chain_embed: torch.Tensor,
        same_chain: torch.Tensor,
    ) -> torch.Tensor:
        """
        chain_embed: (B, L, D)
        same_chain: (B, L, L) 1 if same chain
        Returns: (B, L, L, num_heads) bias for attention.
        """
        B, L, D = chain_embed.shape
        left = chain_embed.unsqueeze(2).expand(B, L, L, D)
        right = chain_embed.unsqueeze(1).expand(B, L, L, D)
        pair_feat = torch.cat([left, right], dim=-1)
        bias = self.proj(pair_feat)
        return bias * same_chain.unsqueeze(-1).float()
