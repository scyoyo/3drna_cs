"""
Evoformer blocks: MSA processing and pair representation with triangle updates.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MSAEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 5, embed_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=4)

    def forward(self, msa_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(msa_ids)


class PairFeatureInit(nn.Module):
    """Initialize pair representation from single repr (outer product mean) + position + covariation."""

    def __init__(self, single_dim: int = 256, pair_dim: int = 128, num_position_bins: int = 32, pair_bias_dim: int = 2):
        super().__init__()
        self.outer_proj = nn.Linear(single_dim * single_dim, pair_dim)
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.pos_embed = nn.Embedding(num_position_bins, pair_dim)
        self.pair_bias_proj = nn.Linear(pair_bias_dim, pair_dim) if pair_bias_dim else None

    def forward(
        self,
        single_repr: torch.Tensor,
        pair_bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = single_repr.shape
        outer = torch.einsum("bni,bnj->bnij", single_repr, single_repr).reshape(B, L, L, -1)
        z = self.outer_proj(outer)
        if pair_bias is not None:
            if pair_bias.shape[-1] != self.pair_dim and self.pair_bias_proj is not None:
                pair_bias = self.pair_bias_proj(pair_bias)
            z = z + pair_bias
        pos = torch.arange(L, device=single_repr.device)
        rel_pos = (pos[:, None] - pos[None, :]).clamp(-15, 15) + 15
        z = z + self.pos_embed(rel_pos)
        if mask is not None:
            pair_mask = mask[:, :, None] * mask[:, None, :]
            z = z * pair_mask.unsqueeze(-1)
        return z


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle update (outgoing or incoming)."""

    def __init__(self, pair_dim: int, mix: str = "outgoing"):
        super().__init__()
        self.mix = mix
        self.norm = nn.LayerNorm(pair_dim)
        self.left_proj = nn.Linear(pair_dim, pair_dim)
        self.right_proj = nn.Linear(pair_dim, pair_dim)
        self.gate = nn.Linear(pair_dim, pair_dim)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.norm(z)
        left = self.left_proj(z)
        right = self.right_proj(z)
        if self.mix == "outgoing":
            out = torch.einsum("bikd,bjkd->bijd", left, right)
        else:
            out = torch.einsum("bkid,bkjd->bijd", left, right)
        out = out * self.gate(z).sigmoid()
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return z + out


class TriangleAttention(nn.Module):
    """Row or column-wise triangle self-attention."""

    def __init__(self, pair_dim: int, num_heads: int = 4, wise: str = "row"):
        super().__init__()
        self.wise = wise
        self.norm = nn.LayerNorm(pair_dim)
        self.attn = nn.MultiheadAttention(pair_dim, num_heads, batch_first=True)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _, D = z.shape
        z = self.norm(z)
        if self.wise == "row":
            z_flat = z.reshape(B * L, L, D)
            attn_mask = None
            if mask is not None:
                attn_mask = (1 - mask).unsqueeze(1).expand(B, L, L).reshape(B * L, 1, L)
                attn_mask = attn_mask.bool().masked_fill(attn_mask, float("-inf"))
            out, _ = self.attn(z_flat, z_flat, z_flat, attn_mask=attn_mask)
            out = out.reshape(B, L, L, D)
        else:
            z_t = z.transpose(1, 2)
            z_flat = z_t.reshape(B * L, L, D)
            out, _ = self.attn(z_flat, z_flat, z_flat)
            out = out.reshape(B, L, L, D).transpose(1, 2)
        return z + out


class MSARowAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, msa: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        msa = self.norm(msa)
        key_padding = None if mask is None else (1 - mask).bool()
        out, _ = self.attn(msa, msa, msa, key_padding_mask=key_padding)
        return msa + out


class MSAColumnAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, msa: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        msa = self.norm(msa)
        msa_t = msa.transpose(1, 2)
        key_padding = None if mask is None else (1 - mask).transpose(1, 2).bool()
        out, _ = self.attn(msa_t, msa_t, msa_t, key_padding_mask=key_padding)
        out = out.transpose(1, 2)
        return msa + out


class EvoformerBlock(nn.Module):
    def __init__(
        self,
        msa_dim: int = 256,
        pair_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.msa_row = MSARowAttention(msa_dim, num_heads, dropout)
        self.msa_col = MSAColumnAttention(msa_dim, num_heads, dropout)
        self.tri_out = TriangleMultiplicativeUpdate(pair_dim, "outgoing")
        self.tri_in = TriangleMultiplicativeUpdate(pair_dim, "ingoing")
        self.tri_attn_row = TriangleAttention(pair_dim, num_heads // 2, "row")
        self.tri_attn_col = TriangleAttention(pair_dim, num_heads // 2, "col")
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(pair_dim * 4, pair_dim),
        )
        self.msa_dim = msa_dim
        self.pair_dim = pair_dim

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        msa = self.msa_row(msa, msa_mask)
        msa = self.msa_col(msa, msa_mask)
        pair = pair + self.tri_out(pair, pair_mask)
        pair = pair + self.tri_in(pair, pair_mask)
        pair = pair + self.tri_attn_row(pair, pair_mask)
        pair = pair + self.tri_attn_col(pair, pair_mask)
        pair = pair + self.pair_transition(pair)
        return msa, pair


class EvoformerStack(nn.Module):
    def __init__(
        self,
        num_blocks: int = 24,
        msa_dim: int = 256,
        pair_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            EvoformerBlock(msa_dim, pair_dim, num_heads, dropout) for _ in range(num_blocks)
        ])

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            msa, pair = block(msa, pair, msa_mask, pair_mask)
        return msa, pair
