"""
Structure module: IPA (Invariant Point Attention) with nucleotide rigid body frames.
Output: C1' coordinates (N, 3).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.geometry import rotation_matrix_from_quaternion


class InvariantPointAttention(nn.Module):
    """SE(3)-equivariant point attention."""

    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        num_heads: int = 4,
        num_point_qk: int = 4,
        num_point_v: int = 8,
        num_channel: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.scalar_dim = num_channel
        self.point_dim = num_point_v * 3

        self.to_q_scalar = nn.Linear(single_dim, num_heads * num_channel)
        self.to_k_scalar = nn.Linear(single_dim, num_heads * num_channel)
        self.to_v_scalar = nn.Linear(single_dim, num_heads * num_channel)
        self.to_q_point = nn.Linear(single_dim, num_heads * num_point_qk * 3)
        self.to_k_point = nn.Linear(single_dim, num_heads * num_point_qk * 3)
        self.to_v_point = nn.Linear(single_dim, num_heads * num_point_v * 3)
        self.to_pair_bias = nn.Linear(pair_dim, num_heads)
        self.to_out = nn.Linear(num_heads * (num_channel + num_point_v * 3), single_dim)

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = single.shape
        H = self.num_heads
        qs = self.to_q_scalar(single).view(B, L, H, -1)
        ks = self.to_k_scalar(single).view(B, L, H, -1)
        vs = self.to_v_scalar(single).view(B, L, H, -1)
        qp = self.to_q_point(single).view(B, L, H, self.num_point_qk, 3)
        kp = self.to_k_point(single).view(B, L, H, self.num_point_qk, 3)
        vp = self.to_v_point(single).view(B, L, H, self.num_point_v, 3)

        qp_global = torch.einsum("bnij,bnaj->bnai", R, qp)
        kp_global = torch.einsum("bnij,bnaj->bnai", R, kp)
        vp_global = torch.einsum("bnij,bnaj->bnai", R, vp)
        qp_global = qp_global + t.unsqueeze(2).unsqueeze(2)
        kp_global = kp_global + t.unsqueeze(2).unsqueeze(2)
        vp_global = vp_global + t.unsqueeze(2).unsqueeze(2)

        attn_scalar = torch.einsum("bnhd,bmhd->bnmh", qs, ks) / math.sqrt(self.scalar_dim)
        attn_point = -0.5 * ((qp_global.unsqueeze(3) - kp_global.unsqueeze(2)) ** 2).sum(dim=(-2, -1))
        pair_bias = self.to_pair_bias(pair).permute(0, 3, 1, 2)
        attn = attn_scalar + attn_point + pair_bias
        attn = attn / math.sqrt(H)
        if mask is not None:
            attn = attn.masked_fill((1 - mask).unsqueeze(2).unsqueeze(3).bool(), float("-inf"))
        attn = F.softmax(attn, dim=2)

        out_scalar = torch.einsum("bnmh,bmhd->bnhd", attn, vs).reshape(B, L, -1)
        out_point = torch.einsum("bnmh,bmhav->bnhav", attn, vp_global).reshape(B, L, H, self.num_point_v, 3)
        out_point_global = torch.einsum("bnij,bnhav->bnaj", R.transpose(-2, -1), out_point - t.unsqueeze(2).unsqueeze(2))
        out_point_flat = out_point_global.reshape(B, L, -1)
        out = self.to_out(torch.cat([out_scalar, out_point_flat], dim=-1))
        return single + out


class StructureModule(nn.Module):
    """
    Iterative structure refinement with IPA. Each layer predicts rotation and translation updates.
    Final C1' = t (translation of frame).
    """

    def __init__(
        self,
        single_dim: int = 256,
        pair_dim: int = 128,
        num_layers: int = 8,
        num_heads: int = 4,
        num_point_qk: int = 4,
        num_point_v: int = 8,
        num_channel: int = 128,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            InvariantPointAttention(single_dim, pair_dim, num_heads, num_point_qk, num_point_v, num_channel)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(single_dim)
        self.to_quaternion = nn.Linear(single_dim, 4)
        self.to_translation = nn.Linear(single_dim, 3)
        self.single_dim = single_dim

    def _init_frames(self, B: int, L: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        R = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3)
        t = torch.zeros(B, L, 3, device=device)
        return R, t

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = single.shape
        R, t = self._init_frames(B, L, single.device)

        for layer in self.layers:
            single = layer(single, pair, R, t, mask)
            single_norm = self.norm(single)
            dq = self.to_quaternion(single_norm)
            dt = self.to_translation(single_norm)
            dq = F.normalize(dq, dim=-1)
            dR = rotation_matrix_from_quaternion(dq)
            R = torch.einsum("bnij,bnjk->bnik", dR, R)
            t = t + torch.einsum("bnij,bnj->bni", R, dt)

        return single, pair, (R, t)
