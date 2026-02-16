"""
Full RNA 3D model: MSA -> Evoformer -> single/pair -> SS head -> Structure module -> C1' coords.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .evoformer import MSAEmbedding, EvoformerStack, PairFeatureInit
from .secondary_structure import SecondaryStructureHead
from .structure_module import StructureModule
from .motif_module import MotifModule

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.geometry import rotation_matrix_from_quaternion


class RNA3DModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 5,
        msa_single_dim: int = 256,
        msa_pair_dim: int = 128,
        evoformer_blocks: int = 24,
        evoformer_heads: int = 8,
        structure_layers: int = 8,
        structure_heads: int = 4,
        num_recycling: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.msa_embed = MSAEmbedding(vocab_size, msa_single_dim)
        self.pair_init = PairFeatureInit(msa_single_dim, msa_pair_dim)
        self.evoformer = EvoformerStack(
            num_blocks=evoformer_blocks,
            msa_dim=msa_single_dim,
            pair_dim=msa_pair_dim,
            num_heads=evoformer_heads,
            dropout=dropout,
        )
        self.ss_head = SecondaryStructureHead(msa_pair_dim)
        self.structure_module = StructureModule(
            single_dim=msa_single_dim,
            pair_dim=msa_pair_dim,
            num_layers=structure_layers,
            num_heads=structure_heads,
        )
        self.motif_module = MotifModule(msa_pair_dim)
        self.num_recycling = num_recycling
        self.msa_single_dim = msa_single_dim
        self.msa_pair_dim = msa_pair_dim

    def _reduce_msa_to_single(self, msa: torch.Tensor) -> torch.Tensor:
        return msa[:, 0]

    def forward(
        self,
        msa_ids: torch.Tensor,
        pair_features: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        num_recycling: Optional[int] = None,
    ) -> dict:
        """
        msa_ids: (B, S, L)
        pair_features: (B, L, L, D) e.g. covariation + position
        Returns dict with coord (B,L,3), ss_logits (B,L,L), pair (B,L,L,pair_dim), single (B,L,single_dim).
        """
        B, S, L = msa_ids.shape
        if pair_mask is None:
            pair_mask = torch.ones(B, L, device=msa_ids.device)
        if msa_mask is None:
            msa_mask = torch.ones(B, S, L, device=msa_ids.device)

        msa = self.msa_embed(msa_ids)
        single = self._reduce_msa_to_single(msa)
        pair = self.pair_init(single, pair_features, pair_mask)

        n_recycle = num_recycling if num_recycling is not None else self.num_recycling
        for _ in range(n_recycle):
            msa, pair = self.evoformer(msa, pair, msa_mask, pair_mask)
            single = self._reduce_msa_to_single(msa)
            single, pair, (R, t) = self.structure_module(single, pair, pair_mask)
            pair = pair + 0.1 * pair_features

        ss_logits = self.ss_head(pair, pair_mask)
        motif_logits = self.motif_module(pair, ss_logits, pair_mask)

        return {
            "coord": t,
            "frames": (R, t),
            "ss_logits": ss_logits,
            "pair": pair,
            "single": single,
            "motif_logits": motif_logits,
        }
