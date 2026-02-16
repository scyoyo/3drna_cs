"""
PyTorch Dataset for RNA 3D structure training.
Input: sequence + MSA features + optional secondary structure prior.
Labels: C1' coordinates + derived base pair map + distance map.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .featurizer import encode_sequence, encode_msa, distance_bins, distance_to_bin
from .msa_processing import get_msa_features, load_msa, subsample_msa
from .stoichiometry import get_chain_sequences, parse_stoichiometry, parse_fasta


class RNAStructureDataset(Dataset):
    """
    Dataset that yields:
    - seq_ids: (L,) nucleotide indices
    - msa_ids: (S, L) MSA token indices
    - msa_mask: (S, L) 1 = valid
    - pair_features: (L, L, D) e.g. covariation, position
    - coord: (L, 3) C1' coordinates
    - mask: (L,) 1 = valid residue
    - base_pair_map: (L, L) 0/1 from 3D-derived pairing (optional)
    - dist_map: (L, L) C1'-C1' distances (optional)
    """

    def __init__(
        self,
        sequences_csv: str | Path,
        labels_csv: str | Path,
        msa_dir: Optional[str | Path] = None,
        pdb_rna_dir: Optional[str | Path] = None,
        max_seq_len: int = 512,
        msa_max_depth: int = 128,
        crop_len: Optional[Tuple[int, int]] = None,
        include_base_pair_label: bool = True,
        include_dist_map: bool = True,
        target_id_column: str = "target_id",
        seed: Optional[int] = None,
    ):
        self.seq_df = pd.read_csv(sequences_csv)
        self.labels_df = pd.read_csv(labels_csv)
        self.msa_dir = Path(msa_dir) if msa_dir else None
        self.pdb_rna_dir = Path(pdb_rna_dir) if pdb_rna_dir else None
        self.max_seq_len = max_seq_len
        self.msa_max_depth = msa_max_depth
        self.crop_len = crop_len  # (min, max) for random crop
        self.include_base_pair_label = include_base_pair_label
        self.include_dist_map = include_dist_map
        self.target_id_column = target_id_column
        self.rng = np.random.default_rng(seed)

        # Build index: target_id -> list of (start_row, num_residues) in labels
        self.target_to_labels = {}
        for i, row in self.labels_df.iterrows():
            tid = row["ID"].rsplit("_", 1)[0] if "_" in str(row["ID"]) else str(row["ID"])
            if tid not in self.target_to_labels:
                self.target_to_labels[tid] = []
            self.target_to_labels[tid].append(i)
        self.target_ids = [tid for tid in self.seq_df[self.target_id_column].astype(str) if tid in self.target_to_labels]
        self._dist_edges, _ = distance_bins(num_bins=36)

    def __len__(self) -> int:
        return len(self.target_ids)

    def _get_labels_for_target(self, target_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get C1' coords (N, 3), resnames (N,), resids (N,) for one target. One structure only (first ref)."""
        rows = self.target_to_labels[target_id]
        # First structure: x_1, y_1, z_1
        x = self.labels_df.loc[rows, "x_1"].values.astype(np.float32)
        y = self.labels_df.loc[rows, "y_1"].values.astype(np.float32)
        z = self.labels_df.loc[rows, "z_1"].values.astype(np.float32)
        coords = np.column_stack([x, y, z])
        resnames = self.labels_df.loc[rows, "resname"].values
        resids = self.labels_df.loc[rows, "resid"].values.astype(np.int32)
        return coords, resnames, resids

    def _derive_base_pair_map(self, coords: np.ndarray, threshold_wc: float = 11.0) -> np.ndarray:
        """Binary (L,L) from C1'-C1' distance < threshold (excluding adjacent)."""
        L = coords.shape[0]
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        bp = ((d > 4.5) & (d < threshold_wc)).astype(np.float32)
        for i in range(L - 1):
            bp[i, i + 1] = 0
            bp[i + 1, i] = 0
        np.fill_diagonal(bp, 0)
        return bp

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target_id = self.target_ids[idx]
        row = self.seq_df[self.seq_df[self.target_id_column].astype(str) == target_id].iloc[0]
        sequence = str(row["sequence"]).upper().replace("T", "U")
        stoichiometry = str(row.get("stoichiometry", "A:1"))
        all_sequences = str(row.get("all_sequences", ""))

        concat_seq, stoi, chain_to_seq = get_chain_sequences(all_sequences, stoichiometry)
        if concat_seq:
            sequence = concat_seq
        L = len(sequence)

        # Labels
        coords, resnames, resids = self._get_labels_for_target(target_id)
        N = coords.shape[0]
        if N != L:
            coords = coords[:L]
            resids = resids[:L]
        mask = np.ones(L, dtype=np.float32)

        # Optional crop
        if self.crop_len and L > self.crop_len[1]:
            crop_size = self.rng.integers(self.crop_len[0], min(self.crop_len[1], L) + 1)
            start = self.rng.integers(0, L - crop_size + 1)
            end = start + crop_size
            sequence = sequence[start:end]
            coords = coords[start:end]
            resids = resids[start:end]
            mask = mask[start:end]
            L = crop_size

        # Encode sequence
        seq_ids = encode_sequence(sequence, max_len=self.max_seq_len)
        L_act = min(L, self.max_seq_len)
        if L_act < self.max_seq_len:
            mask[L_act:] = 0

        # MSA
        msa_ids = np.zeros((self.msa_max_depth, self.max_seq_len), dtype=np.int64) + 4  # pad
        msa_mask = np.zeros((self.msa_max_depth, self.max_seq_len), dtype=np.float32)
        pair_covar = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.float32)

        if self.msa_dir:
            msa_path = self.msa_dir / f"{target_id}.MSA.fasta"
            if msa_path.exists():
                feats = get_msa_features(msa_path, n_max=self.msa_max_depth, compute_covar=True, seed=self.rng.integers(0, 2**31))
                if feats["msa_array"] is not None:
                    S, Lm = feats["msa_array"].shape
                    msa_ids[:S, :min(Lm, self.max_seq_len)] = feats["msa_array"][:, : self.max_seq_len]
                    msa_mask[:S, :min(Lm, self.max_seq_len)] = 1.0
                if feats["covariation"] is not None:
                    Lc = min(feats["covariation"].shape[0], self.max_seq_len)
                    pair_covar[:Lc, :Lc] = feats["covariation"][:Lc, :Lc]

        # Pair features: covariation + relative position bin
        rel_pos = np.abs(np.arange(self.max_seq_len)[:, None] - np.arange(self.max_seq_len)[None, :])
        rel_pos = np.clip(rel_pos, 0, 31).astype(np.float32) / 31.0
        pair_features = np.stack([pair_covar, rel_pos], axis=-1)

        # Base pair label from 3D
        if self.include_base_pair_label:
            bp_map = self._derive_base_pair_map(coords)
            if L_act < self.max_seq_len:
                bp_pad = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.float32)
                bp_pad[:L_act, :L_act] = bp_map
                bp_map = bp_pad
        else:
            bp_map = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.float32)

        # Distance map
        if self.include_dist_map:
            d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            if L_act < self.max_seq_len:
                d_pad = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.float32)
                d_pad[:L_act, :L_act] = d
                d = d_pad
        else:
            d = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.float32)

        # Pad coords to max_seq_len
        coords_pad = np.zeros((self.max_seq_len, 3), dtype=np.float32)
        coords_pad[:L_act] = coords[:L_act]
        mask_pad = np.zeros(self.max_seq_len, dtype=np.float32)
        mask_pad[:L_act] = mask[:L_act]

        return {
            "seq_ids": torch.from_numpy(seq_ids),
            "msa_ids": torch.from_numpy(msa_ids),
            "msa_mask": torch.from_numpy(msa_mask),
            "pair_features": torch.from_numpy(pair_features),
            "coord": torch.from_numpy(coords_pad),
            "mask": torch.from_numpy(mask_pad),
            "base_pair_map": torch.from_numpy(bp_map),
            "dist_map": torch.from_numpy(d),
            "length": L_act,
            "target_id": target_id,
        }
