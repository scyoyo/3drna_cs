"""
Unified prediction pipeline: load model(s), run on test sequences, produce submission.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.featurizer import encode_sequence
from src.data.msa_processing import get_msa_features
from src.data.stoichiometry import get_chain_sequences
from src.models.model import RNA3DModel
from src.ensemble import select_diverse_top5
from src.pretrained.rhofold_wrapper import run_rhofold_multi_seed


def predict_single(
    model: torch.nn.Module,
    sequence: str,
    msa_path: Optional[Path] = None,
    pair_features: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cuda"),
    num_recycling: int = 3,
    max_len: int = 512,
) -> np.ndarray:
    """Predict C1' coords (L, 3) for one sequence."""
    model.eval()
    L = len(sequence)
    seq_ids = encode_sequence(sequence, max_len=max_len)
    seq_t = torch.from_numpy(seq_ids).long().unsqueeze(0).to(device)
    if pair_features is None:
        pair_features = torch.zeros(1, max_len, max_len, 2, device=device)
    else:
        pair_features = pair_features.unsqueeze(0).to(device)
    msa_ids = seq_t.unsqueeze(0).expand(1, 1, -1)
    msa_mask = torch.ones(1, 1, max_len, device=device)
    mask = torch.zeros(1, max_len, device=device)
    mask[0, :L] = 1

    with torch.no_grad():
        out = model(
            msa_ids=msa_ids,
            pair_features=pair_features,
            msa_mask=msa_mask,
            pair_mask=mask,
            num_recycling=num_recycling,
        )
    coord = out["coord"][0, :L].cpu().numpy()
    return coord


def build_submission_row(
    target_id: str,
    resnames: List[str],
    resids: List[int],
    coords_list: List[np.ndarray],
) -> Dict:
    """One row per residue: ID, resname, resid, x_1,y_1,z_1,...,x_5,y_5,z_5."""
    rows = []
    L = len(resnames)
    for i in range(L):
        r = {"ID": f"{target_id}_{i+1}", "resname": resnames[i], "resid": resids[i]}
        for k, coord in enumerate(coords_list[:5]):
            x, y, z = float(coord[i, 0]), float(coord[i, 1]), float(coord[i, 2])
            x = max(-999.999, min(9999.999, x))
            y = max(-999.999, min(9999.999, y))
            z = max(-999.999, min(9999.999, z))
            r[f"x_{k+1}"], r[f"y_{k+1}"], r[f"z_{k+1}"] = x, y, z
        rows.append(r)
    return rows


def run_pipeline(
    test_csv: Path,
    submission_csv: Path,
    model_path: Optional[Path] = None,
    msa_dir: Optional[Path] = None,
    device: str = "cuda",
    use_rhofold: bool = False,
) -> None:
    """
    Read test_sequences.csv, predict 5 structures per target, write submission.csv.
    """
    df = pd.read_csv(test_csv)
    all_rows = []
    for _, row in df.iterrows():
        target_id = str(row["target_id"])
        sequence = str(row["sequence"]).upper().replace("T", "U")
        stoichiometry = str(row.get("stoichiometry", "A:1"))
        all_sequences = str(row.get("all_sequences", ""))
        concat_seq, _, _ = get_chain_sequences(all_sequences, stoichiometry)
        if concat_seq:
            sequence = concat_seq
        L = len(sequence)
        resnames = list(sequence)
        resids = list(range(1, L + 1))

        candidates = []
        if model_path and Path(model_path).exists():
            model = RNA3DModel()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            pair_feat = None
            if msa_dir:
                msa_path = msa_dir / f"{target_id}.MSA.fasta"
                if msa_path.exists():
                    feats = get_msa_features(msa_path, n_max=128, compute_covar=True)
                    if feats.get("covariation") is not None:
                        cov = feats["covariation"]
                        pos = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]).astype(np.float32) / 32.0
                        pair_feat = torch.from_numpy(np.stack([cov[:L, :L], pos], axis=-1)).float()
            for seed in [0, 42, 123]:
                torch.manual_seed(seed)
                coord = predict_single(model, sequence, pair_features=pair_feat, device=torch.device(device))
                candidates.append(coord)
        if use_rhofold:
            rhofold_coords = run_rhofold_multi_seed(sequence, seeds=[0, 42])
            candidates.extend(rhofold_coords)
        if not candidates:
            candidates = [np.zeros((L, 3), dtype=np.float32) for _ in range(5)]
        indices = select_diverse_top5(candidates, n=5)
        selected = [candidates[i] for i in indices]
        rows = build_submission_row(target_id, resnames, resids, selected)
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(submission_csv, index=False)
