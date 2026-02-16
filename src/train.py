"""
Training script for RNA 3D model. Supports staged training (SS pretrain -> 3D -> E2E).
Run on Colab Pro with A100.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from src.data.dataset import RNAStructureDataset
from src.models.model import RNA3DModel
from src.models.loss import RNA3DLoss


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model(cfg: dict) -> RNA3DModel:
    model_cfg = cfg.get("model", {})
    return RNA3DModel(
        vocab_size=model_cfg.get("nucleotide_vocab_size", 5),
        msa_single_dim=model_cfg.get("msa_single_dim", 256),
        msa_pair_dim=model_cfg.get("msa_pair_dim", 128),
        evoformer_blocks=model_cfg.get("evoformer_num_blocks", 24),
        evoformer_heads=model_cfg.get("evoformer_num_heads", 8),
        structure_layers=model_cfg.get("structure_num_layers", 8),
        structure_heads=model_cfg.get("structure_num_heads", 4),
        num_recycling=model_cfg.get("num_recycling", 3),
        dropout=model_cfg.get("evoformer_dropout", 0.1),
    )


def get_dataloader(cfg: dict, stage: str) -> DataLoader:
    train_cfg = cfg.get("train", {})
    data_dir = Path(train_cfg.get("data_dir", "."))
    return DataLoader(
        RNAStructureDataset(
            sequences_csv=data_dir / train_cfg.get("train_sequences", "train_sequences.csv"),
            labels_csv=data_dir / train_cfg.get("train_labels", "train_labels.csv"),
            msa_dir=data_dir / train_cfg.get("msa_dir", "MSA") if (data_dir / train_cfg.get("msa_dir", "MSA")).exists() else None,
            pdb_rna_dir=data_dir / train_cfg.get("pdb_rna_dir", "PDB_RNA") if (data_dir / train_cfg.get("pdb_rna_dir", "PDB_RNA")).exists() else None,
            max_seq_len=train_cfg.get("max_seq_len", 512),
            msa_max_depth=min(128, train_cfg.get("msa_max_depth", 256)),
            crop_len=(train_cfg.get("crop_len_min", 128), train_cfg.get("crop_len_max", 384)) if stage != "ss" else None,
            include_base_pair_label=True,
            include_dist_map=True,
        ),
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )


def train_stage(cfg: dict, stage: str, device: torch.device):
    model = get_model(cfg).to(device)
    train_cfg = cfg.get("train", {})
    loss_weights = {
        "w_fape": train_cfg.get("w_fape", 1.0),
        "w_dist": train_cfg.get("w_dist", 0.3),
        "w_ss": train_cfg.get("w_ss", 0.5),
        "w_violation": train_cfg.get("w_violation", 0.02),
        "w_clash": train_cfg.get("w_clash", 0.1),
        "w_chain_break": train_cfg.get("w_chain_break", 0.05),
        "w_pairing_geom": train_cfg.get("w_pairing_geom", 0.2),
    }
    criterion = RNA3DLoss(**loss_weights)

    if stage == "ss":
        for p in model.structure_module.parameters():
            p.requires_grad = False
        lr = train_cfg.get("stage1_lr", 1e-3)
        epochs = train_cfg.get("stage1_epochs", 20)
    elif stage == "3d":
        for p in model.evoformer.parameters():
            p.requires_grad = True
        freeze_layers = train_cfg.get("stage2_freeze_evoformer_layers", 12)
        for i, block in enumerate(model.evoformer.blocks):
            if i < freeze_layers:
                for p in block.parameters():
                    p.requires_grad = False
        lr = train_cfg.get("stage2_lr", 5e-4)
        epochs = train_cfg.get("stage2_epochs", 30)
    else:
        lr = train_cfg.get("stage3_lr", 1e-4)
        epochs = train_cfg.get("stage3_epochs", 10)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    loader = get_dataloader(cfg, stage)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            msa_ids = batch["msa_ids"].to(device)
            pair_features = batch["pair_features"].to(device)
            coord = batch["coord"].to(device)
            mask = batch["mask"].to(device)
            base_pair_map = batch["base_pair_map"].to(device)
            msa_mask = batch["msa_mask"].to(device)

            B, S, L = msa_ids.shape
            pair_mask = mask
            pair_features = pair_features[:, :L, :L, :]

            out = model(
                msa_ids=msa_ids,
                pair_features=pair_features,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                num_recycling=3,
            )

            losses = criterion(
                pred_coord=out["coord"],
                pred_frames=out["frames"],
                target_coord=coord,
                mask=mask,
                pred_ss=out["ss_logits"],
                target_bp=base_pair_map[:, :L, :L],
            )
            loss = losses["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"Stage {stage} Epoch {epoch+1}/{epochs} loss={total_loss/max(n_batches,1):.4f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml")
    parser.add_argument("--stage", type=str, choices=["ss", "3d", "e2e"], default="3d")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data_dir (e.g. /content/stanford-rna-3d-folding-2)")
    args = parser.parse_args()

    cfg = {}
    if Path(args.config).exists():
        cfg["model"] = load_config(args.config)
    else:
        cfg["model"] = {}
    if Path(args.train_config).exists():
        cfg["train"] = load_config(args.train_config)
    else:
        cfg["train"] = {}
    if args.data_dir is not None:
        cfg.setdefault("train", {})["data_dir"] = args.data_dir

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = train_stage(cfg, args.stage, device)
    out_path = Path("checkpoints") / f"model_{args.stage}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
