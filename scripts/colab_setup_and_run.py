#!/usr/bin/env python3
"""
Colab 一键脚本：检查环境、下载数据（可选）、然后训练。
在 Colab 中可单单元格运行：
  !python scripts/colab_setup_and_run.py --data_dir /content/stanford-rna-3d-folding-2 --light
需要先：1) 安装依赖 pip install kaggle pyyaml  2) 配置好 ~/.kaggle/kaggle.json
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/content/stanford-rna-3d-folding-2")
    parser.add_argument("--light", action="store_true", help="Only download CSV (no MSA/PDB)")
    parser.add_argument("--stage", type=str, default="3d", choices=["ss", "3d", "e2e"])
    parser.add_argument("--skip_download", action="store_true", help="Data already present")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("Downloading competition data...")
        r = subprocess.run(
            [sys.executable, "scripts/download_kaggle_data.py", "--data_dir", str(data_dir)]
            + (["--light"] if args.light else []),
            cwd=root,
        )
        if r.returncode != 0:
            print("Download failed. Check Kaggle API (kaggle.json in ~/.kaggle/).")
            sys.exit(1)

    train_csv = data_dir / "train_sequences.csv"
    labels_csv = data_dir / "train_labels.csv"
    if not train_csv.exists() or not labels_csv.exists():
        print(f"Missing {train_csv} or {labels_csv}. Run without --skip_download or copy data to {data_dir}")
        sys.exit(1)

    print("Starting training...")
    r = subprocess.run(
        [
            sys.executable, "src/train.py",
            "--config", "configs/model.yaml",
            "--train_config", "configs/train.yaml",
            "--stage", args.stage,
            "--device", "cuda",
            "--data_dir", str(data_dir),
        ],
        cwd=root,
    )
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
