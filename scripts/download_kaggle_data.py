"""
Download Stanford RNA 3D Folding Part 2 competition data via Kaggle API.
Usage: python scripts/download_kaggle_data.py [--data_dir DIR] [--light]
--light: only download CSV files (no MSA, no PDB_RNA) for quick Colab runs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


COMPETITION = "stanford-rna-3d-folding-2"
CSV_FILES = [
    "train_sequences.csv",
    "train_labels.csv",
    "validation_sequences.csv",
    "validation_labels.csv",
    "sample_submission.csv",
    "test_sequences.csv",
]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def main():
    parser = argparse.ArgumentParser(description="Download competition data with Kaggle API")
    parser.add_argument("--data_dir", type=str, default="/content/stanford-rna-3d-folding-2", help="Target directory")
    parser.add_argument("--light", action="store_true", help="Only download CSV files (no MSA/PDB_RNA)")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cwd = data_dir

    try:
        run(["kaggle", "competitions", "list", "-c", COMPETITION])
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Kaggle API not found or not logged in. Install and configure:")
        print("  pip install kaggle")
        print("  Place kaggle.json in ~/.kaggle/ (from Kaggle Account -> Create New API Token)")
        sys.exit(1)

    if args.light:
        for f in CSV_FILES:
            try:
                run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(cwd), "-f", f])
            except subprocess.CalledProcessError:
                print(f"Skip (may not exist): {f}")
        zips = list(cwd.glob("*.zip"))
        for z in zips:
            run(["unzip", "-o", str(z), "-d", str(cwd)], check=False)
            try:
                z.unlink()
            except Exception:
                pass
        if not zips:
            print("No zips to unzip (files may be direct).")
        print("Light download done. MSA/ and PDB_RNA/ are not downloaded.")
        return

    run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(cwd)])
    zips = list(cwd.glob("*.zip"))
    for z in zips:
        run(["unzip", "-o", str(z), "-d", str(cwd)])
        z.unlink()
    print("Full download done. To free space you can remove PDB_RNA/: rm -rf", cwd / "PDB_RNA")


if __name__ == "__main__":
    main()
