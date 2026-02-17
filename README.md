# 3drna_cs – Mechanism-driven RNA 3D structure prediction

**Repo:** https://github.com/scyoyo/3drna_cs  

Solution for [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2) (Kaggle).

## Design

- **Hierarchical folding**: Sequence → MSA/coevolution → secondary structure → 3D coordinates (IPA) → multi-chain assembly.
- **Data**: Uses `train_sequences.csv`, `train_labels.csv`, `MSA/*.MSA.fasta`, and optional `PDB_RNA/*.cif`.
- **Training**: Colab Pro (A100) in stages: SS pretrain → 3D structure → E2E + multichain.
- **Inference**: Kaggle notebook with multi-model ensemble and diversity-based top-5 selection.

## Setup

```bash
pip install -r requirements.txt
```

## Project layout

- `configs/` – model and training YAML configs.
- `src/data/` – Dataset, MSA processing, featurizer, stoichiometry.
- `src/models/` – Evoformer, secondary structure, structure module (IPA), motif, multichain, loss.
- `src/pretrained/` – RhoFold+ and RibonanzaNet wrappers.
- `notebooks/` – EDA, Colab training, Kaggle submission.
- `utils/` – CIF parser, geometry, TM-score, postprocess.

## Quick start

1. Download competition data (train_sequences, train_labels, MSA, PDB_RNA as needed).
2. Run EDA: `notebooks/eda.ipynb`.
3. Train on Colab Pro: `notebooks/train_colab.ipynb`.
4. Submit: run `notebooks/submission.ipynb` on Kaggle.

## 在 Google Colab 上一键运行（无需本地跑）

1. **准备**  
   - 在 [Kaggle Account](https://www.kaggle.com/settings) 里 **Create New API Token**，下载得到 `kaggle.json`。  
   - 代码库：Colab 中可直接 **`git clone https://github.com/scyoyo/3drna_cs`**，无需本地打包上传。

2. **打开 Colab**  
   - 新建 Notebook 或打开 `notebooks/colab_run_all.ipynb`（可从 [GitHub 打开该 ipynb](https://github.com/scyoyo/3drna_cs/blob/main/notebooks/colab_run_all.ipynb) 后点 “Open in Colab”）。

3. **按顺序执行**  
   - **第 1 节**：安装依赖（`kaggle`, `pyyaml` 等；Colab 已带 PyTorch）。  
   - **第 2 节**：配置 Kaggle API：把 `kaggle.json` 上传到 `/content/`，或放到 **Google Drive 的 MyDrive 根目录** 后先挂载 Drive，再运行该单元格（会从 `/content/` 或 Drive 读取）。  
   - **第 3 节**：若无 `/content/3drna_cs`，会自动 **`git clone https://github.com/scyoyo/3drna_cs /content/3drna_cs`**；若已克隆过想更新，可在该目录执行 **`git pull`**。也可上传 `3drna_cs.zip` 到 `/content/` 后解压。  
   - **第 4 节**：运行 `scripts/download_kaggle_data.py --light` 下载比赛 CSV；若希望**数据持久化**，可挂载 Drive 后把数据目录设到 Drive（如 `/content/drive/MyDrive/stanford-rna-3d-folding-2`），已有 CSV 时会自动跳过下载。  
   - **第 5 节**：运行训练 `src/train.py --data_dir <你的 DATA_DIR>`。  
   - 可选：挂载 Google Drive，将 `checkpoints/` 复制到 Drive 备份。

4. **数据说明**  
   - `--light`：只下 CSV，训练可跑但无 MSA 特征（适合先跑通）。  
   - 需要 MSA 时：去掉 `--light` 重新下载（或从 Kaggle 网页手动下载 MSA 并放到 `DATA_DIR/MSA/`）。

## License

MIT.
