# 3drna_cs – Mechanism-driven RNA 3D structure prediction

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
   - 将本仓库打包为 `3drna_cs.zip`（或从 GitHub 克隆），并上传到 Colab。

2. **打开 Colab**  
   - 用 Colab 打开 `notebooks/colab_run_all.ipynb`（上传 zip 后解压，或在 Colab 里从 Drive 打开该 ipynb）。

3. **按顺序执行**  
   - **第 1 节**：安装依赖（`kaggle`, `pyyaml` 等；Colab 已带 PyTorch）。  
   - **第 2 节**：把 `kaggle.json` 上传到 `/content/`（左侧文件栏上传到 Colab 根目录），然后运行该单元格完成 API 配置。  
   - **第 3 节**：若已上传 `3drna_cs.zip` 到 `/content/`，会解压并设置项目路径；否则需保证当前环境里已有完整代码（如从 Drive 打开则已是项目内）。  
   - **第 4 节**：运行 `scripts/download_kaggle_data.py --light`，自动下载比赛 CSV（轻量模式，不下载 MSA/PDB_RNA）。  
   - **第 5 节**：运行训练 `src/train.py --data_dir /content/stanford-rna-3d-folding-2`。  
   - 可选：挂载 Google Drive，将 `checkpoints/` 复制到 Drive 备份。

4. **数据说明**  
   - `--light`：只下 CSV，训练可跑但无 MSA 特征（适合先跑通）。  
   - 需要 MSA 时：去掉 `--light` 重新下载（或从 Kaggle 网页手动下载 MSA 并放到 `DATA_DIR/MSA/`）。

## License

MIT.
