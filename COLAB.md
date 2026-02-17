# 在 Google Colab 上一键运行

**仓库地址：** https://github.com/scyoyo/3drna_cs  

本仓库可在 **Google Colab** 上直接运行，无需本地环境；Colab 中可直接 **`git clone`** 本 repo，并 **自动下载 Kaggle 比赛数据**。

## 步骤概览

1. 准备 Kaggle API：在 [Kaggle Account](https://www.kaggle.com/settings) 创建 API Token，下载 `kaggle.json`。
2. 打开 Colab：打开 **`notebooks/colab_run_all.ipynb`**（可从 [GitHub](https://github.com/scyoyo/3drna_cs) 打开该 ipynb 后点 “Open in Colab”）。
3. 代码库获取（二选一即可）：
   - **推荐**：运行「上传/解压代码库」单元格，若无 `/content/3drna_cs` 会自动执行 **`git clone https://github.com/scyoyo/3drna_cs /content/3drna_cs`**；
   - 若已克隆过，想拉取最新代码可在 Colab 中执行：`!cd /content/3drna_cs && git pull`；
   - 或上传 **`3drna_cs.zip`** 到 `/content/` 后由该单元格解压。
4. 在 Colab 中 **按顺序执行** `colab_run_all.ipynb` 中的单元格。

## 详细说明

### 1. 安装依赖

运行前两个代码单元格，安装 `kaggle`、`pyyaml`、`pandas`、`numpy`、`scipy`（Colab 已带 PyTorch）。

### 2. 配置 Kaggle API

- **方式 A**：将 **`kaggle.json`** 上传到 Colab 左侧 `/content/`。
- **方式 B（推荐持久化）**：先把 `kaggle.json` 放到 Google Drive 的 **MyDrive 根目录**（即 `/content/drive/MyDrive/kaggle.json`），在 Notebook 里先运行「可选：挂载 Google Drive」单元格，再运行「上传 kaggle.json 并配置 API」；脚本会优先用 `/content/` 下的文件，若没有则从 Drive 读取。

### 3. 代码库路径

- 运行「上传/解压代码库」单元格时，若不存在 `/content/3drna_cs`，会自动 **`git clone https://github.com/scyoyo/3drna_cs /content/3drna_cs`**。
- 若之前已克隆过，需要更新代码时可在新单元格中执行：`!cd /content/3drna_cs && git pull`，再继续后续步骤。
- 也可上传 **`3drna_cs.zip`** 到 `/content/`，该单元格会解压并设置 `ROOT`。

### 4. 下载比赛数据

运行「下载比赛数据」单元格，会执行：

```bash
python scripts/download_kaggle_data.py --data_dir /content/stanford-rna-3d-folding-2 --light
```

- **`--light`**：只下载 CSV（train/validation labels + sequences），不下载 MSA 和 PDB_RNA，适合先跑通流程。
- **数据放在 Google Drive**：可先把数据下载到 Drive（例如 `MyDrive/stanford-rna-3d-folding-2`），挂载 Drive 后把 `DATA_DIR` 改为 `/content/drive/MyDrive/stanford-rna-3d-folding-2` 再运行；脚本检测到该目录下已有 `train_sequences.csv` 和 `train_labels.csv` 时会自动跳过下载。
- 需要 MSA 时：去掉 `--light` 重新下载（或从 Kaggle 网页手动下载 MSA 到 `DATA_DIR/MSA/`）。

### 5. 运行训练

运行「运行训练」单元格：

```bash
python src/train.py --config configs/model.yaml --train_config configs/train.yaml --stage 3d --device cuda --data_dir /content/stanford-rna-3d-folding-2
```

Checkpoint 会保存在当前目录下的 `checkpoints/`。

### 6. 保存到 Google Drive（可选）

运行最后一个单元格，挂载 Drive 并把 `checkpoints/` 复制到 `MyDrive/3drna_cs_checkpoints`。

## 一键脚本（可选）

若已配置好 `kaggle.json` 并安装依赖，可在 Colab 中单单元格执行：

```python
!python scripts/colab_setup_and_run.py --data_dir /content/stanford-rna-3d-folding-2 --light --stage 3d
```

该脚本会依次：下载数据（`--light`）→ 启动训练。

## 常见问题

- **Kaggle 报 403 / 未授权**：确认已接受比赛规则（Kaggle 比赛页 → Join Competition），且 `kaggle.json` 已正确放在 `~/.kaggle/`。
- **找不到 `src`**：确认 Colab 当前工作目录下或 `/content/3drna_cs` 下存在 `src/`（解压 zip 或克隆后应能看到）。
- **显存不足**：在 `configs/train.yaml` 中减小 `batch_size`、`max_seq_len` 或 `crop_len_max`。
