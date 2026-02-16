# 在 Google Colab 上一键运行

本仓库可在 **Google Colab** 上直接运行，无需本地环境，并支持 **自动下载 Kaggle 比赛数据**。

## 步骤概览

1. 准备 Kaggle API：在 [Kaggle Account](https://www.kaggle.com/settings) 创建 API Token，下载 `kaggle.json`。
2. 打开 Colab：用 Colab 打开 **`notebooks/colab_run_all.ipynb`**。
3. 上传代码与配置：
   - 将本仓库打成 **`3drna_cs.zip`** 上传到 Colab 的 `/content/`，或
   - 从 GitHub 克隆到 `/content/3drna_cs`，或
   - 从 Google Drive 打开已同步的该仓库。
4. 在 Colab 中 **按顺序执行** `colab_run_all.ipynb` 中的单元格。

## 详细说明

### 1. 安装依赖

运行前两个代码单元格，安装 `kaggle`、`pyyaml`、`pandas`、`numpy`、`scipy`（Colab 已带 PyTorch）。

### 2. 配置 Kaggle API

- 将下载好的 **`kaggle.json`** 上传到 Colab：左侧「文件」→ 上传到 `/content/`。
- 运行「上传 kaggle.json 并配置 API」单元格，脚本会把 `kaggle.json` 拷贝到 `~/.kaggle/` 并设置权限。

### 3. 代码库路径

- 若已上传 **`3drna_cs.zip`** 到 `/content/`，运行「上传/解压代码库」单元格会自动解压并设置 `ROOT`。
- 否则会尝试当前目录或父目录；若仍找不到 `src/`，请把仓库放到 `/content/3drna_cs` 或按提示上传/克隆。

### 4. 下载比赛数据

运行「下载比赛数据」单元格，会执行：

```bash
python scripts/download_kaggle_data.py --data_dir /content/stanford-rna-3d-folding-2 --light
```

- **`--light`**：只下载 CSV（train/validation labels + sequences），不下载 MSA 和 PDB_RNA，适合先跑通流程。
- 需要 MSA 时：在 `scripts/download_kaggle_data.py` 中去掉 `--light` 后重新运行（或从 Kaggle 网页手动下载 MSA 到 `DATA_DIR/MSA/`）。

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
