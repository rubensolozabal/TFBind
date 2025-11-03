# TFBind

Tools to encode transcription factor binding site modifications and benchmark multiple regressors on functional group representations.

## Quick Start

```bash
git clone <your-fork-or-this-repo-url>
cd TFBind
```

### Create the `tfbind` Conda Environment (Python 3.12)

```bash
conda create -n tfbind python=3.12
conda activate tfbind
```

Install PyTorch inside the environment (pick the variant that matches your hardware; check [pytorch.org](https://pytorch.org/get-started/locally/) for other combinations):

```bash
# CPU-only wheel
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# or, for CUDA 12.1
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Local Package Installation

With the environment active:

```bash
pip install -e .
```

This installs the `src/` package, exposes utilities to the notebooks, and pulls in the base dependencies from `requirements.txt`. Optional extras used in individual notebooks (for example, `xgboost`) can be installed on demand via `pip install xgboost`.

## Repository Layout

```
TFBind/
├── data/                 # Optional raw inputs (git-ignored by default)
├── datasets/             # Encoded CSVs per transcription factor (required for notebooks)
├── notebooks/            # Exploratory notebooks and training scripts
│   ├── 01_xgboost.ipynb
│   ├── 02_linear.ipynb
│   ├── 03_cnn.py
├── plots/                # Generated figures saved by notebooks and scripts
├── saved_models/         # Checkpoints exported by the CNN training script
├── src/                  # Core encoding logic and neural network definitions
│   ├── constants.py
│   ├── encode.py
│   ├── models.py
│   └── utils.py
├── requirements.txt
└── setup.py
```

### Datasets

Each dataset resides at `datasets/<TF>/dataset_<TF>_encoded.csv` and includes:

- `Sequence` columns describing the plus/minus strands.
- `Groove_major` and `Groove_minor` functional group encodings (lists or strings per position).
- `ln(I)` as the regression target.
- `Change` describing chemical modifications; `src.utils.categorize_change` derives categories used in plots.

## Working With the Models

The project currently benchmarks three regression approaches. All workflows assume the environment is activated (`conda activate tfbind`) and that dataset files for the desired transcription factor exist under `datasets/`.

### 1. Linear Regression Baseline (`notebooks/02_linear.ipynb`)

1. Launch Jupyter: `jupyter lab` (or `jupyter notebook`).
2. Open `notebooks/02_linear.ipynb`.
3. Set the `TF` variable near the top to the transcription factor you wish to model (default is `MITF`).
4. Run the notebook top-to-bottom. It:
   - Loads the encoded dataset, performs one-hot encoding with `src.encode.one_hot_encode_grooves`.
   - Flattens the functional group tensors.
   - Fits a scikit-learn `LinearRegression`, reporting train/test metrics and visualisations.
5. Generated plots (per-category error bars, scatter) appear inline and can be exported from the notebook UI if desired.

### 2. XGBoost Regressor (`notebooks/01_xgboost.ipynb`)

1. Install XGBoost if it is not already present: `pip install xgboost`.
2. Open the notebook in Jupyter.
3. Set the `TF` constant (default `ETS1`) to pick the dataset.
4. Execute all cells. The notebook:
   - Reuses the same encoding pipeline as the linear baseline.
   - Reshapes tensors into flat feature vectors.
   - Trains an `xgboost.XGBRegressor`, reporting cross-validation, MAE/MSE, R², and category-wise diagnostics.
5. Saved artefacts (if any) will be written to `plots/`.

### 3. Convolutional Neural Network (`notebooks/03_train_cnn.py` and `03_eval_cnn.py`)

The CNN workflow is scripted so it can run headless.

**Train**

```bash
python notebooks/03_train_cnn.py
```

- Edit the `TF` constant at the top of `03_train_cnn.py` to target a specific transcription factor (default `EGR1`).
- The script builds PyTorch data loaders from the encoded CSV, trains `src.models.CNN_network`, and saves weights to `saved_models/03_cnn_<TF>.pt`.
- Training/validation loss is printed; if matplotlib is available, a learning curve is displayed at the end.

**Evaluate**

```bash
python notebooks/03_eval_cnn.py --model saved_models/03_cnn_EGR1.pt
```

- The evaluator mirrors the train/test split, loads the saved checkpoint, and reports MAE/MSE/R².
- Scatter plots and per-category MAE bar charts are written to `plots/` (override paths with `--scatter-output` and `--errorbar-output`). Add `--show` to display figures interactively.

### Zero-Shot Variant (`notebooks/02_linear_zeroshot_predict_modifications.ipynb`)

This notebook extends the linear regression baseline for zero-shot prediction on chemical modifications. Follow the same steps as the standard linear notebook; it automatically trains on unmodified (`Category == "none"`) samples and evaluates on modified categories.

## Next Steps

- Add new datasets under `datasets/<TF>/` following the existing naming scheme, then update the `TF` constants in notebooks/scripts.
- Extend `src/models.py` with alternative neural architectures and reuse the data loaders in `03_train_cnn.py`.
- Use the plots exported to `plots/` to compare baselines across transcription factors or modification classes.
