# TFBind

Tools to encode transcription factor binding site modifications and benchmark multiple regressors on functional group representations.

## Quick Start

Clone the repo:
```bash
git clone <your-fork-or-this-repo-url>
cd TFBind
```

Create conda environment:

```bash
conda create -n tfbind python=3.12
conda activate tfbind
```

### Local Package Installation

This installs the `src/` package and exposes TFbind utilities. Also, automatically install dependencies from `requirements.txt`. With the environment active:

```bash
pip install -e .
```



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

The project currently benchmarks three regression approaches. All workflows assume that dataset files for the desired transcription factor exist under `datasets/`.

Implemented regressor models:

| No. | Regressor Name |
|-----|----------------|
| 1 | **Linear Regression Baseline** | 
| 2 | **XGBoost Regressor** | 
| 3 | **Convolutional Neural Network** | 

Interact with the models available in the `notebooks/` subfolder:

1. Open `notebooks/01_linear.ipynb`.
2. Set the `TF` variable near the top to the transcription factor you wish to model (default is `MITF`).
3. Run the notebook.
4. Generated plots appear in `plots/` subfolder.

### Zero-Shot Prediction of chemical modifications

This notebook extends the linear regression baseline for zero-shot prediction on chemical modifications.


