#!/usr/bin/env python3
"""Evaluate the ETS1 CNN model on the held-out test split and plot results."""

from __future__ import annotations

import argparse
import sys
from ast import literal_eval
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.constants import ETS1_LEN, fg_encode_map, COLOR_MODS

DEFAULT_DATASET = Path("datasets/ETS1/dataset_ETS1_encoded.csv")
DEFAULT_MODEL_HINTS = (
    Path("models/02_cnn_ets1.pt"),
)



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained ETS1 CNN model and visualise test predictions."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"CSV with encoded ETS1 data (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the trained CNN weights (auto-detected if omitted).",
    )
    parser.add_argument(
        "--scatter-output",
        type=Path,
        default=Path("plots/ETS1_cnn_test_scatter.png"),
        help="Where to save the scatter plot (default: plots/ETS1_cnn_test_scatter.png).",
    )
    parser.add_argument(
        "--errorbar-output",
        type=Path,
        default=Path("plots/ETS1_cnn_category_mae.png"),
        help="Where to save the per-category MAE bar chart (default: plots/ETS1_cnn_category_mae.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction used for the test split (default: 0.3).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used for the train/test split (default: 0).",
    )
    return parser.parse_args()


from src.encode import groove_stack_to_tensor


def parse_groove_column(value: Iterable) -> list:
    if isinstance(value, list):
        return value
    return literal_eval(value)


def encode_sequences(df: pd.DataFrame) -> np.ndarray:
    tensors = []
    for _, row in df.iterrows():
        major = parse_groove_column(row["Groove_major"])
        minor = parse_groove_column(row["Groove_minor"])
        tensors.append(groove_stack_to_tensor(major, minor, length=ETS1_LEN))
    return np.stack(tensors, axis=0)


def resolve_model_path(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"Model file {explicit} does not exist.")
    for candidate in DEFAULT_MODEL_HINTS:
        if candidate.exists():
            return candidate
    torch_candidates = sorted(
        path
        for path in Path("models").glob("*")
        if path.suffix in {".pt", ".pth", ".bin"}
        and "cnn" in path.stem.lower()
    )
    if torch_candidates:
        return torch_candidates[0]
    raise FileNotFoundError(
        "Could not locate saved CNN weights. "
        "Run notebooks/03_cnn_ETS1.py and save the trained model to the models/ directory."
    )

def load_state_dict(model_path: Path, device: torch.device) -> dict:
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise RuntimeError(
        "Unexpected checkpoint format. Expected a state_dict or a mapping containing 'model'."
    )


def plot_predictions(
    df_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, r2: float
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    for category, group in df_test.groupby("Category"):
        ax.scatter(
            group["ln(I)"],
            group["y_pred"],
            s=18,
            marker=".",
            alpha=0.85,
            label=category,
            c=COLOR_MODS.get(category),
        )
    ax.set_xlabel("Experimental binding affinity", fontsize=12)
    ax.set_ylabel("Predicted binding affinity", fontsize=12)
    p1 = max(float(y_pred.max()), float(y_true.max()))
    p2 = min(float(y_pred.min()), float(y_true.min()))
    ax.set_xlim([p2, p1])
    ax.set_ylim([p2, p1])
    ax.plot([p2, p1], [p2, p1], "k-")
    ax.annotate(r"$R^2$ = {:.3f}".format(r2), (p2, p1), fontsize=11)
    ax.legend(loc="lower right", title="Category")
    ax.set_title("ETS1 (Test dataset)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_category_mae(df_test: pd.DataFrame) -> plt.Figure:
    stats = df_test.groupby("Category").apply(
        lambda d: pd.Series(
            {
                "MAE": np.mean(np.abs(d["y_pred"] - d["ln(I)"])),
                "MAE_std": np.std(np.abs(d["y_pred"] - d["ln(I)"])),
            }
        )
    )
    order = [cat for cat in COLOR_MODS if cat in stats.index]
    stats = stats.reindex(order)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        range(len(order)),
        stats["MAE"].values,
        yerr=stats["MAE_std"].values,
        color=[COLOR_MODS[cat] for cat in order],
        capsize=3,
        alpha=0.9,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error by Category")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def ensure_parent(path: Path) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


from src.models import CNN_network as Net
from src.utils import categorize_change

def main() -> None:
    args = parse_arguments()
    dataset_path = args.dataset
    if not dataset_path.exists():
        sys.exit(f"Dataset file {dataset_path} does not exist.")

    df = pd.read_csv(dataset_path)
    if "Category" not in df.columns:
        df["Category"] = df["Change"].apply(categorize_change)
    tensors = encode_sequences(df).astype(np.float32)
    df["encoded_tensor"] = list(tensors)

    _, df_test = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state
    )
    X_test = np.stack(df_test["encoded_tensor"].to_numpy(), axis=0)
    y_test = df_test["ln(I)"].to_numpy()

    model_path = resolve_model_path(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    state_dict = load_state_dict(model_path, device)
    net.load_state_dict(state_dict)
    net.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(X_test).to(device=device, dtype=torch.float32)
        y_pred = net(inputs).cpu().numpy().squeeze(axis=1)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Loaded model:", model_path)
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R^2: {r2:.4f}")

    df_test = df_test.copy()
    df_test["y_pred"] = y_pred

    scatter_fig = plot_predictions(df_test, y_test, y_pred, r2)
    ensure_parent(args.scatter_output)
    scatter_fig.savefig(args.scatter_output, dpi=300, bbox_inches="tight")

    mae_fig = plot_category_mae(df_test)
    ensure_parent(args.errorbar_output)
    mae_fig.savefig(args.errorbar_output, dpi=300, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(scatter_fig)
        plt.close(mae_fig)


if __name__ == "__main__":
    main()
