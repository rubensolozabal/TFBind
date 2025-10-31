import os
import gc
import math
from ast import literal_eval
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import random as ra

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.constants import *

TF = 'EGR1'


def set_global_seed(seed: Optional[int] = None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    ra.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


from src.encode import groove_stack_to_tensor

def parse_groove_column(value: Iterable) -> list[list[str]]:
    """Safely convert stored groove encodings into a list-of-lists structure."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError("Parsed groove column is not a list")
        return parsed
    raise TypeError(f"Unsupported groove column type {type(value)}")


class GrooveDataset(Dataset):
    """Torch dataset wrapping groove encodings."""

    def __init__(self, dataframe: pd.DataFrame):
        expected_columns = {"Groove_major", "Groove_minor", "ln(I)"}
        missing = expected_columns.difference(dataframe.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = dataframe.copy()
        df["Groove_major"] = df["Groove_major"].apply(parse_groove_column)
        df["Groove_minor"] = df["Groove_minor"].apply(parse_groove_column)
        self._targets = df["ln(I)"].astype(np.float32).to_numpy()
        self._major = df["Groove_major"].to_list()
        self._minor = df["Groove_minor"].to_list()

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        major = self._major[idx]
        minor = self._minor[idx]
        features = groove_stack_to_tensor(major, minor)
        target = self._targets[idx]
        return torch.from_numpy(features), torch.tensor([target], dtype=torch.float32)



from src.models import CNN_network

class Model:
    """High-level training/evaluation wrapper around the CNN."""

    def __init__(self, config: Optional[dict] = None, random_state: Optional[int] = None):
        set_global_seed(random_state)

        self.random_state = random_state
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CNN_network().to(self.device)
        # self.num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self._history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._best_state: Optional[dict] = None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        epochs: int = 50,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        max_grad_norm: Optional[float] = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the CNN and optionally evaluate on a validation set."""
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
            )
            if val_loader is not None
            else None
        )
        criterion = nn.MSELoss()

        best_val = math.inf
        bad_epochs = 0
        self._history = {"train_loss": [], "val_loss": []}
        self._best_state = None

        for epoch in range(1, epochs + 1):
            self.net.train()
            train_loss = 0.0
            examples = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                xb, yb = batch
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.net(xb)
                loss = criterion(preds, yb)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
                optimizer.step()

                train_loss += loss.item() * xb.size(0)
                examples += xb.size(0)

            train_loss /= max(examples, 1)
            self._history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self._history["val_loss"].append(val_loss)
                if scheduler is not None:
                    scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    bad_epochs = 0
                    self._best_state = {
                        "model": self.net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
                        break
            elif scheduler is not None:
                scheduler.step(train_loss)

            if verbose:
                if val_loss is not None:
                    print(
                        f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
                        f"val_loss={val_loss:.5f} | lr={optimizer.param_groups[0]['lr']:.3e}"
                    )
                else:
                    print(
                        f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
                        f"lr={optimizer.param_groups[0]['lr']:.3e}"
                    )

        if self._best_state is not None:
            self.net.load_state_dict(self._best_state["model"])
        gc.collect()
        return self._history

    def evaluate(self, loader: DataLoader) -> float:
        """Return average MSE loss on the provided loader."""
        self.net.eval()
        criterion = nn.MSELoss(reduction="sum")
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.net(xb)
                loss = criterion(preds, yb)
                total_loss += loss.item()
                total_examples += xb.size(0)
        return total_loss / max(total_examples, 1)

    def predict(self, loader: DataLoader) -> np.ndarray:
        """Return model predictions for a dataloader."""
        self.net.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    xb = batch[0]
                else:
                    xb = batch
                xb = xb.to(self.device)
                preds = self.net(xb)
                outputs.append(preds.cpu().numpy())
        if not outputs:
            return np.empty((0, 1), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def save(self, path: os.PathLike | str) -> None:
        """Persist the model weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load_model(self, path: os.PathLike | str) -> None:
        """Load model weights from disk."""
        state_dict = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def plot_history(self) -> None:
        """Plot training/validation loss curves."""
        if not self._history["train_loss"]:
            raise RuntimeError("No training history available to plot")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required to plot history") from exc

        plt.figure(figsize=(8, 4))
        plt.plot(self._history["train_loss"], label="Train")
        if self._history["val_loss"]:
            plt.plot(self._history["val_loss"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("CNN Training History")
        plt.legend()
        plt.tight_layout()
        plt.show()


def create_data_loaders(
    csv_path: os.PathLike | str,
    *,
    batch_size: int = 32,
    random_state: Optional[int] = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Build train/validation/test data loaders from the encoded dataset.

    Returns:
        (train_loader, val_loader, test_loader) where val_loader may be None if
        val_ratio is set to 0.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if "Groove_major" not in df or "Groove_minor" not in df:
        raise ValueError("Dataset must contain Groove_major and Groove_minor columns")

    from sklearn.model_selection import train_test_split

    val_ratio = max(0.0, min(val_ratio, 0.5))
    test_ratio = max(0.0, min(test_ratio, 0.5))
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True,
    )

    if val_ratio > 0:
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_ratio / (1.0 - test_ratio),
            random_state=random_state,
            shuffle=True,
        )
    else:
        val_df = train_df.iloc[[]]

    train_dataset = GrooveDataset(train_df)
    val_dataset = GrooveDataset(val_df) if len(val_df) else None
    test_dataset = GrooveDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main() -> None:
    random_state = 42
    set_global_seed(random_state)
    dataset_path = Path(f"datasets/{TF}/dataset_{TF}_encoded.csv")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path,
        batch_size=32,
        random_state=random_state,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    model = Model(random_state=random_state)
    print(f"Training CNN on device {model.device}.")
    history = model.fit(
        train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=3e-3,
        weight_decay=1e-4,
        patience=8,
        max_grad_norm=1.0,
    )

    test_loss = model.evaluate(test_loader)
    print(f"Test MSE loss: {test_loss:.5f}")

    artefact_dir = Path("models")
    artefact_dir.mkdir(exist_ok=True, parents=True)
    weights_path = artefact_dir / f"03_cnn_{TF}.pt"
    model.save(weights_path)
    print(f"Saved model weights to {weights_path}")

    if history["val_loss"]:
        try:
            model.plot_history()
        except RuntimeError:
            pass

    gc.collect()


if __name__ == "__main__":
    main()
