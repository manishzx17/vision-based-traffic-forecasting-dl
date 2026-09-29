"""Recurrent neural network forecasting architectures and baseline forecasting models."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# Statistical Baseline Forecasters
# =====================================================================

class PersistenceForecaster:
    """Naïve persistence baseline: Carries the last observed value forward across horizon H."""

    def __init__(self, horizon: int = 5):
        self.horizon = horizon

    def predict(self, lookback: np.ndarray) -> np.ndarray:
        """Predict horizon steps from 1D lookback array.

        Args:
            lookback: 1D array of shape (L,).

        Returns:
            1D array of shape (H,) carrying lookback[-1].
        """
        vals = np.asarray(lookback)
        last_val = vals[-1]
        return np.full(self.horizon, last_val, dtype=float)


class HistoricalMeanForecaster:
    """Historical mean baseline: Predicts the training set mean across horizon H."""

    def __init__(self, train_mean: float, horizon: int = 5):
        self.train_mean = float(train_mean)
        self.horizon = horizon

    def predict(self, lookback: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict horizon steps using training mean.

        Returns:
            1D array of shape (H,) with train_mean.
        """
        return np.full(self.horizon, self.train_mean, dtype=float)


class MovingAverageForecaster:
    """Simple Moving Average (SMA-K) baseline: Predicts the mean of the most recent K observations."""

    def __init__(self, k: int = 3, horizon: int = 5):
        self.k = k
        self.horizon = horizon

    def predict(self, lookback: np.ndarray) -> np.ndarray:
        """Predict horizon steps using mean of last K observations.

        Args:
            lookback: 1D array of shape (L,).

        Returns:
            1D array of shape (H,) with mean of lookback[-K:].
        """
        vals = np.asarray(lookback)
        k_val = min(self.k, len(vals))
        sma = float(np.mean(vals[-k_val:]))
        return np.full(self.horizon, sma, dtype=float)


# =====================================================================
# Deep Recurrent Neural Networks
# =====================================================================

class CompactTrafficLSTM(nn.Module):
    """Compact single-layer LSTM with dropout and linear projection for multi-step traffic forecasting."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        num_layers: int = 1,
        output_dim: int = 5,
        dropout: float = 0.10,
    ):
        super(CompactTrafficLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Predictions tensor of shape (batch_size, output_dim).
        """
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Extract last recurrent hidden state
        last_step = lstm_out[:, -1, :]
        last_step = self.dropout(last_step)
        predictions = self.fc(last_step)
        return predictions


class CompactTrafficGRU(nn.Module):
    """Compact single-layer GRU with dropout and linear projection for multi-step traffic forecasting."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        num_layers: int = 1,
        output_dim: int = 5,
        dropout: float = 0.10,
    ):
        super(CompactTrafficGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Predictions tensor of shape (batch_size, output_dim).
        """
        # gru_out shape: (batch_size, seq_len, hidden_dim)
        gru_out, _ = self.gru(x)
        # Extract last recurrent hidden state
        last_step = gru_out[:, -1, :]
        last_step = self.dropout(last_step)
        predictions = self.fc(last_step)
        return predictions


# =====================================================================
# Model Training and Checkpointing Utilities
# =====================================================================

def train_forecasting_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    max_epochs: int = 150,
    patience: int = 25,
    save_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a PyTorch sequence model with MSELoss, Adam, and validation early stopping.

    Args:
        model: CompactTrafficLSTM or CompactTrafficGRU instance.
        x_train: Training inputs (N_train, L, D).
        y_train: Training targets (N_train, H).
        x_val: Validation inputs (N_val, L, D).
        y_val: Validation targets (N_val, H).
        lr: Learning rate.
        weight_decay: L2 regularization coefficient.
        max_epochs: Maximum training epochs.
        patience: Validation loss early stopping patience.
        save_path: Optional path to save best model state dict.
        verbose: Whether to log training progress.

    Returns:
        Tuple of (best_model, training_history_dict).
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, Any]] = None

    train_losses = []
    val_losses = []

    for epoch in range(1, max_epochs + 1):
        # Training step
        model.train()
        optimizer.zero_grad()
        preds_train = model(x_train)
        loss = criterion(preds_train, y_train)
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            preds_val = model(x_val)
            val_loss = criterion(preds_val, y_val).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    history = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    return model, history


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
) -> nn.Module:
    """Load state dictionary from a checkpoint path into a model instance.

    Args:
        model: Instantiated PyTorch model.
        checkpoint_path: Path to .pt file.
        device: Device to map weights to ('cpu', 'cuda', etc.).

    Returns:
        Model with loaded weights in eval mode.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    map_loc = torch.device(device) if device else torch.device("cpu")
    state_dict = torch.load(path, map_location=map_loc)
    model.load_state_dict(state_dict)
    model.eval()
    return model
