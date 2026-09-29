"""Dataset windowing and PyTorch Dataset / DataLoader abstractions for time-series forecasting."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TrafficSequenceDataset(Dataset):
    """PyTorch Dataset wrapper for input sequences X and multi-step targets Y."""

    def __init__(self, x_tensors: Union[torch.Tensor, np.ndarray], y_tensors: Union[torch.Tensor, np.ndarray]):
        """Initialize dataset.

        Args:
            x_tensors: Input tensor of shape (N, L, D).
            y_tensors: Target tensor of shape (N, H).
        """
        if isinstance(x_tensors, np.ndarray):
            self.x = torch.tensor(x_tensors, dtype=torch.float32)
        else:
            self.x = x_tensors.float()

        if isinstance(y_tensors, np.ndarray):
            self.y = torch.tensor(y_tensors, dtype=torch.float32)
        else:
            self.y = y_tensors.float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def create_target_anchored_windows(
    df: pd.DataFrame,
    target_col: str = "total_vehicle_count",
    lookback: int = 10,
    horizon: int = 5,
    n_train: int = 31,
    n_val: int = 7,
    n_test: int = 6,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate target-anchored sliding windows across train, validation, and test partitions.

    Under target-anchored partitioning:
    - Train windows: all target indices fall within [0, n_train - 1].
    - Validation windows: target indices fall within [n_train, n_train + n_val - 1].
    - Test windows: target indices fall within [n_train + n_val, N - 1].

    Args:
        df: Input DataFrame containing the time series.
        target_col: Target column name (default: "total_vehicle_count").
        lookback: Number of historical steps L.
        horizon: Multi-step forecast horizon H.
        n_train: Number of training observations.
        n_val: Number of validation observations.
        n_test: Number of test observations.

    Returns:
        Dict with keys 'train', 'val', 'test', 'all', containing window dictionaries.
    """
    n_total = len(df)
    train_set = set(range(0, n_train))
    val_set = set(range(n_train, n_train + n_val))
    test_set = set(range(n_train + n_val, n_total))

    all_windows = []
    train_windows = []
    val_windows = []
    test_windows = []

    for t in range(lookback - 1, n_total - horizon):
        lookback_idx = list(range(t - lookback + 1, t + 1))
        target_idx = list(range(t + 1, t + horizon + 1))

        item = {
            "origin_t": t,
            "origin_k": t + 1,
            "lookback_idx": lookback_idx,
            "target_idx": target_idx,
            "lookback_k": f"{lookback_idx[0] + 1}..{lookback_idx[-1] + 1}",
            "target_k": f"{target_idx[0] + 1}..{target_idx[-1] + 1}",
            "y_raw": df.loc[target_idx, target_col].values,
        }

        if all(idx in train_set for idx in target_idx):
            item["split"] = "train"
            train_windows.append(item)
        elif all(idx in val_set for idx in target_idx):
            item["split"] = "val"
            val_windows.append(item)
        elif all(idx in test_set for idx in target_idx):
            item["split"] = "test"
            test_windows.append(item)
        else:
            item["split"] = "cross_boundary"

        all_windows.append(item)

    return {
        "train": train_windows,
        "val": val_windows,
        "test": test_windows,
        "all": all_windows,
    }


def create_sequence_tensors(
    df: pd.DataFrame,
    windows_dict: Dict[str, List[Dict[str, Any]]],
    feature_cols: Sequence[str] = ("total_vehicle_count",),
    target_col: str = "total_vehicle_count",
    feature_scaler: Optional[StandardScaler] = None,
    target_scaler: Optional[StandardScaler] = None,
) -> Tuple[Dict[str, torch.Tensor], StandardScaler, StandardScaler]:
    """Convert window lists into scaled PyTorch tensors for model training and evaluation.

    Args:
        df: Input DataFrame.
        windows_dict: Dict with keys 'train', 'val', 'test' from create_target_anchored_windows.
        feature_cols: Sequence of input feature column names.
        target_col: Target column name.
        feature_scaler: Optional pre-fitted feature scaler. If None, fitted on train inputs.
        target_scaler: Optional pre-fitted target scaler. If None, fitted on train targets.

    Returns:
        Tuple of (tensors_dict, fitted_feature_scaler, fitted_target_scaler).
    """
    train_windows = windows_dict["train"]
    val_windows = windows_dict["val"]
    test_windows = windows_dict["test"]

    # Raw training sequences for fitting scalers
    x_train_raw = np.array([df.loc[w["lookback_idx"], list(feature_cols)].values for w in train_windows])
    y_train_raw = np.array([w["y_raw"] for w in train_windows])

    # Fit scalers strictly on training data if not provided
    if feature_scaler is None:
        feature_scaler = StandardScaler()
        # Flatten time and batch dimensions for feature scaler
        b, t, d = x_train_raw.shape
        feature_scaler.fit(x_train_raw.reshape(-1, d))

    if target_scaler is None:
        target_scaler = StandardScaler()
        target_scaler.fit(y_train_raw.reshape(-1, 1))

    d_dim = len(feature_cols)

    def _transform_windows(w_list: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not w_list:
            return torch.empty((0, 10, d_dim)), torch.empty((0, 5))
        x_raw = np.array([df.loc[w["lookback_idx"], list(feature_cols)].values for w in w_list])
        y_raw = np.array([w["y_raw"] for w in w_list])

        b_cur, t_cur, _ = x_raw.shape
        x_scaled = feature_scaler.transform(x_raw.reshape(-1, d_dim)).reshape(b_cur, t_cur, d_dim)
        y_scaled = target_scaler.transform(y_raw.reshape(-1, 1)).reshape(b_cur, -1)

        return torch.tensor(x_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32)

    x_train, y_train = _transform_windows(train_windows)
    x_val, y_val = _transform_windows(val_windows)
    x_test, y_test = _transform_windows(test_windows)

    tensors = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }

    return tensors, feature_scaler, target_scaler


def create_data_loaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int = 4,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation.

    Args:
        x_train: Training inputs tensor (N_train, L, D).
        y_train: Training targets tensor (N_train, H).
        x_val: Validation inputs tensor (N_val, L, D).
        y_val: Validation targets tensor (N_val, H).
        batch_size: Mini-batch size.
        shuffle_train: Whether to shuffle training mini-batches.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = TrafficSequenceDataset(x_train, y_train)
    val_dataset = TrafficSequenceDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
