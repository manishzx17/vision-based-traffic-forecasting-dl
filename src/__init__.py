"""Traffic Flow Prediction using YOLOv8, ByteTrack, and Recurrent Neural Networks (LSTM/GRU).

This package provides modular, reusable components for:
- Vehicle detection (YOLOv8)
- Multi-object vehicle tracking & line counting (ByteTrack)
- Spatial occupancy and traffic feature preprocessing
- Chronological time-series dataset windowing
- Deep sequence models (Compact LSTM & GRU) and statistical baselines
- Multi-horizon evaluation, error analysis, and conformal uncertainty quantification
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

__version__ = "0.1.0"


def get_project_root() -> Path:
    """Return the absolute Path to the project root directory.

    Resolves accurately regardless of whether invoked from root, notebooks/, or src/.
    """
    current_file = Path(__file__).resolve()
    # current_file is at <project_root>/src/__init__.py -> parent.parent is <project_root>
    return current_file.parent.parent


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducible execution.

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the centralized YAML configuration file.

    Args:
        config_path: Optional path to config file. If None, defaults to configs/config.yaml
                     relative to the project root.

    Returns:
        Dictionary containing project configurations.
    """
    root = get_project_root()
    if config_path is None:
        target_path = root / "configs" / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
