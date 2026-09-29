"""Smoke test suite to verify refactored src/ package, configurations, models, and outputs."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
import torch


def test_imports():
    """Verify that all src modules and core functions/classes can be imported cleanly."""
    import src
    from src import get_project_root, load_config, set_seed
    from src.detection import (
        DEFAULT_VEHICLE_CLASS_IDS,
        detect_vehicles_in_frame,
        draw_detections,
        filter_vehicle_boxes,
        load_yolo_model,
    )
    from src.tracking import TrafficTrackerAndCounter, render_tracking_frame
    from src.preprocessing import (
        DEFAULT_ROI_POLYGON,
        aggregate_clip_features,
        compute_transition_magnitude,
        create_roi_mask,
        extract_frame_traffic_state,
        fit_scaler,
        inverse_scale_features,
        scale_features,
        temporal_train_val_test_split,
    )
    from src.datasets import (
        TrafficSequenceDataset,
        create_data_loaders,
        create_sequence_tensors,
        create_target_anchored_windows,
    )
    from src.models import (
        CompactTrafficGRU,
        CompactTrafficLSTM,
        HistoricalMeanForecaster,
        MovingAverageForecaster,
        PersistenceForecaster,
        load_model_checkpoint,
        train_forecasting_model,
    )
    from src.evaluation import (
        calibrate_conformal_intervals,
        compute_horizon_metrics,
        compute_prediction_intervals,
        compute_residuals,
        evaluate_prediction_intervals,
    )

    assert src.__version__ == "0.1.0"


def test_config_loading():
    """Verify that centralized config.yaml loads properly and contains required sections."""
    from src import load_config
    cfg = load_config()

    assert "project" in cfg
    assert "data" in cfg
    assert "split" in cfg
    assert "forecasting" in cfg
    assert "models" in cfg
    assert "training" in cfg
    assert cfg["split"]["n_train"] == 31
    assert cfg["split"]["n_val"] == 7
    assert cfg["split"]["n_test"] == 6
    assert cfg["split"]["val_eval_windows"] == 3
    assert cfg["split"]["test_eval_windows"] == 2


def test_reproducibility_seed():
    """Verify that set_seed ensures deterministic random number generation."""
    from src import set_seed

    set_seed(42)
    a1 = np.random.rand(5)
    t1 = torch.randn(5)

    set_seed(42)
    a2 = np.random.rand(5)
    t2 = torch.randn(5)

    np.testing.assert_array_equal(a1, a2)
    assert torch.equal(t1, t2)


def test_dataset_windowing():
    """Verify that target-anchored windowing produces exact expected window counts."""
    from src import get_project_root, load_config
    from src.datasets import create_target_anchored_windows

    root = get_project_root()
    csv_path = root / "outputs" / "tables" / "traffic_timeseries.csv"
    assert csv_path.exists(), f"Missing dataset: {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 44

    # Baseline L=10
    w10 = create_target_anchored_windows(df, lookback=10, horizon=5)
    assert len(w10["train"]) == 17, f"Expected 17 train windows for L=10, got {len(w10['train'])}"
    assert len(w10["val"]) == 3, f"Expected 3 val windows, got {len(w10['val'])}"
    assert len(w10["test"]) == 2, f"Expected 2 test windows, got {len(w10['test'])}"

    # Robustness L=5
    w5 = create_target_anchored_windows(df, lookback=5, horizon=5)
    assert len(w5["train"]) == 22, f"Expected 22 train windows for L=5, got {len(w5['train'])}"
    assert len(w5["val"]) == 3, f"Expected 3 val windows for L=5, got {len(w5['val'])}"
    assert len(w5["test"]) == 2, f"Expected 2 test windows for L=5, got {len(w5['test'])}"

    # Robustness L=20
    w20 = create_target_anchored_windows(df, lookback=20, horizon=5)
    assert len(w20["train"]) == 7, f"Expected 7 train windows for L=20, got {len(w20['train'])}"
    assert len(w20["val"]) == 3, f"Expected 3 val windows for L=20, got {len(w20['val'])}"
    assert len(w20["test"]) == 2, f"Expected 2 test windows for L=20, got {len(w20['test'])}"


def test_models_forward_and_checkpoints():
    """Verify that Compact LSTM and GRU models forward pass and load checkpoints cleanly."""
    from src import get_project_root
    from src.models import (
        CompactTrafficGRU,
        CompactTrafficLSTM,
        PersistenceForecaster,
        MovingAverageForecaster,
        load_model_checkpoint,
    )

    root = get_project_root()
    models_dir = root / "models"

    # Forward pass on dummy batch
    dummy_x = torch.randn(2, 10, 1)
    lstm = CompactTrafficLSTM(input_dim=1, hidden_dim=16, output_dim=5)
    gru = CompactTrafficGRU(input_dim=1, hidden_dim=16, output_dim=5)

    assert lstm(dummy_x).shape == (2, 5)
    assert gru(dummy_x).shape == (2, 5)

    # Checkpoint loading
    checkpoints = [
        ("gru_best_model.pt", CompactTrafficGRU(1, 16, 1, 5)),
        ("lstm_best_model.pt", CompactTrafficLSTM(1, 16, 1, 5)),
        ("gru_lookback_l5.pt", CompactTrafficGRU(1, 16, 1, 5)),
        ("gru_lookback_l20.pt", CompactTrafficGRU(1, 16, 1, 5)),
        ("ablation_config_a.pt", CompactTrafficLSTM(1, 16, 1, 5)),
        ("ablation_config_b.pt", CompactTrafficLSTM(2, 16, 1, 5)),
        ("ablation_config_c.pt", CompactTrafficLSTM(2, 16, 1, 5)),
        ("ablation_config_d.pt", CompactTrafficLSTM(3, 16, 1, 5)),
    ]

    for ckpt_name, model_inst in checkpoints:
        ckpt_path = models_dir / ckpt_name
        assert ckpt_path.exists(), f"Checkpoint {ckpt_name} missing from models/"
        loaded_model = load_model_checkpoint(model_inst, ckpt_path)
        d = model_inst.input_dim
        out = loaded_model(torch.randn(1, 10, d))
        assert out.shape == (1, 5), f"Failed forward pass for {ckpt_name}"

    # Baselines
    pers = PersistenceForecaster(horizon=5)
    sma = MovingAverageForecaster(k=3, horizon=5)
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(pers.predict(arr), np.full(5, 4.0))
    np.testing.assert_array_equal(sma.predict(arr), np.full(5, 3.0))


def test_evaluation_and_intervals():
    """Verify multi-horizon evaluation and conformal intervals calculation."""
    from src.evaluation import (
        calibrate_conformal_intervals,
        compute_horizon_metrics,
        compute_prediction_intervals,
        compute_residuals,
        evaluate_prediction_intervals,
    )

    y_true = np.array([[2.0, 3.0, 4.0, 5.0, 6.0], [3.0, 4.0, 5.0, 6.0, 7.0]])
    y_pred = np.array([[2.5, 2.8, 4.2, 4.9, 5.5], [3.2, 3.9, 5.1, 6.2, 6.8]])

    metrics = compute_horizon_metrics(y_true, y_pred, split_name="Test")
    assert "Overall_MAE" in metrics
    assert "Overall_RMSE" in metrics
    assert "H1_MAE" in metrics
    assert "H5_MAE" in metrics
    assert metrics["Eval_Windows"] == 2

    res = compute_residuals(y_true, y_pred)
    assert res.shape == (2, 5)

    q = calibrate_conformal_intervals(res, method="max")
    assert len(q) == 5

    lb, ub = compute_prediction_intervals(y_pred, q, clip_min=0.0)
    assert lb.shape == (2, 5)
    assert ub.shape == (2, 5)
    assert np.all(lb >= 0.0)

    df_eval = evaluate_prediction_intervals(y_true, y_pred, lb, ub, split_name="Test", calibration_quantiles=q)
    assert len(df_eval) == 6  # H1..H5 + Overall
    assert "Empirical_Coverage" in df_eval.columns


def test_artifacts_exist():
    """Verify that all core tables, predictions, and figures exist and are non-empty."""
    from src import get_project_root
    root = get_project_root()

    tables = [
        "traffic_timeseries.csv",
        "traffic_clip_features.csv",
        "baseline_forecast_metrics.csv",
        "lstm_forecast_metrics.csv",
        "gru_forecast_metrics.csv",
        "feature_ablation_metrics.csv",
        "uncertainty_metrics.csv",
        "transition_performance_metrics.csv",
        "lookback_robustness_metrics.csv",
    ]
    for tbl in tables:
        p = root / "outputs" / "tables" / tbl
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty table: {tbl}"

    preds = [
        "test_forecast_predictions.csv",
        "test_prediction_intervals.csv",
    ]
    for prd in preds:
        p = root / "outputs" / "predictions" / prd
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty prediction: {prd}"

    figs = [
        "traffic_timeseries_overview.png",
        "baseline_forecasting_comparison.png",
        "gru_training_loss_curves.png",
        "gru_vs_lstm_vs_baselines.png",
        "feature_ablation_loss_curves.png",
        "prediction_intervals.png",
        "transition_performance_comparison.png",
        "lookback_mae_comparison.png",
        "lookback_rmse_comparison.png",
    ]
    for fig in figs:
        p = root / "outputs" / "figures" / fig
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty figure: {fig}"


if __name__ == "__main__":
    print("Running smoke tests manually...")
    test_imports()
    print("✓ test_imports passed")
    test_config_loading()
    print("✓ test_config_loading passed")
    test_reproducibility_seed()
    print("✓ test_reproducibility_seed passed")
    test_dataset_windowing()
    print("✓ test_dataset_windowing passed")
    test_models_forward_and_checkpoints()
    print("✓ test_models_forward_and_checkpoints passed")
    test_evaluation_and_intervals()
    print("✓ test_evaluation_and_intervals passed")
    test_artifacts_exist()
    print("✓ test_artifacts_exist passed")
    print("\nALL SMOKE TESTS PASSED CLEANLY!")
