"""Multi-horizon forecast evaluation, residual diagnostics, and conformal uncertainty quantification."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def compute_horizon_metrics(
    y_true: Union[np.ndarray, List[List[float]]],
    y_pred: Union[np.ndarray, List[List[float]]],
    split_name: str = "Test",
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute overall and horizon-specific (H1-H5) MAE, RMSE, and MSE.

    Args:
        y_true: Ground truth target array of shape (N_windows, H).
        y_pred: Predicted target array of shape (N_windows, H).
        split_name: Partition name ('Train', 'Validation', 'Test').
        model_name: Optional model identifier string.

    Returns:
        Dictionary with Overall and H1-H5 MAE and RMSE metrics.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    n_windows, h_dim = yt.shape
    errors = yp - yt
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    overall_mae = float(np.mean(abs_errors))
    overall_rmse = float(np.sqrt(np.mean(sq_errors)))
    overall_mse = float(np.mean(sq_errors))

    metrics: Dict[str, Any] = {
        "Split": split_name,
        "Eval_Windows": n_windows,
        "Horizon_Steps": h_dim,
        "Overall_MAE": overall_mae,
        "Overall_RMSE": overall_rmse,
        "Overall_MSE": overall_mse,
    }

    if model_name:
        metrics["Model"] = model_name

    for h in range(h_dim):
        h_idx = h + 1
        h_abs = abs_errors[:, h]
        h_sq = sq_errors[:, h]
        metrics[f"H{h_idx}_MAE"] = float(np.mean(h_abs))
        metrics[f"H{h_idx}_RMSE"] = float(np.sqrt(np.mean(h_sq)))
        metrics[f"H{h_idx}_MSE"] = float(np.mean(h_sq))

    # Horizon degradation / drift percentage from H1 to H5
    if h_dim >= 2 and metrics["H1_MAE"] > 0:
        metrics["H1_to_H5_MAE_Drift_Pct"] = round(
            ((metrics[f"H{h_dim}_MAE"] - metrics["H1_MAE"]) / metrics["H1_MAE"]) * 100.0,
            2,
        )

    return metrics


def compute_residuals(
    y_true: Union[np.ndarray, List[List[float]]],
    y_pred: Union[np.ndarray, List[List[float]]],
) -> np.ndarray:
    """Compute signed forecasting residuals e = y_true - y_pred.

    Args:
        y_true: Ground truth target array (N, H).
        y_pred: Model predictions array (N, H).

    Returns:
        Array of residuals with shape (N, H).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return yt - yp


def calibrate_conformal_intervals(
    val_residuals: np.ndarray,
    method: str = "max",
    quantile: float = 0.90,
) -> np.ndarray:
    """Calibrate horizon-specific nonconformity quantiles from validation absolute residuals.

    Args:
        val_residuals: Validation residuals of shape (N_val, H). Can be signed or absolute.
        method: 'max' for finite-sample conformal ceiling (maximum validation residual),
                or 'quantile' for sample percentile.
        quantile: Percentile level if method == 'quantile' (default: 0.90).

    Returns:
        1D array of calibration quantiles of shape (H,).
    """
    abs_res = np.abs(np.asarray(val_residuals, dtype=float))

    if method == "max":
        # Finite-sample ceiling quantile per horizon
        q_vec = np.max(abs_res, axis=0)
    elif method == "quantile":
        q_vec = np.quantile(abs_res, quantile, axis=0)
    else:
        raise ValueError(f"Unknown calibration method: {method}. Choose 'max' or 'quantile'.")

    return q_vec


def compute_prediction_intervals(
    y_pred: np.ndarray,
    calibration_quantiles: np.ndarray,
    clip_min: Optional[float] = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct lower and upper prediction intervals around point predictions.

    Args:
        y_pred: Point forecasts of shape (N, H).
        calibration_quantiles: Horizon calibration offsets of shape (H,).
        clip_min: Minimum physical value threshold (e.g. 0.0 for non-negative vehicle counts).

    Returns:
        Tuple of (lower_bounds, upper_bounds), each of shape (N, H).
    """
    yp = np.asarray(y_pred, dtype=float)
    q = np.asarray(calibration_quantiles, dtype=float)

    lower = yp - q
    upper = yp + q

    if clip_min is not None:
        lower = np.maximum(lower, clip_min)

    return lower, upper


def evaluate_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    split_name: str = "Test",
    calibration_quantiles: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate empirical coverage and interval widths for prediction intervals across horizons.

    Args:
        y_true: Ground truth target array (N, H).
        y_pred: Point forecasts (N, H).
        lower_bounds: Lower interval bounds (N, H).
        upper_bounds: Upper interval bounds (N, H).
        split_name: Partition name ('Validation' or 'Test').
        calibration_quantiles: Optional 1D array of calibration quantiles.

    Returns:
        DataFrame summarizing coverage, interval width, MAE, and RMSE per horizon and overall.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    lb = np.asarray(lower_bounds, dtype=float)
    ub = np.asarray(upper_bounds, dtype=float)

    n_windows, h_dim = yt.shape
    records = []

    for h in range(h_dim):
        y_h = yt[:, h]
        y_hat_h = yp[:, h]
        l_h = lb[:, h]
        u_h = ub[:, h]

        covered = (y_h >= l_h) & (y_h <= u_h)
        coverage = float(np.mean(covered))
        aiw = float(np.mean(u_h - l_h))
        mae = float(np.mean(np.abs(y_h - y_hat_h)))
        rmse = float(np.sqrt(np.mean((y_h - y_hat_h) ** 2)))

        rec: Dict[str, Any] = {
            "Split": split_name,
            "Horizon": f"H{h + 1}",
            "Empirical_Coverage": coverage,
            "Average_Interval_Width": aiw,
            "MAE": mae,
            "RMSE": rmse,
            "Covered_Points": int(np.sum(covered)),
            "Total_Points": n_windows,
        }
        if calibration_quantiles is not None:
            rec["Calibration_Quantile"] = float(calibration_quantiles[h])

        records.append(rec)

    # Add overall aggregate row
    overall_covered = (yt >= lb) & (yt <= ub)
    overall_coverage = float(np.mean(overall_covered))
    overall_aiw = float(np.mean(ub - lb))
    overall_mae = float(np.mean(np.abs(yt - yp)))
    overall_rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    overall_rec: Dict[str, Any] = {
        "Split": split_name,
        "Horizon": "Overall",
        "Empirical_Coverage": overall_coverage,
        "Average_Interval_Width": overall_aiw,
        "MAE": overall_mae,
        "RMSE": overall_rmse,
        "Covered_Points": int(np.sum(overall_covered)),
        "Total_Points": n_windows * h_dim,
    }
    if calibration_quantiles is not None:
        overall_rec["Calibration_Quantile"] = float(np.mean(calibration_quantiles))

    records.append(overall_rec)

    return pd.DataFrame(records)
