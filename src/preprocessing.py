"""Roadway spatial occupancy, traffic state feature extraction, and temporal preprocessing."""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

# Roadway Region of Interest (ROI) polygon in 320x240 frame coordinates bounding active lanes
DEFAULT_ROI_POLYGON = np.array([
    [170, 75],   # Top-left (median barrier at upper curve, below distant horizon)
    [270, 75],   # Top-right (outer shoulder at upper curve)
    [315, 235],  # Bottom-right (outer lane / merge area above bottom logo)
    [85, 235],   # Bottom-left (concrete barrier base)
    [125, 140],  # Mid-left contour along concrete median
], dtype=np.int32)


def create_roi_mask(
    frame_shape: Tuple[int, int] = (240, 320),
    roi_polygon: np.ndarray = DEFAULT_ROI_POLYGON,
) -> Tuple[np.ndarray, int]:
    """Create a binary mask for the roadway Region of Interest.

    Args:
        frame_shape: (height, width) of video frames (default: (240, 320)).
        roi_polygon: Polygon vertices defining active travel lanes.

    Returns:
        Tuple of (binary_mask_uint8, total_roi_area_pixels).
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)
    roi_area_px = int(np.sum(mask > 0))
    return mask, roi_area_px


def extract_frame_traffic_state(
    frame: np.ndarray,
    model: YOLO,
    roi_mask: np.ndarray,
    roi_area_px: int,
    conf_thresh: float = 0.25,
    tracker_config: str = "bytetrack.yaml",
) -> Dict[str, Any]:
    """Extract vehicle counts and spatial occupancy ratio for a single frame.

    Occupancy is defined as: Area(Union(B_i) ∩ ROI) / Area(ROI).
    Vehicle counts use the centroid-inside-ROI inclusion rule.

    Args:
        frame: BGR image frame (H, W, 3).
        model: Preloaded YOLO model with tracking enabled.
        roi_mask: Binary mask for the roadway ROI.
        roi_area_px: Total pixel area of the ROI mask.
        conf_thresh: Detection confidence threshold.
        tracker_config: Tracker configuration YAML.

    Returns:
        Dictionary with vehicle counts, class breakdown, and occupancy ratio.
    """
    h, w = frame.shape[:2]
    results = model.track(
        frame,
        tracker=tracker_config,
        classes=[2, 5, 7],  # car (2), bus (5), truck (7)
        conf=conf_thresh,
        persist=True,
        verbose=False,
    )[0]

    vehicle_union_mask = np.zeros((h, w), dtype=np.uint8)
    car_count = 0
    truck_count = 0
    bus_count = 0
    detected_boxes: List[Dict[str, Any]] = []

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()
        confs = results.boxes.conf.cpu().tolist()

        for bbox, cid, tid, conf in zip(boxes, classes, track_ids, confs):
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            box_roi_intersection = roi_mask[y1:y2, x1:x2] > 0
            intersection_area = int(np.sum(box_roi_intersection))

            if intersection_area > 0:
                vehicle_union_mask[y1:y2, x1:x2] = 255

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cx_int, cy_int = int(min(w - 1, max(0, cx))), int(min(h - 1, max(0, cy)))
            centroid_in_roi = bool(roi_mask[cy_int, cx_int] > 0)

            if centroid_in_roi:
                if cid == 2:
                    car_count += 1
                elif cid == 7:
                    truck_count += 1
                elif cid == 5:
                    bus_count += 1

            if intersection_area > 0 or centroid_in_roi:
                cname = model.names[cid] if hasattr(model, "names") else str(cid)
                detected_boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cid,
                    "class_name": cname,
                    "track_id": tid,
                    "conf": conf,
                    "intersection_area": intersection_area,
                    "centroid_in_roi": centroid_in_roi,
                })

    occupied_roi_mask = cv2.bitwise_and(vehicle_union_mask, roi_mask)
    occupied_area_px = int(np.sum(occupied_roi_mask > 0))
    occupancy_ratio = occupied_area_px / roi_area_px if roi_area_px > 0 else 0.0

    return {
        "vehicle_count": car_count + truck_count + bus_count,
        "car_count": car_count,
        "truck_count": truck_count,
        "bus_count": bus_count,
        "occupancy_ratio": occupancy_ratio,
        "occupied_area_px": occupied_area_px,
        "occupied_mask": occupied_roi_mask,
        "detected_boxes": detected_boxes,
    }


def aggregate_clip_features(frame_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-frame traffic states into clip-level summary features.

    Args:
        frame_records: List of frame traffic state dictionaries.

    Returns:
        Dictionary of clip summary metrics (mean/max count, HVR, occupancy).
    """
    if not frame_records:
        return {
            "total_vehicle_count": 0.0,
            "max_vehicle_count": 0.0,
            "car_count": 0.0,
            "truck_count": 0.0,
            "heavy_vehicle_ratio": 0.0,
            "spatial_occupancy_ratio": 0.0,
        }

    counts = [r["vehicle_count"] for r in frame_records]
    cars = [r["car_count"] for r in frame_records]
    trucks = [r["truck_count"] + r.get("bus_count", 0) for r in frame_records]
    occupancies = [r["occupancy_ratio"] for r in frame_records]

    mean_count = float(np.mean(counts))
    max_count = float(np.max(counts))
    mean_cars = float(np.mean(cars))
    mean_trucks = float(np.mean(trucks))
    total_heavy = mean_trucks
    hvr = float(total_heavy / mean_count) if mean_count > 0 else 0.0
    mean_occupancy = float(np.mean(occupancies))

    return {
        "total_vehicle_count": round(mean_count, 4),
        "max_vehicle_count": round(max_count, 4),
        "car_count": round(mean_cars, 4),
        "truck_count": round(mean_trucks, 4),
        "heavy_vehicle_ratio": round(hvr, 4),
        "spatial_occupancy_ratio": round(mean_occupancy, 4),
    }


def temporal_train_val_test_split(
    df: pd.DataFrame,
    n_train: int = 31,
    n_val: int = 7,
    n_test: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform strict chronological train/validation/test partitioning.

    Enforces temporal order: No shuffling, no lookahead leakage.

    Args:
        df: Input DataFrame sorted by chronological time / observation index.
        n_train: Number of training observations (default: 31, 70.45%).
        n_val: Number of validation observations (default: 7, 15.91%).
        n_test: Number of test observations (default: 6, 13.64%).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    total = n_train + n_val + n_test
    if len(df) != total:
        raise ValueError(f"DataFrame length {len(df)} does not match partition sum {total}.")

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def fit_scaler(train_values: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler exclusively on training partition observations.

    Args:
        train_values: 1D or 2D array of training observations.

    Returns:
        Fitted StandardScaler instance.
    """
    vals = np.asarray(train_values)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(vals)
    return scaler


def scale_features(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Transform features using a fitted scaler.

    Args:
        values: 1D or 2D array of values.
        scaler: Fitted StandardScaler.

    Returns:
        Scaled array of identical shape.
    """
    vals = np.asarray(values)
    orig_shape = vals.shape
    if vals.ndim == 1:
        scaled = scaler.transform(vals.reshape(-1, 1)).flatten()
    elif vals.ndim == 2:
        scaled = scaler.transform(vals)
    elif vals.ndim == 3:
        # Batch of sequences: (B, T, D)
        B, T, D = vals.shape
        scaled = scaler.transform(vals.reshape(-1, D)).reshape(B, T, D)
    else:
        raise ValueError(f"Unsupported array dimensionality: {vals.ndim}")
    return scaled


def inverse_scale_features(scaled_values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse transform scaled features back to original scale.

    Args:
        scaled_values: 1D, 2D, or 3D scaled array.
        scaler: Fitted StandardScaler.

    Returns:
        Array in original physical units (vehicles/frame, occupancy, etc.).
    """
    vals = np.asarray(scaled_values)
    if vals.ndim == 1:
        unscaled = scaler.inverse_transform(vals.reshape(-1, 1)).flatten()
    elif vals.ndim == 2:
        unscaled = scaler.inverse_transform(vals)
    elif vals.ndim == 3:
        B, T, D = vals.shape
        unscaled = scaler.inverse_transform(vals.reshape(-1, D)).reshape(B, T, D)
    else:
        raise ValueError(f"Unsupported array dimensionality: {vals.ndim}")
    return unscaled


def compute_transition_magnitude(series: Union[pd.Series, np.ndarray], shift: int = 1) -> np.ndarray:
    """Compute traffic transition magnitude defined as abs(y_t - y_{t-1}).

    Args:
        series: Array or Series of traffic volume observations.
        shift: Lag step for transition difference (default: 1).

    Returns:
        Array of transition magnitudes with 0 prepended for initial index.
    """
    vals = np.asarray(series)
    diffs = np.abs(np.diff(vals, n=shift))
    return np.concatenate([[0.0] * shift, diffs])
