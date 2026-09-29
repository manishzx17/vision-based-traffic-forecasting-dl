"""YOLOv8 vehicle detection module for traffic surveillance video frames."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

# Standard COCO vehicle class mappings
DEFAULT_VEHICLE_CLASS_IDS = [2, 5, 7]  # 2: car, 5: bus, 7: truck
DEFAULT_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEFAULT_COLORS = {
    2: (46, 204, 113),   # Car: Emerald Green
    3: (52, 152, 219),   # Motorcycle: Blue
    5: (241, 196, 15),   # Bus: Sun Yellow
    7: (231, 76, 60),    # Truck: Coral Red
}


def load_yolo_model(weights_path: str = "yolov8n.pt", device: Optional[str] = None) -> YOLO:
    """Load a pretrained YOLOv8 model checkpoint.

    Args:
        weights_path: Path to YOLO weights (default: "yolov8n.pt").
        device: Device specification ('cpu', 'cuda', 'mps', or None for auto).

    Returns:
        Loaded Ultralytics YOLO model.
    """
    model = YOLO(weights_path)
    if device is not None:
        model.to(device)
    return model


def filter_vehicle_boxes(
    results: Any,
    target_class_ids: Sequence[int] = DEFAULT_VEHICLE_CLASS_IDS,
    conf_thresh: float = 0.25,
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Extract and filter vehicle bounding boxes from YOLO results.

    Args:
        results: Prediction result object from Ultralytics YOLO model.
        target_class_ids: List/tuple of COCO class IDs to retain.
        conf_thresh: Minimum detection confidence threshold.
        class_names: Optional mapping from class_id to name. If None, uses results.names or default.

    Returns:
        List of dictionaries containing class_id, class_name, confidence, and bbox [x1, y1, x2, y2].
    """
    if results is None or not hasattr(results, "boxes") or len(results.boxes) == 0:
        return []

    names = class_names or getattr(results, "names", DEFAULT_CLASS_NAMES)
    filtered = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        if cls_id in target_class_ids and conf >= conf_thresh:
            xyxy = box.xyxy[0].cpu().numpy()
            cname = names.get(cls_id, str(cls_id))
            filtered.append({
                "class_id": cls_id,
                "class_name": cname,
                "confidence": conf,
                "bbox": xyxy,  # [x1, y1, x2, y2]
            })

    return filtered


def detect_vehicles_in_frame(
    model: YOLO,
    frame: np.ndarray,
    conf_thresh: float = 0.25,
    target_class_ids: Sequence[int] = DEFAULT_VEHICLE_CLASS_IDS,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run YOLO inference on a single frame and return filtered vehicle detections.

    Args:
        model: Loaded YOLO model.
        frame: BGR image frame (H, W, 3).
        conf_thresh: Confidence threshold.
        target_class_ids: Class IDs to retain.
        verbose: Whether YOLO should print inference details.

    Returns:
        List of filtered vehicle detections.
    """
    results = model(frame, verbose=verbose)[0]
    return filter_vehicle_boxes(
        results=results,
        target_class_ids=target_class_ids,
        conf_thresh=conf_thresh,
    )


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Draw bounding boxes and class/confidence labels on an image frame.

    Args:
        image: BGR frame to annotate.
        detections: List of detection dictionaries.
        colors: Optional class ID to BGR color mapping.

    Returns:
        Annotated copy of the frame.
    """
    palette = colors or DEFAULT_COLORS
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cid = det["class_id"]
        conf = det["confidence"]
        cname = det["class_name"]
        color = palette.get(cid, (0, 255, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        label = f"{cname} {conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated
