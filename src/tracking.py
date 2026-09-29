"""ByteTrack multi-object vehicle tracking and virtual line-crossing counter."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.detection import DEFAULT_COLORS, DEFAULT_VEHICLE_CLASS_IDS


class TrafficTrackerAndCounter:
    """Per-clip ByteTrack vehicle tracking and virtual line-crossing counter.

    Tracking state restarts independently for every video clip to avoid track identity
    bleeding between non-contiguous surveillance clips.
    """

    def __init__(
        self,
        model: YOLO,
        line_y: int = 160,
        line_x_range: Tuple[int, int] = (100, 310),
        conf_thresh: float = 0.25,
        target_classes: Sequence[int] = DEFAULT_VEHICLE_CLASS_IDS,
        tracker_config: str = "bytetrack.yaml",
    ):
        """Initialize the tracker and line counter.

        Args:
            model: Preloaded Ultralytics YOLO model.
            line_y: Horizontal pixel coordinate of counting gate.
            line_x_range: (min_x, max_x) pixel span of virtual counting line.
            conf_thresh: Confidence threshold for detector.
            target_classes: List of target vehicle class IDs.
            tracker_config: Ultralytics tracker yaml configuration.
        """
        self.model = model
        self.line_y = line_y
        self.line_x_range = line_x_range
        self.conf_thresh = conf_thresh
        self.target_classes = list(target_classes)
        self.tracker_config = tracker_config

        # Per-clip state tracking
        self.prev_centroids: Dict[int, Tuple[float, float]] = {}  # tid -> (cx, cy)
        self.counted_ids: Set[int] = set()                       # unique vehicle IDs counted
        self.counted_events: List[Dict[str, Any]] = []          # crossing event records
        self.frame_records: List[Dict[str, Any]] = []           # per-frame tracking details
        self.unique_track_ids: Set[int] = set()

    def reset_state(self) -> None:
        """Reset internal tracking state between clips."""
        self.prev_centroids.clear()
        self.counted_ids.clear()
        self.counted_events.clear()
        self.frame_records.clear()
        self.unique_track_ids.clear()

    def process_clip(self, frames: Sequence[np.ndarray]) -> Dict[str, Any]:
        """Process all frames of a clip sequentially through ByteTrack.

        Args:
            frames: Sequence of video frames (BGR).

        Returns:
            Dictionary containing aggregated summary statistics for the clip.
        """
        self.reset_state()

        for f_idx, frame in enumerate(frames, start=1):
            res = self.model.track(
                frame,
                tracker=self.tracker_config,
                classes=self.target_classes,
                conf=self.conf_thresh,
                persist=True,
                verbose=False,
            )[0]

            active_frame_boxes = []
            if res.boxes is not None and res.boxes.id is not None:
                tids = res.boxes.id.int().cpu().tolist()
                cids = res.boxes.cls.int().cpu().tolist()
                boxes = res.boxes.xyxy.cpu().numpy()

                for tid, cid, bbox in zip(tids, cids, boxes):
                    self.unique_track_ids.add(tid)
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    cname = self.model.names[cid] if hasattr(self.model, "names") else str(cid)

                    active_frame_boxes.append({
                        "tid": tid,
                        "cid": cid,
                        "cname": cname,
                        "cx": cx,
                        "cy": cy,
                        "bbox": bbox,
                    })

                    # Line crossing test: southbound motion crossing y = line_y
                    if tid in self.prev_centroids:
                        _, prev_cy = self.prev_centroids[tid]

                        # Crossing from south to north in image pixel coordinate space
                        if prev_cy > self.line_y and cy <= self.line_y:
                            if self.line_x_range[0] <= cx <= self.line_x_range[1]:
                                if tid not in self.counted_ids:
                                    self.counted_ids.add(tid)
                                    self.counted_events.append({
                                        "frame_idx": f_idx,
                                        "track_id": tid,
                                        "class_name": cname,
                                        "crossing_x": round(cx, 1),
                                        "crossing_y": round(cy, 1),
                                        "direction": "Southbound (outbound)",
                                    })
                    self.prev_centroids[tid] = (cx, cy)

            self.frame_records.append({
                "frame_idx": f_idx,
                "active_vehicles": len(active_frame_boxes),
                "boxes": active_frame_boxes,
            })

        return self.get_summary(len(frames))

    def get_summary(self, total_frames: int) -> Dict[str, Any]:
        """Return aggregated summary statistics for the processed clip.

        Args:
            total_frames: Number of frames in the clip.

        Returns:
            Dict containing vehicle counts, class breakdown, and density.
        """
        df_events = pd.DataFrame(self.counted_events)
        class_counts = (
            df_events["class_name"].value_counts().to_dict()
            if len(df_events) > 0
            else {}
        )
        mean_active = (
            float(np.mean([fr["active_vehicles"] for fr in self.frame_records]))
            if self.frame_records
            else 0.0
        )

        return {
            "total_frames": total_frames,
            "total_unique_tracks": len(self.unique_track_ids),
            "vehicles_counted_line": len(self.counted_ids),
            "cars_counted": int(class_counts.get("car", 0)),
            "trucks_counted": int(class_counts.get("truck", 0)),
            "buses_counted": int(class_counts.get("bus", 0)),
            "mean_instantaneous_density": round(mean_active, 2),
        }


def render_tracking_frame(
    frame: np.ndarray,
    frame_idx: int,
    df_tracks: pd.DataFrame,
    line_y: Optional[int] = None,
    line_x_range: Optional[Tuple[int, int]] = None,
    trail_history: int = 10,
    palette: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Render tracked vehicle bounding boxes, IDs, and centroid trajectory trails.

    Args:
        frame: BGR frame to render upon.
        frame_idx: Frame index to draw active detections for.
        df_tracks: DataFrame containing columns ['frame_idx', 'track_id', 'class_id', 'class_name', 'cx', 'cy', 'bbox'].
        line_y: Optional y coordinate of virtual counting line.
        line_x_range: Optional (x_min, x_max) span of counting line.
        trail_history: Number of preceding frames to trace centroid breadcrumbs.
        palette: Optional class ID to BGR color mapping.

    Returns:
        Annotated copy of the frame.
    """
    annotated = frame.copy()
    colors = palette or DEFAULT_COLORS

    # Draw optional counting line
    if line_y is not None and line_x_range is not None:
        cv2.line(
            annotated,
            (line_x_range[0], line_y),
            (line_x_range[1], line_y),
            (0, 0, 255),
            2,
        )

    # Historical centroid breadcrumb trails
    start_f = max(1, frame_idx - trail_history)
    df_hist = df_tracks[(df_tracks["frame_idx"] >= start_f) & (df_tracks["frame_idx"] <= frame_idx)]
    for _, group in df_hist.groupby("track_id"):
        pts = group[["cx", "cy"]].values.astype(int)
        for i in range(len(pts) - 1):
            cv2.line(annotated, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 255), 1)

    # Active bounding boxes at current frame
    df_current = df_tracks[df_tracks["frame_idx"] == frame_idx]
    for _, row in df_current.iterrows():
        x1, y1, x2, y2 = map(int, row["bbox"])
        cid = int(row["class_id"])
        tid = row["track_id"]
        cname = row["class_name"]
        color = colors.get(cid, (0, 255, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        label = f"ID:{tid} {cname}"
        cv2.putText(
            annotated,
            label,
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated
