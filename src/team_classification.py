"""HSV shirt-color classification and track-level label smoothing."""
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.utils.io import save_json


def crop_jersey_region(frame, bbox_xyxy: Sequence[float]) -> Optional[np.ndarray]:
    """Crop the approximate torso area from an xyxy player box."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 4 or box_h < 8:
        return None
    left, right = x1 + int(box_w * 0.2), x2 - int(box_w * 0.2)
    top, bottom = y1 + int(box_h * 0.2), y1 + int(box_h * 0.58)
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def estimate_dominant_color(image: np.ndarray) -> Tuple[int, int, int]:
    """Return median HSV color, ignoring low-saturation pixels when possible."""
    if image is None or image.size == 0:
        raise ValueError("Cannot estimate color from an empty image")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    saturated = pixels[pixels[:, 1] >= 35]
    selected = saturated if len(saturated) else pixels
    return tuple(int(value) for value in np.median(selected, axis=0))


def _normalize_ranges(spec: Any) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Accept {lower, upper}, a list of such pairs, or {ranges: [...]} ."""
    if isinstance(spec, Mapping) and "ranges" in spec:
        spec = spec["ranges"]
    if isinstance(spec, Mapping):
        spec = [spec]
    normalized = []
    for item in spec:
        if isinstance(item, Mapping):
            lower, upper = item["lower"], item["upper"]
        else:
            lower, upper = item
        normalized.append((np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)))
    return normalized


def classify_team(
    jersey_crop: np.ndarray,
    color_ranges: Mapping[str, Any],
    *,
    min_fraction: float = 0.08,
) -> Tuple[str, float]:
    """Return (team label, matching HSV pixel fraction); UNKNOWN if weak."""
    if jersey_crop is None or jersey_crop.size == 0 or not color_ranges:
        return "UNKNOWN", 0.0
    hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
    scores = {}
    for team, spec in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in _normalize_ranges(spec):
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        scores[str(team)] = float(np.count_nonzero(mask)) / mask.size
    team, score = max(scores.items(), key=lambda item: item[1])
    return (team, score) if score >= min_fraction else ("UNKNOWN", score)


def smooth_team_labels(
    video_data: Dict[str, Any],
    frame_images: Optional[Mapping[int, np.ndarray]] = None,
    color_ranges: Optional[Mapping[str, Any]] = None,
    *,
    video_path: str | Path | None = None,
    min_fraction: float = 0.08,
    output_path: str | Path = "outputs/teams/team_labels.json",
) -> Dict[str, Any]:
    """Add stable team labels to detections using per-track majority votes.

    `video_data` contains per-frame detections, optionally with tracking IDs.
    Supply `video_path` to stream frames from disk, or `frame_images` for a small
    in-memory clip. The input object is copied so detection data is preserved.
    """
    import copy

    if not color_ranges:
        raise ValueError("color_ranges must define at least one team")
    if video_path is None and frame_images is None:
        raise ValueError("Provide either video_path or frame_images")

    output = copy.deepcopy(video_data)
    votes = defaultdict(Counter)

    def classify_frame(frame_record: dict, frame: Optional[np.ndarray]) -> None:
        for detection in frame_record.get("detections", []):
            if detection.get("label") != "person":
                continue
            track_id = detection.get("track_id")
            if frame is None:
                detection["team"] = "UNKNOWN"
                detection["team_score"] = 0.0
                continue
            crop = crop_jersey_region(frame, detection["bbox_xyxy"])
            team, score = classify_team(crop, color_ranges, min_fraction=min_fraction)
            detection["team_score"] = score
            if track_id is None or team == "UNKNOWN":
                detection["team"] = team
            else:
                votes[track_id][team] += 1
                detection["team"] = votes[track_id].most_common(1)[0][0]

    frame_records = output.get("frames", [])
    if video_path is not None:
        from src.video_io import iter_frames

        record_index = 0
        for frame_index, frame in iter_frames(video_path):
            while (
                record_index < len(frame_records)
                and int(frame_records[record_index]["frame_index"]) < frame_index
            ):
                record_index += 1
            if (
                record_index < len(frame_records)
                and int(frame_records[record_index]["frame_index"]) == frame_index
            ):
                classify_frame(frame_records[record_index], frame)
                record_index += 1
            if record_index >= len(frame_records):
                break
    else:
        for frame_record in frame_records:
            frame_index = int(frame_record["frame_index"])
            classify_frame(frame_record, frame_images.get(frame_index))

    save_json(output, output_path)
    return output
