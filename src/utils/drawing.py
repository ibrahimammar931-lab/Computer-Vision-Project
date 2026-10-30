"""OpenCV drawing helpers for reviewing detection and team labels."""
from typing import Any, Dict, Iterable, Mapping, Sequence

import cv2
import numpy as np


DEFAULT_COLORS = {
    "TEAM_A": (40, 200, 40),
    "TEAM_B": (220, 80, 40),
    "GOALKEEPER": (180, 60, 200),
    "REFEREE": (0, 180, 255),
    "UNKNOWN": (180, 180, 180),
    "BALL": (0, 230, 255),
}


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Mapping[str, Any]],
    *,
    team_colors: Mapping[str, Sequence[int]] | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw xyxy boxes and labels; ball detections are highlighted yellow."""
    output = frame.copy() if copy else frame
    colors = dict(DEFAULT_COLORS)
    if team_colors:
        colors.update({key: tuple(int(v) for v in value) for key, value in team_colors.items()})
    for detection in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox_xyxy"]]
        label = str(detection.get("label", "object"))
        is_ball = label.lower() in {"sports ball", "ball"} or detection.get("class_id") == 32
        team = "BALL" if is_ball else str(detection.get("team", "UNKNOWN")).upper()
        color = colors.get(team, colors.get("UNKNOWN", (180, 180, 180)))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        track_id = detection.get("track_id")
        confidence = detection.get("confidence")
        text = f"{team} {('#' + str(track_id)) if track_id is not None else ''}"
        if confidence is not None:
            text += f" {float(confidence):.2f}"
        text_y = max(18, y1 - 6)
        cv2.putText(output, text.strip(), (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return output


def draw_ball(frame: np.ndarray, detection: Mapping[str, Any], *, color=(0, 230, 255)) -> np.ndarray:
    """Draw one ball box and center point."""
    output = frame.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in detection["bbox_xyxy"]]
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    cv2.circle(output, center, max(3, min(x2 - x1, y2 - y1) // 2), color, 2)
    return output
