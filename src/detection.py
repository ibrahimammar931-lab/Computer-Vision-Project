"""YOLO detection helpers for players and the ball."""
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from src.utils.io import save_json
from src.video_io import iter_frames, load_video_metadata


def load_model(model_path: str, device: Optional[str] = None) -> YOLO:
    """Load an Ultralytics YOLO model from a weights path or model name."""
    model = YOLO(model_path)
    if device is not None:
        model.to(device)
    return model


def detect_frame(
    model: YOLO,
    frame,
    *,
    player_conf: float = 0.25,
    ball_conf: float = 0.10,
    player_class_id: int = 0,
    ball_class_id: int = 32,
    imgsz: int = 960,
    device: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Detect configured player and ball classes in one BGR frame."""
    if not 0 <= player_conf <= 1 or not 0 <= ball_conf <= 1:
        raise ValueError("Confidence thresholds must be between 0 and 1")
    kwargs = {"conf": min(player_conf, ball_conf), "imgsz": imgsz, "classes": [player_class_id, ball_class_id], "verbose": False}
    if device is not None:
        kwargs["device"] = device
    result = model.predict(frame, **kwargs)[0]
    names = result.names
    detections = []
    if result.boxes is None:
        return detections
    for box, confidence, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        class_id = int(class_id.item())
        confidence = float(confidence.item())
        threshold = player_conf if class_id == player_class_id else ball_conf
        if confidence < threshold:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        detections.append({
            "class_id": class_id,
            "label": str(label),
            "confidence": confidence,
            "bbox_xyxy": [x1, y1, x2, y2],
        })
    return detections


def detect_video(
    model: YOLO,
    video_path: str | Path,
    *,
    camera_id: str = "cam_01",
    player_conf: float = 0.25,
    ball_conf: float = 0.10,
    player_class_id: int = 0,
    ball_class_id: int = 32,
    imgsz: int = 960,
    device: Optional[str] = None,
    frame_step: int = 1,
) -> Dict[str, Any]:
    """Detect objects over a video and return per-frame records."""
    metadata = load_video_metadata(video_path)
    frames = []
    for frame_index, frame in iter_frames(video_path, step=frame_step):
        frames.append({
            "frame_index": frame_index,
            "detections": detect_frame(
                model, frame, player_conf=player_conf, ball_conf=ball_conf,
                player_class_id=player_class_id, ball_class_id=ball_class_id,
                imgsz=imgsz, device=device,
            ),
        })
    return {"camera_id": camera_id, "video_path": str(video_path), "metadata": metadata, "frames": frames}


def save_detections(detections: Dict[str, Any], path: str | Path = "outputs/detections/detections_cam_01.json") -> str:
    """Save video-level detection records as JSON."""
    return save_json(detections, path)
