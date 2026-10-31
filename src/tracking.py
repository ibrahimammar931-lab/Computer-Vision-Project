"""Multi-object tracking with Ultralytics YOLO and ByteTrack."""
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from src.utils.io import save_json
from src.video_io import iter_frames, load_video_metadata


def track_frame(
    model: YOLO,
    frame,
    *,
    player_conf: float = 0.25,
    ball_conf: float = 0.10,
    player_class_id: int = 0,
    ball_class_id: int = 32,
    imgsz: int = 960,
    tracker: str = "bytetrack.yaml",
    device: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Detect and assign ByteTrack IDs in one frame.

    ``persist=True`` is important: it keeps the ByteTrack state from the
    previous call, so use one model instance for a whole video.
    """
    if not 0 <= player_conf <= 1 or not 0 <= ball_conf <= 1:
        raise ValueError("Confidence thresholds must be between 0 and 1")

    kwargs: dict[str, Any] = {
        "persist": True,
        "tracker": tracker,
        "conf": min(player_conf, ball_conf),
        "imgsz": imgsz,
        "classes": [player_class_id, ball_class_id],
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    result = model.track(frame, **kwargs)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes
    track_ids = (
        [int(track_id) for track_id in boxes.id.cpu().tolist()]
        if boxes.id is not None
        else [None] * len(boxes)
    )
    detections = []
    for box, confidence, class_id, track_id in zip(
        boxes.xyxy, boxes.conf, boxes.cls, track_ids
    ):
        class_id = int(class_id.item())
        confidence = float(confidence.item())
        threshold = player_conf if class_id == player_class_id else ball_conf
        if confidence < threshold:
            continue
        label = result.names.get(class_id, str(class_id)) if isinstance(result.names, dict) else result.names[class_id]
        detections.append({
            "class_id": class_id,
            "label": str(label),
            "confidence": confidence,
            "bbox_xyxy": [float(value) for value in box.tolist()],
            "track_id": track_id,
        })
    return detections


def track_video(
    model: YOLO,
    video_path: str | Path,
    *,
    camera_id: str = "cam_01",
    player_conf: float = 0.25,
    ball_conf: float = 0.10,
    player_class_id: int = 0,
    ball_class_id: int = 32,
    imgsz: int = 960,
    tracker: str = "bytetrack.yaml",
    device: Optional[str] = None,
    frame_step: int = 1,
) -> Dict[str, Any]:
    """Track players and the ball across a video, returning JSON-ready data."""
    if frame_step != 1:
        raise ValueError("ByteTrack needs consecutive frames, so tracking frame_step must be 1")

    metadata = load_video_metadata(video_path)
    frames = []
    for frame_index, frame in iter_frames(video_path, step=1):
        frames.append({
            "frame_index": frame_index,
            "detections": track_frame(
                model,
                frame,
                player_conf=player_conf,
                ball_conf=ball_conf,
                player_class_id=player_class_id,
                ball_class_id=ball_class_id,
                imgsz=imgsz,
                tracker=tracker,
                device=device,
            ),
        })
    return {
        "camera_id": camera_id,
        "video_path": str(video_path),
        "metadata": metadata,
        "tracker": tracker,
        "frames": frames,
    }


def save_tracks(
    tracks: Dict[str, Any],
    path: str | Path = "outputs/tracking/tracks_cam_01.json",
) -> str:
    """Save per-frame tracking results, including ByteTrack IDs, as JSON."""
    return save_json(tracks, path)
