"""Small OpenCV helpers for reading and writing video frames."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def _open_video(video_path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {video_path}")
    return cap


def load_video_metadata(video_path: str | Path) -> dict:
    """Return basic video metadata as a JSON-friendly dictionary."""
    cap = _open_video(video_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        return asdict(VideoMetadata(str(video_path), width, height, fps, frame_count, duration))
    finally:
        cap.release()


def read_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Read one zero-based frame; raise IndexError if it is unavailable."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    cap = _open_video(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise IndexError(f"Frame {frame_index} is unavailable in {video_path}")
        return frame
    finally:
        cap.release()


def iter_frames(
    video_path: str | Path,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (zero-based frame index, frame) over [start, stop) at `step`."""
    if start < 0 or step <= 0 or (stop is not None and stop < start):
        raise ValueError("Invalid frame range")
    cap = _open_video(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_index = start
        while stop is None or frame_index < stop:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_index, frame
            if step == 1:
                frame_index += 1
            else:
                next_index = frame_index + step
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_index)
                frame_index = next_index
    finally:
        cap.release()


def save_frame(path: str | Path, frame: np.ndarray) -> str:
    """Save a BGR frame, creating its parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"Could not write image: {path}")
    return str(path)
