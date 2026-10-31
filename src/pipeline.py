"""Application-level orchestration for detection and shirt-color labeling."""
from pathlib import Path
from typing import Any

from src.config import load_configs, resolve_project_path
from src.detection import detect_video, load_model, save_detections
from src.team_classification import smooth_team_labels
from src.tracking import save_tracks, track_video
from src.utils.io import load_json


def _camera_settings(cameras_config: dict[str, Any], camera_id: str) -> dict[str, Any]:
    cameras = cameras_config.get("cameras", cameras_config)
    camera = cameras.get(camera_id, {}) if isinstance(cameras, dict) else {}
    if not camera and cameras_config.get("video_path"):
        camera = cameras_config
    return camera if isinstance(camera, dict) else {}


def _required_video_path(camera: dict[str, Any], override: str | Path | None) -> Path:
    value = override or camera.get("video_path")
    if not value:
        raise ValueError("Provide --video or set video_path for the selected camera in configs/cameras.yaml")
    path = resolve_project_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    return path


def run_detection(
    *,
    camera_id: str = "cam_01",
    video_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config_dir: str | Path | None = None,
    model_path: str | None = None,
    device: str | None = None,
) -> str:
    configs = load_configs(config_dir)
    project = configs["project"]
    detection = configs["detection"]
    camera = _camera_settings(configs["cameras"], camera_id)
    video = _required_video_path(camera, video_path)
    weights = model_path or detection.get("model_path", "yolov8n.pt")
    model = load_model(str(weights), device=device or detection.get("device"))
    result = detect_video(
        model,
        video,
        camera_id=camera_id,
        player_conf=float(detection.get("player_confidence", detection.get("player_conf", 0.25))),
        ball_conf=float(detection.get("ball_confidence", detection.get("ball_conf", 0.10))),
        player_class_id=int(detection.get("player_class_id", 0)),
        ball_class_id=int(detection.get("ball_class_id", 32)),
        imgsz=int(detection.get("imgsz", 960)),
        device=device or detection.get("device"),
        frame_step=int(detection.get("frame_step", 1)),
    )
    default_output = project.get("detection_output", f"outputs/detections/detections_{camera_id}.json")
    destination = resolve_project_path(output_path or default_output)
    saved = save_detections(result, destination)
    print(f"Saved detections: {saved}")
    return saved


def run_teams(
    *,
    camera_id: str = "cam_01",
    video_path: str | Path | None = None,
    detections_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> str:
    configs = load_configs(config_dir)
    project = configs["project"]
    camera = _camera_settings(configs["cameras"], camera_id)
    video = _required_video_path(camera, video_path)
    default_detections = project.get("detection_output", f"outputs/detections/detections_{camera_id}.json")
    detections_file = resolve_project_path(detections_path or default_detections)
    data = load_json(detections_file)
    color_ranges = project.get("team_colors", project.get("team_color_ranges", {}))
    if not color_ranges:
        raise ValueError("Set team_colors in configs/project.yaml before running the teams command")
    default_output = project.get("team_labels_output", f"outputs/teams/team_labels_{camera_id}.json")
    destination = resolve_project_path(output_path or default_output)
    smooth_team_labels(data, color_ranges=color_ranges, video_path=video, output_path=destination)
    print(f"Saved team labels: {destination}")
    return str(destination)


def run_tracking(
    *,
    camera_id: str = "cam_01",
    video_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config_dir: str | Path | None = None,
    model_path: str | None = None,
    device: str | None = None,
) -> str:
    """Run YOLO detection with ByteTrack and save a JSON track file."""
    configs = load_configs(config_dir)
    project = configs["project"]
    detection = configs["detection"]
    tracking = configs.get("tracking", {})
    camera = _camera_settings(configs["cameras"], camera_id)
    video = _required_video_path(camera, video_path)
    weights = model_path or detection.get("model_path", "yolov8n.pt")
    selected_device = device or detection.get("device")
    model = load_model(str(weights), device=selected_device)
    result = track_video(
        model,
        video,
        camera_id=camera_id,
        player_conf=float(detection.get("player_confidence", detection.get("player_conf", 0.25))),
        ball_conf=float(detection.get("ball_confidence", detection.get("ball_conf", 0.10))),
        player_class_id=int(detection.get("player_class_id", 0)),
        ball_class_id=int(detection.get("ball_class_id", 32)),
        imgsz=int(detection.get("imgsz", 960)),
        tracker=str(tracking.get("tracker", "bytetrack.yaml")),
        device=selected_device,
    )
    default_output = project.get("tracking_output", f"outputs/tracking/tracks_{camera_id}.json")
    destination = resolve_project_path(output_path or default_output)
    saved = save_tracks(result, destination)
    print(f"Saved tracks: {saved}")
    return saved
