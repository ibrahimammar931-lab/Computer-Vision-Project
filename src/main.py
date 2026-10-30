"""Command-line entry point for project pipeline stages."""
import argparse

from src.pipeline import run_detection, run_teams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Football offside video pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("detect", "teams"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--camera", default="cam_01")
        sub.add_argument("--video", help="Override the configured camera video path")
        sub.add_argument("--output", help="Override output JSON path")
        sub.add_argument("--config-dir", help="Directory containing project.yaml, detection.yaml, cameras.yaml")
        if command == "detect":
            sub.add_argument("--model", help="Override detection.model_path")
            sub.add_argument("--device", help="Ultralytics device, e.g. cpu or 0")
        else:
            sub.add_argument("--detections", help="Detection JSON path; defaults to the configured output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {"camera_id": args.camera, "video_path": args.video, "output_path": args.output, "config_dir": args.config_dir}
    if args.command == "detect":
        run_detection(**common, model_path=args.model, device=args.device)
    else:
        run_teams(**common, detections_path=args.detections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
