"""Load project YAML settings and resolve paths relative to the repository."""
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_configs(config_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load pipeline settings; empty YAML files become empty mappings."""
    root = Path(config_dir) if config_dir is not None else CONFIG_DIR
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return {
        name: _read_yaml(root / f"{name}.yaml")
        for name in ("project", "detection", "cameras", "tracking")
    }


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root, not the shell cwd."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
