"""JSON helpers shared by pipeline stages."""
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> str:
    """Write JSON atomically and return the output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=indent, ensure_ascii=False, default=_json_default)
        stream.write("\n")
    temporary_path.replace(path)
    return str(path)


def load_json(path: str | Path) -> Any:
    """Load a JSON file using UTF-8."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)
