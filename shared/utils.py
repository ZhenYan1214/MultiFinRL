"""共用工具：載入設定、讀寫 JSON。"""
import json
from pathlib import Path

import yaml

from . import paths


def load_config() -> dict:
    with open(paths.CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
