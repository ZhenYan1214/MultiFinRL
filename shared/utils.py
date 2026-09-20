"""共用工具：載入設定、讀寫 JSON。"""
import hashlib
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


def stable_json_sha256(obj) -> str:
    """以穩定鍵排序計算 JSON 內容指紋，供資料與向量版本相容性檢查。"""
    payload = json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
