"""Train/validation/test 的唯一時間切分協定。

切分採時間順序，並在 train→validation、validation→test 邊界各保留一段 gap，
避免「未來 N 日報酬」標籤跨越資料集邊界。Manifest 儲存明確日期清單，而不只存比例，
如此即使某個下游資料源少了日期，也不會悄悄算出另一套切分。
"""
from __future__ import annotations

import hashlib
from typing import Callable, Iterable, TypeVar

from shared import paths
from shared.utils import read_json, write_json


T = TypeVar("T")
SPLIT_NAMES = ("train", "gap_train_validation", "validation", "gap_validation_test", "test")


def build_temporal_manifest(ticker: str, dates: Iterable[str], *,
                            train_ratio: float = 0.7, validation_ratio: float = 0.15,
                            gap_trading_days: int = 5,
                            metadata: dict | None = None) -> dict:
    ordered = list(dates)
    if ordered != sorted(ordered) or len(ordered) != len(set(ordered)):
        raise ValueError("dates 必須是已排序且不重複的 YYYY-MM-DD 日期")
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("train_ratio 與 validation_ratio 必須介於 0 和 1 之間")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio 必須小於 1")
    if gap_trading_days < 0:
        raise ValueError("gap_trading_days 不可為負數")

    n = len(ordered)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))
    groups = {
        "train": ordered[:train_end],
        "gap_train_validation": ordered[train_end:train_end + gap_trading_days],
        "validation": ordered[train_end + gap_trading_days:validation_end],
        "gap_validation_test": ordered[validation_end:validation_end + gap_trading_days],
        "test": ordered[validation_end + gap_trading_days:],
    }
    if not groups["train"] or not groups["validation"] or not groups["test"]:
        raise ValueError(
            f"切分後有空集合：N={n}, train={len(groups['train'])}, "
            f"validation={len(groups['validation'])}, test={len(groups['test'])}, "
            f"gap={gap_trading_days}"
        )

    return {
        "protocol": "strict_temporal_v1",
        "ticker": ticker,
        "n_dates": n,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "gap_trading_days": gap_trading_days,
        "dates_sha256": hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest(),
        "dates": groups,
        "counts": {name: len(values) for name, values in groups.items()},
        "periods": {
            name: [values[0], values[-1]] if values else []
            for name, values in groups.items()
        },
        "metadata": metadata or {},
    }


def save_temporal_manifest(manifest: dict) -> None:
    write_json(manifest, paths.temporal_split_path(manifest["ticker"]))


def load_temporal_manifest(ticker: str) -> dict:
    manifest_path = paths.temporal_split_path(ticker)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"找不到 {ticker} 的時間切分 manifest：{manifest_path}。"
            f"請先重跑 python -m module_a_data.build_dataset --ticker {ticker}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("protocol") != "strict_temporal_v1":
        raise ValueError(f"不支援的時間切分格式：{manifest.get('protocol')}")
    if manifest.get("ticker") != ticker:
        raise ValueError(
            f"manifest ticker={manifest.get('ticker')} 與請求的 {ticker} 不一致"
        )
    groups = manifest.get("dates", {})
    if any(name not in groups or not isinstance(groups[name], list) for name in SPLIT_NAMES):
        raise ValueError("manifest 缺少 train/gap/validation/gap/test 的完整日期清單")
    flattened = [date for name in SPLIT_NAMES for date in groups[name]]
    if flattened != sorted(flattened) or len(flattened) != len(set(flattened)):
        raise ValueError("manifest 日期必須全域時間遞增且不重複")
    expected_counts = {name: len(groups[name]) for name in SPLIT_NAMES}
    if manifest.get("counts") != expected_counts or manifest.get("n_dates") != len(flattened):
        raise ValueError("manifest 的 counts/n_dates 與日期清單不一致")
    return manifest


def split_rows(rows: Iterable[T], manifest: dict,
               date_getter: Callable[[T], str] = lambda row: row[0],
               *, require_all_manifest_dates: bool = True) -> dict[str, list[T]]:
    """依 manifest 分組；gap 日期保留在回傳值中，但不會進 train/validation/test。"""
    row_list = list(rows)
    by_date = {str(date_getter(row)): row for row in row_list}
    if len(by_date) != len(row_list):
        raise ValueError("輸入資料含重複日期，不可悄悄覆蓋後重切")
    manifest_dates = {
        date for name in SPLIT_NAMES for date in manifest["dates"].get(name, [])
    }
    if require_all_manifest_dates:
        missing = sorted(manifest_dates - set(by_date))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"資料缺少 manifest 中的 {len(missing)} 個日期（前幾筆：{preview}）。"
                "請完整重跑上游資料與向量產生流程，不可自行重算另一套切分。"
            )
    return {
        name: [by_date[date] for date in manifest["dates"].get(name, []) if date in by_date]
        for name in SPLIT_NAMES
    }


def split_name_by_date(manifest: dict) -> dict[str, str]:
    return {
        date: name
        for name in SPLIT_NAMES
        for date in manifest["dates"].get(name, [])
    }
