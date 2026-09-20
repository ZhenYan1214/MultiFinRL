"""共用的時間序實驗切分。

所有模組都使用同一份 split 標記，避免在資料取交集後各自重算 70/15/15，造成日期邊界
不一致。對使用未來報酬產生的標籤，切分邊界前的樣本會標成 ``purged``，確保其目標日期
不會跨入下一個資料區段。
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


MODEL_SPLITS = ("train", "validation", "test")
VALID_SPLITS = (*MODEL_SPLITS, "purged")


def assign_temporal_splits(
    dates: Sequence[str],
    target_dates: Sequence[str] | None = None,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> list[str]:
    """依日期切 train/validation/test，並 purge 目標跨越邊界的樣本。

    ``target_dates[i]`` 是 ``dates[i]`` 這筆監督訊號最後使用到的市場日期。例如五日報酬
    標籤就是 t+5；若 t 屬於 train、t+5 已進入 validation，這筆會標成 purged。
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio 必須介於 0 與 1，實際為 {train_ratio}")
    if not 0 <= validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError(
            "validation_ratio 必須 >= 0，且 train_ratio + validation_ratio 必須 < 1"
        )
    if list(dates) != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError("dates 必須是已排序且不重複的 YYYY-MM-DD 日期")
    if target_dates is not None and len(target_dates) != len(dates):
        raise ValueError("target_dates 長度必須和 dates 相同")
    if len(dates) < 3:
        raise ValueError("至少需要 3 筆資料才能切 train/validation/test")

    train_end = int(len(dates) * train_ratio)
    validation_end = int(len(dates) * (train_ratio + validation_ratio))
    if train_end == 0 or validation_end <= train_end or validation_end >= len(dates):
        raise ValueError("資料量或切分比例無法產生非空的 train/validation/test")

    validation_start = dates[train_end]
    test_start = dates[validation_end]
    result: list[str] = []
    for i, date in enumerate(dates):
        if i < train_end:
            split = "train"
            next_boundary = validation_start
        elif i < validation_end:
            split = "validation"
            next_boundary = test_start
        else:
            split = "test"
            next_boundary = None

        if target_dates is not None and next_boundary is not None:
            if target_dates[i] >= next_boundary:
                split = "purged"
        result.append(split)
    return result


def partition_by_split(rows: Iterable, split_position: int = -1) -> tuple[list, list, list]:
    """依 row 內既有 split 欄位分組；purged 樣本不進任何模型資料集。"""
    groups = {name: [] for name in MODEL_SPLITS}
    for row in rows:
        split = row[split_position]
        if split in groups:
            groups[split].append(row)
        elif split != "purged":
            raise ValueError(f"未知 split: {split}")
    return groups["train"], groups["validation"], groups["test"]


def split_summary(dates: Sequence[str], splits: Sequence[str]) -> dict:
    """建立可寫進 manifest/meta/report 的切分摘要。"""
    if len(dates) != len(splits):
        raise ValueError("dates 與 splits 長度不同")
    counts = Counter(splits)
    periods = {}
    for name in VALID_SPLITS:
        selected = [date for date, split in zip(dates, splits) if split == name]
        periods[name] = [selected[0], selected[-1]] if selected else []
    return {
        "counts": {name: int(counts.get(name, 0)) for name in VALID_SPLITS},
        "periods": periods,
    }
