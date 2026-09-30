"""事件驗證頭：拿 Z_fused 預測當天事件類型，驗證 Z_fused 有沒有吸收到事件資訊。

跟 classifier.py（分類驗證）是同一種性質的健檢，差別在驗證的目標不同：
    classifier.py：Z_fused 能不能預測「未來」的價格方向（預測力）。
    這支：Z_fused 能不能反映「當天」實際發生了什麼事件（資訊保真度）。
兩者都準，才代表 Z_fused 是真的吸收了輸入資訊，不是靠某種捷徑矇對其中一項。
（docs/decisions.md #33，補上原本自製分工 PDF「用事件抽取驗證 Z_fused」的設計意圖。）

輸入：
    data/labels/event_ground_truth/{ticker}.json   人工／LLM 標記的事件 ground truth
    data/outputs/z_fused/{ticker}/{date}.npy        對應日期的 Z_fused 向量
輸出：
    data/outputs/metrics/event_validation_head_report.json

樣本數只有 ground truth 涵蓋的天數（目前 149 天），仍必須沿用 A 的共用
strict temporal manifest：Train fit、Validation 選 C、Train+Validation refit，Test 最後只評估
一次。某類事件若在任一區間正樣本過少，報告會明確標注「資料量不足，
不評估」；不會為了多用幾筆資料改回隨機 cross-validation。

用法：
    python -m module_c_fusion.validation.event_validation_head --ticker AAPL
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from shared import paths
from shared.temporal_split import load_temporal_manifest, split_rows
from shared.utils import read_json, write_json

EVENT_TYPES = ["EARNINGS", "MA", "PRODUCT_LAUNCH", "LAWSUIT",
               "GUIDANCE", "DIVIDEND", "MANAGEMENT_CHANGE"]
MIN_TRAIN_POSITIVES = 5


def load_data(ticker: str, allowed_dates: set[str] | None = None):
    gt = read_json(paths.event_ground_truth_path(ticker))
    dates = sorted(date for date in gt if allowed_dates is None or date in allowed_dates)
    X, Y = [], []
    for d in dates:
        z = np.load(paths.OUTPUTS / "z_fused" / ticker / f"{d}.npy")
        X.append(z)
        Y.append([1 if e in gt[d] else 0 for e in EVENT_TYPES])
    return [(date, x, y) for date, x, y in zip(dates, X, Y)]


def _binary_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    tp = int(((prediction == 1) & (y == 1)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float((prediction == y).mean())
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def evaluate_category(train_rows: list, validation_rows: list, test_rows: list,
                      category_index: int, c_values: list[float]) -> dict:
    def arrays(rows):
        return (np.stack([row[1] for row in rows]),
                np.asarray([row[2][category_index] for row in rows]))

    x_train, y_train = arrays(train_rows)
    x_validation, y_validation = arrays(validation_rows)
    x_test, y_test = arrays(test_rows)
    counts = {
        "train": int(y_train.sum()),
        "validation": int(y_validation.sum()),
        "test": int(y_test.sum()),
    }
    if counts["train"] < MIN_TRAIN_POSITIVES or len(np.unique(y_train)) < 2:
        return {
            "positive_days": counts,
            "evaluated": False,
            "reason": f"Train 正樣本只有 {counts['train']} 天，無法穩定 fit",
        }
    if counts["validation"] == 0 or counts["test"] == 0:
        return {
            "positive_days": counts,
            "evaluated": False,
            "reason": "Validation 或 Test 沒有正樣本，不回報無意義的 F1",
        }

    candidates = []
    best = None
    for c_value in c_values:
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=c_value,
        ).fit(x_train, y_train)
        result = _binary_metrics(y_validation, classifier.predict(x_validation))
        candidates.append({"C": c_value, "f1": result["f1"], "accuracy": result["accuracy"]})
        key = (result["f1"], result["accuracy"], -c_value)
        if best is None or key > best[0]:
            best = (key, c_value)

    x_fit = np.concatenate([x_train, x_validation], axis=0)
    y_fit = np.concatenate([y_train, y_validation], axis=0)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2000, C=best[1],
    ).fit(x_fit, y_fit)
    result = _binary_metrics(y_test, classifier.predict(x_test))
    return {
        "positive_days": counts,
        "evaluated": True,
        "selected_C": best[1],
        "validation_candidates": candidates,
        **result,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--c_values", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    args = ap.parse_args()

    manifest = load_temporal_manifest(args.ticker)
    allowed_dates = {
        date for dates in manifest["dates"].values() for date in dates
    }
    rows = load_data(args.ticker, allowed_dates)
    groups = split_rows(rows, manifest, require_all_manifest_dates=False)
    train_rows = groups["train"]
    validation_rows = groups["validation"]
    test_rows = groups["test"]
    if not train_rows or not validation_rows or not test_rows:
        raise SystemExit("event ground truth 與 manifest 交集後有空集合，無法做嚴格時間評估")
    z_dim = int(train_rows[0][1].shape[0])
    print(f"[event_validation_head] {args.ticker}: train={len(train_rows)} "
          f"validation={len(validation_rows)} test={len(test_rows)}，Z_fused 維度={z_dim}")

    per_category = {}
    for i, event_type in enumerate(EVENT_TYPES):
        result = evaluate_category(
            train_rows, validation_rows, test_rows, i, args.c_values,
        )
        per_category[event_type] = result
        if result["evaluated"]:
            print(f"  {event_type}: positives={result['positive_days']} C={result['selected_C']} "
                 f"precision={result['precision']:.3f} recall={result['recall']:.3f} f1={result['f1']:.3f}")
        else:
            print(f"  {event_type}: {result['reason']}")

    evaluated = [r for r in per_category.values() if r["evaluated"]]
    tp = sum(r["tp"] for r in evaluated)
    fp = sum(r["fp"] for r in evaluated)
    fn = sum(r["fn"] for r in evaluated)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
               if micro_precision + micro_recall else 0.0)

    report = {
        "protocol": manifest["protocol"],
        "ticker": args.ticker,
        "split_manifest": str(paths.temporal_split_path(args.ticker)),
        "n_train": len(train_rows),
        "n_validation": len(validation_rows),
        "n_test": len(test_rows),
        "z_fused_dim": z_dim,
        "selection_metric": "validation.f1",
        "final_fit": "train_plus_validation",
        "per_category": per_category,
        "micro_avg": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1,
                     "tp": tp, "fp": fp, "fn": fn,
                     "categories_included": [k for k, r in per_category.items() if r["evaluated"]]},
    }
    out = paths.OUTPUTS / "metrics" / "event_validation_head_report.json"
    write_json(report, out)
    print(f"\n[event_validation_head] micro-avg (只算資料量足夠的類別): "
         f"precision={micro_precision:.3f}, recall={micro_recall:.3f}, f1={micro_f1:.3f}")
    print(f"[event_validation_head] -> {out}")


if __name__ == "__main__":
    main()
