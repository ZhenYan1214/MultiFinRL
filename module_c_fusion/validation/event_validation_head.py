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

樣本數只有 ground truth 涵蓋的天數（目前 149 天），用 k-fold cross-validation
盡量把有限的資料用滿，而不是切一份固定的 train/test（樣本太少切了會不穩定）。
部分事件類別（如 MA、MANAGEMENT_CHANGE）在 149 天裡只出現 1 次，這種類別
無法用 cross-validation 做有意義的評估（一定會有某個 fold 的訓練集完全沒看過
正樣本），會在報告裡明確標注「資料量不足，不評估」，不是程式壞掉。

用法：
    python -m module_c_fusion.validation.event_validation_head --ticker AAPL
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

from shared import paths
from shared.utils import read_json, write_json

EVENT_TYPES = ["EARNINGS", "MA", "PRODUCT_LAUNCH", "LAWSUIT",
               "GUIDANCE", "DIVIDEND", "MANAGEMENT_CHANGE"]
MIN_POSITIVES_FOR_CV = 5  # 少於這個數字的類別，cross-validation 結果不可信，只報計數不評分


def load_data(ticker: str):
    gt = read_json(paths.event_ground_truth_path(ticker))
    dates = sorted(gt.keys())
    X, Y = [], []
    for d in dates:
        z = np.load(paths.OUTPUTS / "z_fused" / ticker / f"{d}.npy")
        X.append(z)
        Y.append([1 if e in gt[d] else 0 for e in EVENT_TYPES])
    return dates, np.stack(X), np.array(Y)


def evaluate_category(x: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> dict:
    """單一事件類別的 cross-validation P/R/F1。y 是 0/1 向量。"""
    n_pos = int(y.sum())
    if n_pos < MIN_POSITIVES_FOR_CV:
        return {"n_positive_days": n_pos, "evaluated": False,
               "reason": f"正樣本只有 {n_pos} 天，少於門檻 {MIN_POSITIVES_FOR_CV}，cross-validation 不可信"}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1)
    pred = cross_val_predict(clf, x, y, cv=skf)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n_positive_days": n_pos, "evaluated": True,
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dates, X, Y = load_data(args.ticker)
    print(f"[event_validation_head] {args.ticker}: {len(dates)} 天，Z_fused 維度={X.shape[1]}")

    per_category = {}
    for i, event_type in enumerate(EVENT_TYPES):
        result = evaluate_category(X, Y[:, i], seed=args.seed)
        per_category[event_type] = result
        if result["evaluated"]:
            print(f"  {event_type}: n={result['n_positive_days']} "
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
        "ticker": args.ticker,
        "n_days": len(dates),
        "z_fused_dim": int(X.shape[1]),
        "cv_folds": 5,
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
