"""以市場情緒分類任務驗證 Z_fused 品質。

- 對照 A 產生的 label（BULLISH/BEARISH/NEUTRAL）計算準確率。
- 沿用 A 建立的唯一時間切分 manifest；classifier 不可自行重切日期。
- train fit、validation 選 C、test 最後只評估一次。
- 報告輸出到 data/outputs/metrics/classification_report.json。

用法：
    python -m module_c_fusion.validation.classifier --ticker AAPL
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from shared import paths
from shared.temporal_split import load_temporal_manifest, split_rows
from shared.utils import read_json, write_json
from module_c_fusion.fusion.consolidate import load_index

LABEL_TO_ID = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def load_z_and_labels(ticker: str):
    """讀取所有 (date, Z_fused, label)，依日期排序。

    優先讀彙整索引（data/outputs/z_fused/{ticker}_index.npz）；
    索引還沒建立時 fallback 成逐日掃描（速度較慢，但結果一致）。
    """
    idx = load_index(ticker)
    if idx is not None:
        return list(zip(idx["dates"].tolist(), list(idx["z"]), idx["label"].tolist()))

    z_dir = paths.OUTPUTS / "z_fused" / ticker
    rows = []
    for f in sorted(z_dir.glob("*.npy")):
        date = f.stem
        label_file = paths.daily_json(ticker, date)
        if not label_file.exists():
            continue
        rows.append((date, np.load(f), LABEL_TO_ID[read_json(label_file)["label"]]))
    return rows


def arrays(rows):
    return np.stack([z for _, z, _ in rows]), np.asarray([y for _, _, y in rows])


def metrics(y_true, prediction) -> dict:
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, labels=[0, 1, 2],
                             average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1, 2]).tolist(),
        "detail": classification_report(
            y_true, prediction, labels=[0, 1, 2],
            target_names=[ID_TO_LABEL[i] for i in range(3)],
            output_dict=True, zero_division=0,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--weighted", action="store_true",
                    help="診斷用：依訓練集類別出現頻率加權，預設關閉（結果見 docs/decisions.md #28）")
    ap.add_argument("--c_values", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0],
                    help="只用 validation macro F1 選擇的 Logistic Regression C 候選")
    args = ap.parse_args()

    rows = load_z_and_labels(args.ticker)
    if len(rows) < 10:
        raise SystemExit(f"Z_fused 不足（{len(rows)} 筆），先跑 module_c_fusion.fusion.train")

    manifest = load_temporal_manifest(args.ticker)
    groups = split_rows(rows, manifest, require_all_manifest_dates=True)
    train, val, test = groups["train"], groups["validation"], groups["test"]
    Xtr, ytr = arrays(train)
    Xval, yval = arrays(val)

    class_weight = "balanced" if args.weighted else None
    validation_results = []
    best = None
    for c_value in args.c_values:
        candidate = LogisticRegression(max_iter=1000, class_weight=class_weight, C=c_value)
        candidate.fit(Xtr, ytr)
        result = metrics(yval, candidate.predict(Xval))
        validation_results.append({
            "C": c_value,
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
        })
        key = (result["macro_f1"], result["accuracy"], -c_value)
        if best is None or key > best[0]:
            best = (key, c_value)

    selected_c = best[1]
    train_validation = train + val
    Xfit, yfit = arrays(train_validation)
    Xte, yte = arrays(test)
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight, C=selected_c)
    clf.fit(Xfit, yfit)
    test_result = metrics(yte, clf.predict(Xte))

    report = {
        "protocol": manifest["protocol"],
        "ticker": args.ticker,
        "weighted": args.weighted,
        "selected_C": selected_c,
        "selection_metric": "validation.macro_f1",
        "validation_candidates": validation_results,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_period": manifest["periods"]["train"],
        "validation_period": manifest["periods"]["validation"],
        "test_period": [test[0][0], test[-1][0]] if test else [],
        "gap_trading_days": manifest["gap_trading_days"],
        "split_manifest": str(paths.temporal_split_path(args.ticker)),
        "final_fit": "train_plus_validation",
        **test_result,
    }
    out = paths.OUTPUTS / "metrics" / "classification_report.json"
    write_json(report, out)
    print(f"[classifier] selected C={selected_c} on validation macro F1")
    print(f"[classifier] test accuracy={report['accuracy']:.4f}, "
          f"macro_f1={report['macro_f1']:.4f} -> {out}")


if __name__ == "__main__":
    main()
