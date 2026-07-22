"""以市場情緒分類任務驗證 Z_fused 品質。

- 對照 A 產生的 label（BULLISH/BEARISH/NEUTRAL）計算準確率。
- train/val/test 依時間切分（70/15/15），不可隨機打散（避免時間洩漏）。
- 報告輸出到 data/outputs/metrics/classification_report.json。

用法：
    python -m module_c_fusion.validation.classifier --ticker AAPL
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from shared import paths
from shared.utils import read_json, write_json

LABEL_TO_ID = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def load_z_and_labels(ticker: str):
    """讀取所有 (date, Z_fused, label)，依日期排序。"""
    z_dir = paths.OUTPUTS / "z_fused" / ticker
    rows = []
    for f in sorted(z_dir.glob("*.npy")):
        date = f.stem
        label_file = paths.daily_json(ticker, date)
        if not label_file.exists():
            continue
        rows.append((date, np.load(f), LABEL_TO_ID[read_json(label_file)["label"]]))
    return rows


def time_split(rows, train_ratio=0.7, val_ratio=0.15):
    """時間序切分：前 70% train、中 15% val、後 15% test。"""
    n = len(rows)
    i, j = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    return rows[:i], rows[i:j], rows[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    args = ap.parse_args()

    rows = load_z_and_labels(args.ticker)
    if len(rows) < 10:
        raise SystemExit(f"Z_fused 不足（{len(rows)} 筆），先跑 module_c_fusion.fusion.train")

    train, val, test = time_split(rows)
    Xtr, ytr = np.stack([z for _, z, _ in train]), [y for _, _, y in train]
    Xte, yte = np.stack([z for _, z, _ in test]), [y for _, _, y in test]

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    labels_present = sorted(set(yte) | set(pred))
    report = {
        "ticker": args.ticker,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_period": [test[0][0], test[-1][0]] if test else [],
        "accuracy": accuracy_score(yte, pred),
        "detail": classification_report(
            yte, pred, labels=labels_present,
            target_names=[ID_TO_LABEL[i] for i in labels_present],
            output_dict=True, zero_division=0),
    }
    out = paths.OUTPUTS / "metrics" / "classification_report.json"
    write_json(report, out)
    print(f"[classifier] accuracy={report['accuracy']:.4f} -> {out}")


if __name__ == "__main__":
    main()
