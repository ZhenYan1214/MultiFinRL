"""以市場情緒分類任務驗證 Z_fused 品質。

- 對照 A 產生的 label（BULLISH/BEARISH/NEUTRAL）計算準確率。
- 使用 A 寫入、跨模組共用的時間 split，不對資料交集重新切分。
- 報告輸出到 data/outputs/metrics/classification_report_{TICKER}.json。

用法：
    python -m module_c_fusion.validation.classifier --ticker AAPL
"""
import argparse
import datetime as dt

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from shared import paths
from shared.temporal_split import partition_by_split
from shared.utils import write_json
from module_c_fusion.fusion.consolidate import load_index

LABEL_TO_ID = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def load_z_and_labels(ticker: str):
    """讀取所有 (date, Z_fused, label)，依日期排序。

    只讀帶資料指紋的彙整索引，避免逐日檔案殘留時靜默混入舊實驗。
    """
    idx = load_index(ticker)
    if idx is None or "split" not in idx:
        raise ValueError("缺少新版 Z_fused index；請重跑 fusion.train/consolidate")
    return list(zip(idx["dates"].tolist(), list(idx["z"]),
                    idx["label"].tolist(), idx["split"].tolist()))


def _evaluate(clf, rows) -> dict:
    x = np.stack([z for _, z, _, _ in rows])
    y = [label for _, _, label, _ in rows]
    pred = clf.predict(x)
    labels_present = sorted(set(y) | set(pred))
    return {
        "n": len(rows),
        "period": [rows[0][0], rows[-1][0]],
        "accuracy": accuracy_score(y, pred),
        "detail": classification_report(
            y, pred, labels=labels_present,
            target_names=[ID_TO_LABEL[i] for i in labels_present],
            output_dict=True, zero_division=0,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--weighted", action="store_true",
                    help="診斷用：依訓練集類別出現頻率加權，預設關閉（結果見 docs/decisions.md #28）")
    args = ap.parse_args()

    rows = load_z_and_labels(args.ticker)
    if len(rows) < 10:
        raise SystemExit(f"Z_fused 不足（{len(rows)} 筆），先跑 module_c_fusion.fusion.train")

    train, val, test = partition_by_split(rows)
    if not train or not val or not test:
        raise SystemExit(
            f"split 不完整：train={len(train)}, validation={len(val)}, test={len(test)}；"
            "請重跑 build_dataset 與 fusion.train"
        )
    Xtr = np.stack([z for _, z, _, _ in train])
    ytr = [y for _, _, y, _ in train]

    class_weight = "balanced" if args.weighted else None
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight)  # 新版 sklearn 預設就是 multinomial，不用再指定
    clf.fit(Xtr, ytr)
    validation_result = _evaluate(clf, val)
    test_result = _evaluate(clf, test)
    report = {
        "run_id": dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "ticker": args.ticker,
        "weighted": args.weighted,
        "protocol": "fusion and probe fit on train; validation reported separately; test held out",
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_period": [train[0][0], train[-1][0]],
        "validation": validation_result,
        "test": test_result,
        # 保留既有欄位名稱，讓舊的報告讀取腳本仍取得正式 test 指標。
        "test_period": test_result["period"],
        "accuracy": test_result["accuracy"],
        "detail": test_result["detail"],
    }
    out = paths.OUTPUTS / "metrics" / f"classification_report_{args.ticker}.json"
    write_json(report, out)
    archive = (paths.OUTPUTS / "metrics" /
               f"classification_report_{args.ticker}_{report['run_id']}.json")
    write_json(report, archive)
    print(f"[classifier] validation accuracy={validation_result['accuracy']:.4f}; "
          f"held-out test accuracy={test_result['accuracy']:.4f} -> {archive}（latest: {out}）")


if __name__ == "__main__":
    main()
