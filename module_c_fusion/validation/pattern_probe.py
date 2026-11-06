"""型態類別的線性 probe（第一階段驗證，architecture.html 04 段）：Z_fused 讀得出 K 線型態嗎？

凍結 Z_fused，用 Logistic Regression 預測當天 K 線敘述的型態（shared/patterns.PATTERN_IDS，最多 7 類）。
跟 classifier.py 同一套協定：Train fit、Validation 選 C、Train＋Validation refit、Test 只評一次。
另外回報多數類 baseline；probe 只贏一點點代表 Z_fused 幾乎沒有型態資訊。

預設只看「遮掉 H_t」的融合層（fusion/train.py --mask_ht --tag mask_ht，decisions.md #125）：
K 線敘述直接寫了型態名稱，沒遮掉 H_t 的 Z_fused 從文字就讀得到型態（AAPL 測通時 0.995），
量不出圖本身有沒有學到。程式會讀融合層訓練報告確認 setup.mask_ht；要看沒遮的版本當對照，
加 --allow_ht（報告會標成 leaky_reference）。

用法：
    python -m module_c_fusion.fusion.train --ticker AAPL --mask_ht --tag mask_ht
    python -m module_c_fusion.validation.pattern_probe --ticker AAPL
    python -m module_c_fusion.validation.pattern_probe --ticker AAPL --tag "" --allow_ht   # 對照：沒遮 H_t
"""
import argparse
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from shared import paths
from shared.patterns import PATTERN_IDS
from shared.temporal_split import load_temporal_manifest, split_rows
from shared.utils import read_json, write_json
from module_c_fusion.fusion.consolidate import load_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--tag", default="mask_ht", help="讀 fusion/train.py --tag 的 Z_fused；空字串 = 正式 Z_fused")
    ap.add_argument("--allow_ht", action="store_true", help="允許讀沒遮 H_t 的 Z_fused（只當對照）")
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()
    args.tag = args.tag or None

    suffix = f"{args.ticker}_{args.tag}" if args.tag else args.ticker
    train_report = paths.OUTPUTS / "metrics" / f"fusion_train_report_{suffix}.json"
    mask_ht = bool(read_json(train_report).get("setup", {}).get("mask_ht")) if train_report.exists() else None
    if not mask_ht and not args.allow_ht:
        raise SystemExit(f"{train_report} 不是遮掉 H_t 的融合層（mask_ht={mask_ht}）。先跑 "
                         f"python -m module_c_fusion.fusion.train --ticker {args.ticker} --mask_ht --tag mask_ht；"
                         "要看沒遮的對照版本加 --allow_ht")
    idx = load_index(args.ticker, tag=args.tag)
    if idx is None:
        raise SystemExit("找不到 Z_fused 索引，先跑 module_c_fusion.fusion.train")
    rows = []
    for date, z in zip(idx["dates"].tolist(), idx["z"]):
        index = paths.vector_dir(args.ticker, date) / "index.json"
        if index.exists():
            rows.append((date, (z, PATTERN_IDS.index(read_json(index)["caption"]["pattern"]))))
    groups = split_rows(rows, load_temporal_manifest(args.ticker), require_all_manifest_dates=True)

    def xy(name):
        return (np.stack([v[0] for _, v in groups[name]]), np.asarray([v[1] for _, v in groups[name]]))

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = xy("train"), xy("validation"), xy("test")
    mean, std = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_va, x_te = (x_tr - mean) / std, (x_va - mean) / std, (x_te - mean) / std
    labels = sorted(set(y_tr))
    scores = {}
    for c in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(max_iter=2000, C=c).fit(x_tr, y_tr)
        scores[c] = f1_score(y_va, clf.predict(x_va), labels=labels, average="macro", zero_division=0)
    best_c = max(scores, key=scores.get)
    clf = LogisticRegression(max_iter=2000, C=best_c).fit(np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]))
    pred = clf.predict(x_te)
    majority = Counter(y_tr).most_common(1)[0][0]
    report = {
        "run_id": args.run_id, "ticker": args.ticker, "tag": args.tag, "target": "caption_pattern",
        "fusion_mask_ht": mask_ht, "role": "main" if mask_ht else "leaky_reference",
        "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te),
        "validation_macro_f1_by_C": {str(k): round(v, 4) for k, v in scores.items()}, "selected_C": best_c,
        "test_accuracy": round(accuracy_score(y_te, pred), 4),
        "test_macro_f1": round(f1_score(y_te, pred, labels=labels, average="macro", zero_division=0), 4),
        "majority_baseline_accuracy": round(float(np.mean(y_te == majority)), 4),
        "test_distribution": {PATTERN_IDS[k]: v for k, v in sorted(Counter(y_te.tolist()).items())},
    }
    out = paths.OUTPUTS / "metrics" / f"pattern_probe_report_{suffix}.json"
    write_json(report, out)
    print(f"[pattern_probe] test accuracy={report['test_accuracy']}（多數類 {report['majority_baseline_accuracy']}），"
          f"macro F1={report['test_macro_f1']} -> {out}")


if __name__ == "__main__":
    main()
