"""標籤定義對照：Train-only 全域 q33／q67 vs 滾動 N 日 q33／q67。

只改變「標籤怎麼算」，輸入特徵、樣本日期、分類頭設定與 seeds 全部固定：

- global ：現行做法。只用 Train 的 5 日報酬算 q33／q67，固定套用到所有日期。
- rolling：第 t 天的門檻只用 t−5 以前「已實現」的最近 N 筆 5 日報酬（預設 N=250）。
           每天只看過去，Train／Validation 規則相同，不需另外限定只用 Train。

輸入（皆為既有產物，不重新抽特徵）：
- vit_single_k20／vit_dual_k20_volume：像素畫法實驗快取的 Frozen ViT CLS（966／202 天）。
- numeric_w20：20 日逐日報酬＋相對成交量的 Logistic Regression。
- zfused_lr：正式 pipeline 的 Z_fused＋Logistic Regression（僅 AAPL／NVDA 有 Z_fused）。

滾動門檻需要 2021-01-04 以前的暖身股價；另外下載到本實驗輸出目錄，只用於計算門檻，
不寫入 data/raw/ohlcv。Test 完全不評估。

用法（需先跑過 run_pixel_render_comparison.py）：
    python experiments/vision/run_rolling_label_comparison.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import paths
from shared.temporal_split import load_temporal_manifest
from experiments.vision.run_pixel_render_comparison import (
    OUTPUT_ROOT as PIXEL_ROOT, SPLITS, TICKERS, load_ohlcv, metric_payload, numeric_baseline,
    stats, train_head,
)

PROTOCOL = "rolling_label_comparison_v1"
OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "rolling_label"
VIT_FEATURES = PIXEL_ROOT / "px_vs_mpl_w20" / "features"
VIT_CONFIGS = {"vit_single_k20": ["mpl_candlestick"],
               "vit_dual_k20_volume": ["mpl_candlestick", "mpl_volume"]}
SCHEMES = ("global", "rolling")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")
HORIZON, WINDOW, WARMUP_START = 5, 20, "2019-10-01"


def extended_close(ticker: str, output: Path) -> pd.Series:
    """正式 OHLCV 前面接上暖身期收盤價；重疊日期必須與正式資料一致。"""
    from module_a_data.crawler.fetch_ohlcv import fetch_ohlcv

    formal = load_ohlcv(ticker)["Close"].astype(float)
    cache = output / "warmup_ohlcv" / f"{ticker}.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        end = (formal.index[5] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fetch_ohlcv(ticker, WARMUP_START, end).to_csv(cache)
    warmup = pd.read_csv(cache, parse_dates=["Date"]).set_index("Date")["Close"].astype(float)
    overlap = warmup.index.intersection(formal.index)
    if len(overlap) == 0 or not np.allclose(warmup.loc[overlap], formal.loc[overlap], rtol=1e-6):
        raise ValueError(f"{ticker} 暖身股價與正式 OHLCV 的重疊日期不一致")
    return pd.concat([warmup[warmup.index < formal.index[0]], formal])


def build_labels(ticker: str, manifest: dict, output: Path, n_days: int) -> dict:
    close = extended_close(ticker, output)
    forward = close.shift(-HORIZON) / close - 1
    known = forward.shift(HORIZON)        # 第 t 天能看到的最新已實現報酬是 forward[t-5]
    rolling_low = known.rolling(n_days, min_periods=n_days).quantile(1 / 3)
    rolling_high = known.rolling(n_days, min_periods=n_days).quantile(2 / 3)
    train_dates = pd.to_datetime(manifest["dates"]["train"])
    global_low, global_high = np.quantile(forward.loc[train_dates], [0.3333, 0.6667])

    labels, shares, thresholds = {s: {} for s in SCHEMES}, {s: {} for s in SCHEMES}, {}
    for split in SPLITS:
        dates = pd.to_datetime(manifest["dates"][split])
        returns = forward.loc[dates].to_numpy()
        low, high = rolling_low.loc[dates].to_numpy(), rolling_high.loc[dates].to_numpy()
        if np.isnan(low).any() or np.isnan(returns).any():
            raise ValueError(f"{ticker} {split} 的滾動門檻或報酬不完整")
        labels["global"][split] = np.where(
            returns < global_low, 0, np.where(returns > global_high, 2, 1))
        labels["rolling"][split] = np.where(returns < low, 0, np.where(returns > high, 2, 1))
        for scheme in SCHEMES:
            counts = np.bincount(labels[scheme][split], minlength=3)
            shares[scheme][split] = (counts / counts.sum()).round(4).tolist()
        thresholds[split] = {
            "low_range": [float(low.min()), float(low.max())],
            "high_range": [float(high.min()), float(high.max())],
            "agreement_with_global": float(
                (labels["global"][split] == labels["rolling"][split]).mean()),
        }
    return {"labels": labels, "shares": shares, "rolling": thresholds,
            "global": {"lower": float(global_low), "upper": float(global_high)}}


def numeric_metrics(ticker: str, manifest: dict, labels: dict, seed: int) -> dict:
    frame = load_ohlcv(ticker)
    position = {date.strftime("%Y-%m-%d"): i for i, date in enumerate(frame.index)}
    rows = {
        split: {"windows": [frame.iloc[position[d] - WINDOW + 1:position[d] + 1]
                            for d in manifest["dates"][split]],
                "labels": labels[split]}
        for split in SPLITS
    }
    return numeric_baseline({"rows": rows}, seed)


def zfused_metrics(ticker: str, manifest: dict, labels: dict, seed: int) -> dict | None:
    """正式 Z_fused（Fusion 以全域標籤在 Train 訓練）＋標準化 Logistic Regression。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from module_c_fusion.fusion.consolidate import load_index

    index = load_index(ticker)
    if index is None:
        return None
    z_by_date = dict(zip(index["dates"].tolist(), index["z"]))
    x = {split: np.stack([z_by_date[d] for d in manifest["dates"][split]]) for split in SPLITS}
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=2000, random_state=seed))
    model.fit(x["train"], labels["train"])
    return metric_payload(labels["validation"], model.predict_proba(x["validation"]))


def mean_ticker(by_ticker: dict) -> dict:
    return {key: float(np.mean([value[key] for value in by_ticker.values()])) for key in METRICS}


def write_markdown(report: dict, output: Path) -> None:
    n_days = report["rolling_days"]
    lines = [
        f"# 標籤定義對照：Train-only 全域門檻 vs 滾動 {n_days} 日門檻", "",
        f"- Protocol：`{report['protocol']}`；Test 未開啟。",
        f"- 每檔 Train／Validation = {report['split_counts']['train']}／"
        f"{report['split_counts']['validation']}；ViT 分類頭 seeds {report['seeds']}、"
        f"固定 {report['epochs']} epochs。",
        "- 兩種標籤的任務不同，Macro-F1 不能直接比大小；跨標籤請看 ROC-AUC（隨機 0.5）與 "
        "balanced accuracy（隨機 0.333）。",
        "", "## 標籤分布（BEARISH／NEUTRAL／BULLISH）", "",
        "| 股票 | 門檻 | Train | Validation | 與全域標籤相同（Train／Val） |", "|---|---|---|---|---|",
    ]
    for ticker, info in report["labels"].items():
        g, r = info["global"], info["rolling"]
        lines.append(f"| {ticker} | 全域 {g['lower']:+.3f}／{g['upper']:+.3f} | "
                     f"{info['shares']['global']['train']} | "
                     f"{info['shares']['global']['validation']} | — |")
        lines.append(
            f"| {ticker} | 滾動 下 {r['validation']['low_range'][0]:+.3f}～"
            f"{r['validation']['low_range'][1]:+.3f}、上 {r['validation']['high_range'][0]:+.3f}～"
            f"{r['validation']['high_range'][1]:+.3f}（Validation 期間） | "
            f"{info['shares']['rolling']['train']} | {info['shares']['rolling']['validation']} | "
            f"{r['train']['agreement_with_global']:.2f}／"
            f"{r['validation']['agreement_with_global']:.2f} |")
    lines += ["", "## Validation（mean-ticker）", "",
              "| 輸入 | 標籤 | Macro-F1 | Balanced acc | ROC-AUC | Log loss |",
              "|---|---|---:|---:|---:|---:|"]
    for name, by_scheme in report["summary"].items():
        for scheme in SCHEMES:
            values = by_scheme[scheme]
            spread = f" ± {values['macro_f1_std']:.4f}" if "macro_f1_std" in values else ""
            lines.append(f"| {name} | {scheme} | {values['macro_f1']:.4f}{spread} | "
                         f"{values['balanced_accuracy']:.4f} | {values['roc_auc']:.4f} | "
                         f"{values['log_loss']:.4f} |")
    lines += ["", "## 各股票 ROC-AUC（global → rolling）", "",
              "| 輸入 | " + " | ".join(report["tickers"]) + " |",
              "|---|" + "---|" * len(report["tickers"])]
    for name, by_scheme in report["by_ticker_auc"].items():
        lines.append(f"| {name} | " + " | ".join(
            (f"{by_scheme['global'][t]:.4f} → {by_scheme['rolling'][t]:.4f}"
             if t in by_scheme["global"] else "—") for t in report["tickers"]) + " |")
    lines += ["", "## 限制", ""] + [f"- {text}" for text in report["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-days", type=int, default=250)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    args = parser.parse_args()

    output = OUTPUT_ROOT / (args.run_name or f"rolling_n{args.rolling_days}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_temporal_manifest("AAPL")   # 三檔共同日期
    labels = {t: build_labels(t, manifest, output, args.rolling_days) for t in TICKERS}

    def vit_features(ticker: str, config: str, split: str) -> np.ndarray:
        parts = []
        for kind in VIT_CONFIGS[config]:
            source = VIT_FEATURES / f"{ticker}_{kind}_{split}.npy"
            if not source.exists():
                raise SystemExit(f"找不到 {source}，先跑 run_pixel_render_comparison.py")
            parts.append(np.load(source))
        return np.concatenate(parts, axis=1)

    summary, by_ticker_auc = {}, {}
    for config in VIT_CONFIGS:
        summary[config], by_ticker_auc[config] = {}, {}
        for scheme in SCHEMES:
            per_seed, per_ticker_auc = [], {t: [] for t in TICKERS}
            for seed in args.seeds:
                by_ticker = {}
                for ticker in TICKERS:
                    y = labels[ticker]["labels"][scheme]
                    probabilities = train_head(
                        vit_features(ticker, config, "train"), y["train"],
                        vit_features(ticker, config, "validation"),
                        seed, args.epochs, args.head_lr)
                    by_ticker[ticker] = metric_payload(y["validation"], probabilities)
                    per_ticker_auc[ticker].append(by_ticker[ticker]["roc_auc"])
                per_seed.append(mean_ticker(by_ticker))
            summary[config][scheme] = {
                key: float(np.mean([item[key] for item in per_seed])) for key in METRICS}
            summary[config][scheme]["macro_f1_std"] = stats(
                [item["macro_f1"] for item in per_seed])["std"]
            by_ticker_auc[config][scheme] = {
                t: float(np.mean(values)) for t, values in per_ticker_auc.items()}
        print(f"[rolling_label] {config} 完成", flush=True)

    for name, function in (("numeric_w20", numeric_metrics), ("zfused_lr", zfused_metrics)):
        summary[name], by_ticker_auc[name] = {}, {}
        for scheme in SCHEMES:
            by_ticker = {}
            for ticker in TICKERS:
                result = function(ticker, manifest, labels[ticker]["labels"][scheme], args.seeds[0])
                if result is not None:
                    by_ticker[ticker] = result
            summary[name][scheme] = mean_ticker(by_ticker)
            by_ticker_auc[name][scheme] = {t: v["roc_auc"] for t, v in by_ticker.items()}
        print(f"[rolling_label] {name} 完成", flush=True)

    report = {
        "protocol": PROTOCOL, "tickers": TICKERS, "rolling_days": args.rolling_days,
        "horizon_days": HORIZON, "seeds": args.seeds, "epochs": args.epochs,
        "split_counts": {s: len(manifest["dates"][s]) for s in SPLITS},
        "split_manifest": str(paths.temporal_split_path("AAPL").relative_to(ROOT)),
        "test_was_opened": False,
        "labels": {t: {k: v for k, v in labels[t].items() if k != "labels"} for t in TICKERS},
        "summary": summary, "by_ticker_auc": by_ticker_auc,
        "limitations": [
            "兩種標籤是不同的預測任務；Macro-F1 的差異同時包含任務難度與類別比例的變化。",
            "N 為事先指定，沒有用 Validation 挑選；若之後再試其他 N，就不再是事前指定。",
            "zfused_lr 只有 AAPL／NVDA；其 Fusion 是以全域標籤在 Train 訓練，對 rolling 標籤略為不利。",
            "每檔 202 筆 Validation；先前觀察到同一設定因樣本小幅變動可差 0.037。",
            "這段 Validation 已在先前多輪實驗中被看過。",
            "暖身股價只用於計算滾動門檻；MSFT 沒有正式 manifest，沿用 AAPL 的日期切分。",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output)
    print(f"[rolling_label] report -> {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
