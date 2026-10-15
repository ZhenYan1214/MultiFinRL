"""Frozen ViT：2016 起的長期間 × 預測天期 × 圖組合 × 分類頭設定，walk-forward 驗證。

與先前只有單一 202 天 Validation 的對照不同，這裡用逐年往前推的 walk-forward：
每個 fold 以「該年以前」的全部樣本訓練、在該年驗證（預設 2019～2025 共 7 個 folds），
三檔合計約 5,000 筆 Validation，可分辨的差距約為 0.01。

- 股票：預設 AAPL／NVDA／MSFT，可用 --tickers 擴充。OHLCV 另外下載（yfinance、auto_adjust=False，與正式來源相同）
  到本實驗輸出目錄，並核對與正式 data/raw/ohlcv 重疊日期一致。
- 期間：2016-01 起，到正式 Validation 結束日為止；正式 Test 期間不建立任何樣本。
- 圖：全部以正式 chart_generator 的 mplfinance／matplotlib 樣式重畫（20 日窗口）：
  K 線、Volume、SMA（收盤價線＋SMA5／10／20）。
- 輸入：single_k20、dual_k20_volume、dual_k20_sma、triple_k20_volume_sma（Frozen CLS 串接）。
- 標籤：未來第 h 個交易日報酬（h=1／3／5）的三分類；門檻為各 fold、各股票 Train 的
  q33／q67。Train 與 Validation 之間固定留 5 個交易日，三種 h 使用完全相同的樣本。
- 分類頭：LayerNorm＋Dropout＋Linear，固定 epochs、回報最後一個 epoch（不以 Validation 挑）。

統計單位是「fold × 股票」（預設 21 個），seeds 先在單位內平均；配對比較回報單位差值的
平均、標準誤與勝出單位數。跨天期請比較 ROC-AUC／balanced accuracy，不要比 Macro-F1。

用法：
    python experiments/vision/run_horizon_period_grid.py --smoke
    python experiments/vision/run_horizon_period_grid.py 2>&1 | tee /tmp/vit_horizon_grid.log
中斷後用相同指令續跑：圖、特徵與已完成的設定區塊都會跳過。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import paths
from shared.temporal_split import load_temporal_manifest
from experiments.vision.run_pixel_render_comparison import (
    load_ohlcv, metric_payload, numeric_baseline, train_head,
)

PROTOCOL = "vit_horizon_period_grid_v1"
OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "vit_horizon_grid"
WINDOW, GAP = 20, 5
KINDS = ("k20", "volume", "sma")
CONFIGS = {
    "single_k20": ["k20"],
    "dual_k20_volume": ["k20", "volume"],
    "dual_k20_sma": ["k20", "sma"],
    "triple_k20_volume_sma": ["k20", "volume", "sma"],
}
HEADS = {"lr1e-3_e30": (1e-3, 30), "lr1e-4_e100": (1e-4, 100)}
COMPARISONS = [("dual_k20_volume", "single_k20"), ("dual_k20_sma", "single_k20"),
               ("triple_k20_volume_sma", "single_k20"), ("dual_k20_sma", "dual_k20_volume"),
               ("triple_k20_volume_sma", "dual_k20_volume")]
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")


# --- 資料 ---

def load_long_ohlcv(ticker: str, output: Path, start: str, end: str) -> pd.DataFrame:
    """下載長期間 OHLCV 到實驗目錄；正式 pipeline 有該股票時，重疊日期收盤價必須一致。"""
    from module_a_data.crawler.fetch_ohlcv import fetch_ohlcv

    cache = output.parent / "ohlcv" / f"{ticker}_{start}_{end}.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        fetch_ohlcv(ticker, start, end).to_csv(cache)
    frame = pd.read_csv(cache, parse_dates=["Date"]).set_index("Date").sort_index()
    if not (paths.RAW_OHLCV / f"{ticker}.csv").exists():
        return frame          # 擴充股票池的股票沒有正式 OHLCV 可核對
    formal = load_ohlcv(ticker)["Close"].astype(float)
    overlap = frame.index.intersection(formal.index)
    if len(overlap) < 100 or not np.allclose(frame.loc[overlap, "Close"], formal.loc[overlap],
                                             rtol=1e-6):
        raise ValueError(f"{ticker} 長期間 OHLCV 與正式 OHLCV 的重疊日期不一致")
    return frame


def sample_dates(frame: pd.DataFrame, first: str, last: str) -> list[pd.Timestamp]:
    return [date for date in frame.index if pd.Timestamp(first) <= date <= pd.Timestamp(last)]


def chart_path(output: Path, ticker: str, kind: str, date: pd.Timestamp) -> Path:
    return output.parent / "charts" / ticker / kind / f"{date:%Y-%m-%d}.png"


def render_ticker(job: tuple) -> tuple[str, int]:
    """單一股票的 K 線／Volume／SMA 圖；樣式與正式 chart_generator 相同。已存在的跳過。"""
    ticker, frame, dates, output = job
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    from module_a_data.preprocess.chart_generator import (
        _STYLE, _compute_indicators, _plot_indicator, _save_figure,
    )

    indicators = _compute_indicators(frame)
    position = {date: i for i, date in enumerate(frame.index)}
    made = 0
    for date in dates:
        window = frame.iloc[position[date] - WINDOW + 1:position[date] + 1]
        target = chart_path(output, ticker, "k20", date)
        if not target.exists():
            fig, _ = mpf.plot(window, type="candle", style=_STYLE, volume=False, axisoff=True,
                              returnfig=True, figsize=(3, 3), scale_padding=0)
            _save_figure(fig, target, (224, 224))
            made += 1
        target = chart_path(output, ticker, "volume", date)
        if not target.exists():
            up = window["Close"].to_numpy() >= window["Open"].to_numpy()
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(np.arange(len(window)), window["Volume"].to_numpy(dtype=float),
                   color=np.where(up, "green", "red"), width=0.75)
            ax.set_xlim(-0.5, len(window) - 0.5)
            ax.set_ylim(bottom=0)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            _save_figure(fig, target, (224, 224))
            made += 1
        target = chart_path(output, ticker, "sma", date)
        if not target.exists():
            values = indicators.loc[window.index]
            if values[["sma_5", "sma_10", "sma_20"]].isna().any().any():
                raise ValueError(f"{ticker} {date:%Y-%m-%d} 的 SMA 窗口不完整")
            fig, ax = plt.subplots(figsize=(3, 3))
            _plot_indicator(ax, "sma", values)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            _save_figure(fig, target, (224, 224))
            made += 1
    return ticker, made


def extract_features(tickers, dates, output: Path, model_id: str, device: str, batch: int) -> dict:
    """各股票／圖別的 Frozen CLS [N,768]，與 dates 逐列對齊；已存在的快取直接讀取。"""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    cache = output.parent / "features"
    cache.mkdir(parents=True, exist_ok=True)
    stamp = f"{dates[0]:%Y%m%d}_{dates[-1]:%Y%m%d}_{len(dates)}"
    processor = backbone = None
    features = {}
    for ticker in tickers:
        for kind in KINDS:
            target = cache / f"{ticker}_{kind}_{stamp}.npy"
            if not target.exists():
                if backbone is None:
                    processor = AutoImageProcessor.from_pretrained(model_id)
                    backbone = AutoModel.from_pretrained(model_id).to(device).eval()
                parts = []
                for start in range(0, len(dates), batch):
                    images = []
                    for date in dates[start:start + batch]:
                        with Image.open(chart_path(output, ticker, kind, date)) as image:
                            images.append(image.convert("RGB"))
                    with torch.no_grad():
                        pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                        parts.append(backbone(pixel_values=pixels.to(device))
                                     .last_hidden_state[:, 0].cpu().numpy())
                np.save(target, np.concatenate(parts).astype(np.float32))
                print(f"[horizon_grid:features] {ticker} {kind} {len(dates)}", flush=True)
            features[(ticker, kind)] = np.load(target)
    return features


def build_folds(dates: list[pd.Timestamp], years: list[int]) -> list[dict]:
    """逐年 walk-forward：Validation 為該年樣本，Train 為更早樣本並去掉最後 GAP 天。"""
    index = pd.DatetimeIndex(dates)
    folds = []
    for year in years:
        validation = np.flatnonzero(index.year == year)
        if len(validation) == 0:
            continue
        train = np.arange(0, validation[0] - GAP)
        if len(train) < 200:
            raise ValueError(f"fold {year} 的 Train 不足 200 天")
        folds.append({"name": str(year), "train": train, "validation": validation})
    return folds


def labels_for(returns: np.ndarray, fold: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    lower, upper = np.quantile(returns[fold["train"]], [0.3333, 0.6667])
    label = np.where(returns < lower, 0, np.where(returns > upper, 2, 1))
    return label[fold["train"]], label[fold["validation"]], {"lower": float(lower),
                                                             "upper": float(upper)}


# --- 彙整 ---

def unit_mean(per_unit: dict) -> dict:
    """per_unit: {(fold, ticker): {metric: value}} -> 各指標的單位平均與標準誤。"""
    result = {}
    for key in METRICS:
        values = np.asarray([unit[key] for unit in per_unit.values()], dtype=float)
        result[key] = {"mean": float(values.mean()),
                       "se": float(values.std(ddof=1) / np.sqrt(len(values)))}
    return result


def paired(candidate: dict, baseline: dict, key: str) -> dict:
    deltas = np.asarray([candidate[unit][key] - baseline[unit][key] for unit in candidate])
    return {"mean": float(deltas.mean()), "se": float(deltas.std(ddof=1) / np.sqrt(len(deltas))),
            "wins": int((deltas > 0).sum()), "units": int(len(deltas))}


def write_markdown(report: dict, output: Path) -> None:
    lines = [
        "# Frozen ViT：長期間 × 預測天期 × 圖組合（walk-forward）", "",
        f"- Protocol：`{report['protocol']}`；正式 Test 期間未建立樣本。",
        f"- 股票：{'／'.join(report['tickers'])}；樣本期間 {report['period'][0]} ～ "
        f"{report['period'][1]}（每檔 {report['n_dates']} 天）。",
        f"- Walk-forward folds（驗證年份）：{'、'.join(report['fold_names'])}；"
        f"Validation 合計 {report['n_validation_total']} 筆；統計單位 = fold × 股票，共 "
        f"{report['n_units']} 個。",
        f"- Seeds：{report['seeds']}（先在單位內平均）；分類頭固定 epochs，回報最後一個 epoch。",
        "- 跨天期請比較 ROC-AUC（隨機 0.5）與 balanced accuracy（隨機 0.333），不要比 Macro-F1。",
    ]
    for head in report["heads"]:
        lines += ["", f"## 分類頭 `{head}`", "",
                  "| 天期 | 輸入 | ROC-AUC | Balanced acc | Macro-F1 | Log loss |",
                  "|---|---|---:|---:|---:|---:|"]
        for horizon in report["horizons"]:
            block = report["summary"][head][str(horizon)]
            for name, values in block.items():
                lines.append(
                    f"| t+{horizon} | {name} | {values['roc_auc']['mean']:.4f} ± "
                    f"{values['roc_auc']['se']:.4f} | {values['balanced_accuracy']['mean']:.4f} | "
                    f"{values['macro_f1']['mean']:.4f} | {values['log_loss']['mean']:.4f} |")
            numeric = report["numeric"][str(horizon)]
            lines.append(f"| t+{horizon} | numeric_logistic_w20 | {numeric['roc_auc']['mean']:.4f} "
                         f"± {numeric['roc_auc']['se']:.4f} | "
                         f"{numeric['balanced_accuracy']['mean']:.4f} | "
                         f"{numeric['macro_f1']['mean']:.4f} | {numeric['log_loss']['mean']:.4f} |")
        lines += ["", f"### 配對比較（`{head}`；單位差值平均 ± 標準誤，勝出單位數）", "",
                  "| 天期 | 候選 vs 基準 | ROC-AUC 差 | Macro-F1 差 |", "|---|---|---|---|"]
        for horizon in report["horizons"]:
            for item in report["comparisons"][head][str(horizon)]:
                auc, f1 = item["roc_auc"], item["macro_f1"]
                lines.append(
                    f"| t+{horizon} | {item['candidate']} vs {item['baseline']} | "
                    f"{auc['mean']:+.4f} ± {auc['se']:.4f}（{auc['wins']}/{auc['units']}） | "
                    f"{f1['mean']:+.4f} ± {f1['se']:.4f}（{f1['wins']}/{f1['units']}） |")
    best_head = report["heads"][-1]
    lines += ["", f"## 各 fold 的 ROC-AUC（`{best_head}`，各股票平均）", "",
              "| 天期 | 輸入 | " + " | ".join(report["fold_names"]) + " |",
              "|---|---|" + "---:|" * len(report["fold_names"])]
    for horizon in report["horizons"]:
        for name, by_fold in report["by_fold_auc"][best_head][str(horizon)].items():
            lines.append(f"| t+{horizon} | {name} | " + " | ".join(
                f"{by_fold[fold]:.3f}" for fold in report["fold_names"]) + " |")
    lines += ["", "## 限制", ""] + [f"- {text}" for text in report["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="grid_2016")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "NVDA", "MSFT"])
    parser.add_argument("--start", default="2016-01-01", help="第一個樣本日期")
    parser.add_argument("--download-start", default="2015-09-01", help="窗口與 SMA 暖身用")
    parser.add_argument("--fold-years", type=int, nargs="+",
                        default=[2019, 2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--render-workers", type=int, default=6, help="平行畫圖的行程數")
    parser.add_argument("--model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="工程短測：AAPL、2023 起、2 folds、1 seed；輸出不作效果證據")
    args = parser.parse_args()
    if max(args.horizons) > GAP:
        parser.error(f"horizons 不可超過 Train／Validation 間隔 {GAP}")
    if args.smoke:
        args.run_name, args.tickers, args.start = "smoke", ["AAPL"], "2023-01-01"
        args.download_start, args.fold_years, args.seeds = "2022-09-01", [2024, 2025], [40]

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    output = OUTPUT_ROOT / args.run_name / "run"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # 樣本只到正式 Validation 結束日；正式 Test 期間不建立任何樣本。
    manifest = load_temporal_manifest("AAPL")
    last_sample = manifest["dates"]["validation"][-1]
    download_end = (pd.Timestamp(manifest["dates"]["gap_validation_test"][-1])
                    + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = {t: load_long_ohlcv(t, output, args.download_start, download_end)
              for t in args.tickers}
    dates = sample_dates(frames[args.tickers[0]], args.start, last_sample)
    for ticker, frame in frames.items():
        if sample_dates(frame, args.start, last_sample) != dates:
            raise ValueError(f"{ticker} 的交易日與 {args.tickers[0]} 不一致")
    print(f"[horizon_grid] {len(dates)} 個樣本日期：{dates[0]:%Y-%m-%d} ～ {dates[-1]:%Y-%m-%d}",
          flush=True)

    with ProcessPoolExecutor(max_workers=min(len(args.tickers), args.render_workers)) as pool:
        for ticker, made in pool.map(
                render_ticker, [(t, frames[t], dates, output) for t in args.tickers]):
            print(f"[horizon_grid:charts] {ticker} 新增 {made} 張", flush=True)
    features = extract_features(args.tickers, dates, output, args.model_id, device,
                                args.image_batch)
    folds = build_folds(dates, args.fold_years)

    position = {t: {date: i for i, date in enumerate(frames[t].index)} for t in args.tickers}
    returns = {}
    for ticker in args.tickers:
        close = frames[ticker]["Close"].to_numpy(float)
        index = np.asarray([position[ticker][date] for date in dates])
        for horizon in args.horizons:
            returns[(ticker, horizon)] = close[index + horizon] / close[index] - 1

    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for head in args.heads:
        lr, epochs = HEADS[head]
        for horizon in args.horizons:
            for config, kinds in CONFIGS.items():
                key = f"{head}|{horizon}|{config}"
                if key in results:
                    continue
                block_started = time.perf_counter()
                per_unit = {}
                for fold in folds:
                    for ticker in args.tickers:
                        x = np.concatenate([features[(ticker, kind)] for kind in kinds], axis=1)
                        y_train, y_val, _ = labels_for(returns[(ticker, horizon)], fold)
                        seeds = [metric_payload(y_val, train_head(
                            x[fold["train"]], y_train, x[fold["validation"]], seed, epochs, lr))
                            for seed in args.seeds]
                        per_unit[f"{fold['name']}|{ticker}"] = {
                            metric: float(np.mean([item[metric] for item in seeds]))
                            for metric in METRICS}
                results[key] = per_unit
                results_path.write_text(json.dumps(results), encoding="utf-8")
                print(f"[horizon_grid] {key} 完成（{time.perf_counter() - block_started:.0f} 秒）",
                      flush=True)

    numeric_units, thresholds = {}, {}
    for horizon in args.horizons:
        numeric_units[str(horizon)] = {}
        for fold in folds:
            for ticker in args.tickers:
                y_train, y_val, cut = labels_for(returns[(ticker, horizon)], fold)
                frame, where = frames[ticker], position[ticker]
                windows = [frame.iloc[where[d] - WINDOW + 1:where[d] + 1] for d in dates]
                rows = {"train": {"windows": [windows[i] for i in fold["train"]],
                                  "labels": y_train},
                        "validation": {"windows": [windows[i] for i in fold["validation"]],
                                       "labels": y_val}}
                numeric_units[str(horizon)][f"{fold['name']}|{ticker}"] = numeric_baseline(
                    {"rows": rows}, args.seeds[0])
                thresholds[f"{horizon}|{fold['name']}|{ticker}"] = cut

    summary, comparisons, by_fold_auc = {}, {}, {}
    for head in args.heads:
        summary[head], comparisons[head], by_fold_auc[head] = {}, {}, {}
        for horizon in args.horizons:
            blocks = {config: results[f"{head}|{horizon}|{config}"] for config in CONFIGS}
            summary[head][str(horizon)] = {c: unit_mean(units) for c, units in blocks.items()}
            comparisons[head][str(horizon)] = [
                {"candidate": c, "baseline": b,
                 "roc_auc": paired(blocks[c], blocks[b], "roc_auc"),
                 "macro_f1": paired(blocks[c], blocks[b], "macro_f1")}
                for c, b in COMPARISONS]
            by_fold_auc[head][str(horizon)] = {
                config: {fold["name"]: float(np.mean(
                    [units[f"{fold['name']}|{t}"]["roc_auc"] for t in args.tickers]))
                    for fold in folds}
                for config, units in blocks.items()}

    report = {
        "protocol": PROTOCOL, "smoke": args.smoke, "tickers": args.tickers,
        "period": [f"{dates[0]:%Y-%m-%d}", f"{dates[-1]:%Y-%m-%d}"], "n_dates": len(dates),
        "window_days": WINDOW, "gap_trading_days": GAP, "horizons": args.horizons,
        "fold_names": [fold["name"] for fold in folds],
        "fold_sizes": {fold["name"]: {"train": int(len(fold["train"])),
                                      "validation": int(len(fold["validation"]))}
                       for fold in folds},
        "n_validation_total": int(sum(len(f["validation"]) for f in folds) * len(args.tickers)),
        "n_units": len(folds) * len(args.tickers),
        "configs": CONFIGS, "heads": args.heads,
        "head_settings": {h: {"lr": HEADS[h][0], "epochs": HEADS[h][1]} for h in args.heads},
        "seeds": args.seeds, "model_id": args.model_id, "test_was_opened": False,
        "label_thresholds": thresholds,
        "summary": summary, "comparisons": comparisons, "by_fold_auc": by_fold_auc,
        "numeric": {h: unit_mean(units) for h, units in numeric_units.items()},
        "unit_results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "limitations": [
            "fold × 股票的單位彼此相關（同一年各股票常同漲同跌），標準誤偏樂觀；股票越少越明顯。",
            "2025 fold 與先前多輪實驗使用的 Validation 期間大幅重疊；其餘 folds 是第一次使用。",
            "同時比較 4 種輸入 × 3 種天期 × 2 種分類頭，未做多重比較校正。",
            "不同天期是不同任務；t+1／t+3 的樣本標籤重疊較少，但單日報酬雜訊較大。",
            "只比較 Frozen CLS 串接＋線性頭，不代表正式 Fusion（token 級融合）會有相同結果。",
            "股價為分割調整、未做股利調整的收盤價，與正式 pipeline 一致。",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output)
    print(f"[horizon_grid] report -> {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
