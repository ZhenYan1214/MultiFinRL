"""Frozen ViT：2016 起的長期間 × 預測天期 × 圖組合 × 分類頭設定，walk-forward 驗證。

與先前只有單一 202 天 Validation 的對照不同，這裡用逐年往前推的 walk-forward：
每個 fold 以「該年以前」的全部樣本訓練、在該年驗證（預設 2019～2025 共 7 個 folds），
三檔合計約 5,000 筆 Validation，可分辨的差距約為 0.01。

- 股票：預設 AAPL／NVDA／MSFT，可用 --tickers 擴充。OHLCV 另外下載（yfinance、auto_adjust=False，與正式來源相同）
  到本實驗輸出目錄，並核對與正式 data/raw/ohlcv 重疊日期一致。
- 期間：2016-01 起，到正式 Validation 結束日為止；正式 Test 期間不建立任何樣本。
- 圖：全部以正式 chart_generator 的 mplfinance／matplotlib 樣式重畫（20 日窗口）：
  K 線、Volume、SMA（收盤價線＋SMA5／10／20），以及 kvol（mplfinance 內建成交量面板，
  K 線與成交量畫在同一張圖）；kvol_mono（成交量改為單一顏色）、kvol_frame（kvol 加上
  外框，不含刻度與文字）；k20_tight／kvol_tight（同 k20／kvol，但去掉 mplfinance 預設的
  左右與上下留白，讓 K 棒貼齊圖框）。
- 輸入：預設 single_k20、dual_k20_volume、dual_k20_sma、triple_k20_volume_sma（Frozen CLS
  串接）；`--configs` 可另選 single_kvol。只會畫、只會抽所選設定需要的圖別。
- 標籤（`--label-mode`）：
  tercile      未來第 h 個交易日報酬的三分類，門檻為各 fold、各股票 Train 的 q33／q67（預設）。
  binary       未來第 h 個交易日收盤是否高於今日（漲／不漲）。
  binary_anyup 未來 h 個交易日中任一天收盤是否高於今日（推測為 Chen 2025 的定義，以「漲」為主）。
  Train 與 Validation 之間固定留 5 個交易日，各種 h 使用完全相同的樣本。
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
CONFIGS = {
    "single_k20": ["k20"],
    "dual_k20_volume": ["k20", "volume"],
    "dual_k20_sma": ["k20", "sma"],
    "triple_k20_volume_sma": ["k20", "volume", "sma"],
    "single_kvol": ["kvol"],
    "single_kvol_mono": ["kvol_mono"],
    "single_kvol_frame": ["kvol_frame"],
    "single_k20_tight": ["k20_tight"],
    "single_kvol_tight": ["kvol_tight"],
}
DEFAULT_CONFIGS = ["single_k20", "dual_k20_volume", "dual_k20_sma", "triple_k20_volume_sma"]
HEADS = {"lr1e-3_e30": (1e-3, 30), "lr1e-4_e100": (1e-4, 100)}
COMPARISONS = [("dual_k20_volume", "single_k20"), ("dual_k20_sma", "single_k20"),
               ("triple_k20_volume_sma", "single_k20"), ("dual_k20_sma", "dual_k20_volume"),
               ("triple_k20_volume_sma", "dual_k20_volume"), ("single_kvol", "single_k20"),
               ("single_kvol", "dual_k20_volume"), ("single_kvol_mono", "single_kvol"),
               ("single_kvol_frame", "single_kvol"), ("single_k20_tight", "single_k20"),
               ("single_kvol_tight", "single_kvol")]
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")
LABEL_MODES = ("tercile", "binary", "binary_anyup")
MONO_VOLUME = "#1f77b4"          # mplfinance 預設的成交量顏色


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


def tighten(axes, window: pd.DataFrame, volume: bool) -> None:
    """去掉 mplfinance 預設留白：x 貼齊第一／最後一根 K 棒，價格 y 貼齊最低／最高價。

    成交量面板只去掉上方留白，下限沿用 mplfinance 原本的設定，避免同時改變量的相對高度。
    """
    for ax in axes:
        ax.set_xlim(-0.5, len(window) - 0.5)
    axes[0].set_ylim(float(window["Low"].min()), float(window["High"].max()))
    if volume:
        axes[2].set_ylim(axes[2].get_ylim()[0], float(window["Volume"].max()))


def render_ticker(job: tuple) -> tuple[str, int]:
    """單一股票所需圖別的圖；樣式與正式 chart_generator 相同。已存在的跳過。"""
    ticker, frame, dates, output, kinds = job
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
        for kind, tight in (("k20", False), ("k20_tight", True)):
            target = chart_path(output, ticker, kind, date)
            if kind not in kinds or target.exists():
                continue
            fig, axes = mpf.plot(window, type="candle", style=_STYLE, volume=False,
                                 axisoff=True, returnfig=True, figsize=(3, 3), scale_padding=0)
            if tight:
                tighten(axes, window, volume=False)
            _save_figure(fig, target, (224, 224))
            made += 1
        for kind, volume_color, frame_on in (("kvol", "inherit", False),
                                             ("kvol_mono", MONO_VOLUME, False),
                                             ("kvol_frame", "inherit", True),
                                             ("kvol_tight", "inherit", False)):
            target = chart_path(output, ticker, kind, date)
            if kind not in kinds or target.exists():
                continue
            # kvol／kvol_frame 的成交量沿用正式 Volume 圖的綠漲紅跌；kvol_mono 改為單一顏色。
            style = mpf.make_mpf_style(
                marketcolors=mpf.make_marketcolors(up="green", down="red", edge="inherit",
                                                   wick="inherit", volume=volume_color),
                gridstyle="",
            )
            fig, axes = mpf.plot(window, type="candle", style=style, volume=True,
                                 axisoff=not frame_on, returnfig=True, figsize=(3, 3),
                                 scale_padding=0)
            if kind == "kvol_tight":
                tighten(axes, window, volume=True)
            if frame_on:                 # 只留外框，不留刻度、數字與座標軸標題
                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel("")
                    ax.set_ylabel("")
            _save_figure(fig, target, (224, 224))
            made += 1
        target = chart_path(output, ticker, "volume", date)
        if "volume" in kinds and not target.exists():
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
        if "sma" in kinds and not target.exists():
            values = indicators.loc[window.index]
            if values[["sma_5", "sma_10", "sma_20"]].isna().any().any():
                raise ValueError(f"{ticker} {date:%Y-%m-%d} 的 SMA 窗口不完整")
            fig, ax = plt.subplots(figsize=(3, 3))
            _plot_indicator(ax, "sma", values)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            _save_figure(fig, target, (224, 224))
            made += 1
    return ticker, made


def load_backbone(model_id: str, device: str):
    """優先讀本機快取，避免已下載的模型因 Hugging Face 連線狀態失敗；沒有快取才下載。"""
    from transformers import AutoImageProcessor, AutoModel

    try:
        processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModel.from_pretrained(model_id, local_files_only=True)
    except OSError:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    return processor, model.to(device).eval()


def feature_path(output: Path, ticker: str, kind: str, dates: list[pd.Timestamp]) -> Path:
    stamp = f"{dates[0]:%Y%m%d}_{dates[-1]:%Y%m%d}_{len(dates)}"
    return output.parent / "features" / f"{ticker}_{kind}_{stamp}.npy"


def cached_feature(output: Path, cache_from: list[Path], ticker: str, kind: str,
                   dates: list[pd.Timestamp]) -> Path | None:
    """本 run 或 --cache-from 指定 runs 中已存在的特徵檔（只讀取，不寫入對方目錄）。"""
    for run_output in (output, *cache_from):
        path = feature_path(run_output, ticker, kind, dates)
        if path.exists():
            return path
    return None


def extract_features(tickers, dates, kinds, output: Path, model_id: str, device: str,
                     batch: int, cache_from: list[Path] | None = None) -> dict:
    """各股票／圖別的 Frozen CLS [N,768]，與 dates 逐列對齊；已存在的快取直接讀取。"""
    import torch

    (output.parent / "features").mkdir(parents=True, exist_ok=True)
    processor = backbone = None
    features = {}
    for ticker in tickers:
        for kind in kinds:
            existing = cached_feature(output, cache_from or [], ticker, kind, dates)
            target = existing or feature_path(output, ticker, kind, dates)
            if existing is None:
                if backbone is None:
                    processor, backbone = load_backbone(model_id, device)
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


def labels_for(values: np.ndarray, fold: dict, mode: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """tercile：Train 的 q33／q67 三分類；binary*：values > 0 為 1（漲），否則 0。"""
    if mode == "tercile":
        lower, upper = np.quantile(values[fold["train"]], [0.3333, 0.6667])
        label = np.where(values < lower, 0, np.where(values > upper, 2, 1))
        info = {"lower": float(lower), "upper": float(upper)}
    else:
        label = (values > 0).astype(int)
        info = {"train_up_share": float(label[fold["train"]].mean()),
                "validation_up_share": float(label[fold["validation"]].mean())}
    return label[fold["train"]], label[fold["validation"]], info


def classify_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """三分類沿用 metric_payload；二分類以「漲」的機率計算 ROC-AUC。"""
    if probabilities.shape[1] == 3:
        return metric_payload(labels, probabilities)
    from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss,
                                 roc_auc_score)
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro",
                                   zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def fit_head(x_train, y_train, x_val, seed: int, epochs: int, lr: float,
             n_classes: int) -> np.ndarray:
    """三分類直接用既有 train_head（結果與先前完全相同）；二分類用同結構、同超參數的版本。"""
    if n_classes == 3:
        return train_head(x_train, y_train, x_val, seed, epochs, lr)
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    head = nn.Sequential(nn.LayerNorm(x_train.shape[1]), nn.Dropout(0.1),
                         nn.Linear(x_train.shape[1], n_classes))
    counts = np.bincount(y_train, minlength=n_classes)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(len(y_train) / (n_classes * counts), dtype=torch.float32))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    features, labels = torch.from_numpy(x_train), torch.from_numpy(y_train).long()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        order = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(order), 64):
            index = order[start:start + 64]
            optimizer.zero_grad(set_to_none=True)
            criterion(head(features[index]), labels[index]).backward()
            optimizer.step()
    head.eval()
    with torch.no_grad():
        return torch.softmax(head(torch.from_numpy(x_val)), dim=1).numpy()


def numeric_metrics(windows: list[pd.DataFrame], fold: dict, y_train: np.ndarray,
                    y_val: np.ndarray, seed: int) -> dict:
    """不看圖的對照；三分類沿用 numeric_baseline，二分類用相同特徵與模型。"""
    rows = {"train": {"windows": [windows[i] for i in fold["train"]], "labels": y_train},
            "validation": {"windows": [windows[i] for i in fold["validation"]],
                           "labels": y_val}}
    if len(np.unique(y_train)) == 3:
        return numeric_baseline({"rows": rows}, seed)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def numeric(window: pd.DataFrame) -> np.ndarray:
        close = window["Close"].to_numpy(float)
        volume = window["Volume"].to_numpy(float)
        return np.concatenate([np.diff(np.log(close)), volume / volume.mean() - 1])

    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=seed))
    model.fit(np.stack([numeric(w) for w in rows["train"]["windows"]]), y_train)
    return classify_metrics(y_val, model.predict_proba(
        np.stack([numeric(w) for w in rows["validation"]["windows"]])))


def majority_metrics(y_train: np.ndarray, y_val: np.ndarray, n_classes: int) -> dict:
    """永遠猜 Train 多數類；ROC-AUC 固定 0.5，log loss 以 Train 類別比例計算。"""
    from sklearn.metrics import log_loss
    share = np.bincount(y_train, minlength=n_classes) / len(y_train)
    majority = int(share.argmax())
    probabilities = np.tile(np.clip(share, 1e-6, 1), (len(y_val), 1))
    return {
        "accuracy": float((y_val == majority).mean()),
        "balanced_accuracy": 1.0 / n_classes,
        "macro_f1": float(np.mean([
            (2 * (y_val == c).mean() / (1 + (y_val == c).mean())) if c == majority else 0.0
            for c in range(n_classes)])),
        "roc_auc": 0.5,
        "log_loss": float(log_loss(y_val, probabilities / probabilities.sum(1, keepdims=True),
                                   labels=list(range(n_classes)))),
    }


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
        f"- 標籤：`{report.get('label_mode', 'tercile')}`（{report.get('n_classes', 3)} 類）。"
        "majority 為永遠猜 Train 多數類的對照；accuracy 受類別比例影響，請與 majority 一起看。",
        f"- 跨天期請比較 ROC-AUC（隨機 0.5）與 balanced accuracy（隨機 "
        f"{1 / report.get('n_classes', 3):.3f}），不要比 Macro-F1。",
    ]
    for head in report["heads"]:
        lines += ["", f"## 分類頭 `{head}`", "",
                  "| 天期 | 輸入 | ROC-AUC | Balanced acc | Accuracy | Macro-F1 | Log loss |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for horizon in report["horizons"]:
            rows = dict(report["summary"][head][str(horizon)])
            rows["numeric_logistic_w20"] = report["numeric"][str(horizon)]
            if "majority" in report:
                rows["majority"] = report["majority"][str(horizon)]
            for name, values in rows.items():
                lines.append(
                    f"| t+{horizon} | {name} | {values['roc_auc']['mean']:.4f} ± "
                    f"{values['roc_auc']['se']:.4f} | {values['balanced_accuracy']['mean']:.4f} | "
                    f"{values['accuracy']['mean']:.4f} | {values['macro_f1']['mean']:.4f} | "
                    f"{values['log_loss']['mean']:.4f} |")
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
    # 預設只跑 t+5（2026-10-17 起）；t+1／t+3 已測過且沒有差異，需要時再以 --horizons 指定。
    parser.add_argument("--horizons", type=int, nargs="+", default=[5])
    parser.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list(CONFIGS))
    parser.add_argument("--label-mode", default="tercile", choices=LABEL_MODES)
    parser.add_argument("--cache-from", nargs="+", default=[],
                        help="讀取其他 run 已抽好的特徵（只讀取），避免重畫與重抽")
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

    configs = {name: CONFIGS[name] for name in args.configs}
    kinds = sorted({kind for kinds_ in configs.values() for kind in kinds_})
    pairs = [(c, b) for c, b in COMPARISONS if c in configs and b in configs]
    cache_from = [OUTPUT_ROOT / name / "run" for name in args.cache_from]
    jobs = []
    for t in args.tickers:   # 已有特徵的圖別不必再畫
        missing = [k for k in kinds if cached_feature(output, cache_from, t, k, dates) is None]
        jobs.append((t, frames[t], dates, output, missing))
    with ProcessPoolExecutor(max_workers=min(len(args.tickers), args.render_workers)) as pool:
        for ticker, made in pool.map(render_ticker, jobs):
            print(f"[horizon_grid:charts] {ticker} 新增 {made} 張", flush=True)
    features = extract_features(args.tickers, dates, kinds, output, args.model_id, device,
                                args.image_batch, cache_from)
    folds = build_folds(dates, args.fold_years)

    position = {t: {date: i for i, date in enumerate(frames[t].index)} for t in args.tickers}
    returns = {}
    for ticker in args.tickers:
        close = frames[ticker]["Close"].to_numpy(float)
        index = np.asarray([position[ticker][date] for date in dates])
        for horizon in args.horizons:
            if args.label_mode == "binary_anyup":
                future = np.max([close[index + k] for k in range(1, horizon + 1)], axis=0)
            else:
                future = close[index + horizon]
            returns[(ticker, horizon)] = future / close[index] - 1
    n_classes = 3 if args.label_mode == "tercile" else 2

    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for head in args.heads:
        lr, epochs = HEADS[head]
        for horizon in args.horizons:
            for config, config_kinds in configs.items():
                key = f"{head}|{horizon}|{config}"
                if key in results:
                    continue
                block_started = time.perf_counter()
                per_unit = {}
                for fold in folds:
                    for ticker in args.tickers:
                        x = np.concatenate([features[(ticker, kind)] for kind in config_kinds],
                                           axis=1)
                        y_train, y_val, _ = labels_for(returns[(ticker, horizon)], fold,
                                                       args.label_mode)
                        seeds = [classify_metrics(y_val, fit_head(
                            x[fold["train"]], y_train, x[fold["validation"]], seed, epochs, lr,
                            n_classes)) for seed in args.seeds]
                        per_unit[f"{fold['name']}|{ticker}"] = {
                            metric: float(np.mean([item[metric] for item in seeds]))
                            for metric in METRICS}
                results[key] = per_unit
                results_path.write_text(json.dumps(results), encoding="utf-8")
                print(f"[horizon_grid] {key} 完成（{time.perf_counter() - block_started:.0f} 秒）",
                      flush=True)

    numeric_units, majority_units, thresholds = {}, {}, {}
    windows = {t: [frames[t].iloc[position[t][d] - WINDOW + 1:position[t][d] + 1] for d in dates]
               for t in args.tickers}
    for horizon in args.horizons:
        numeric_units[str(horizon)], majority_units[str(horizon)] = {}, {}
        for fold in folds:
            for ticker in args.tickers:
                y_train, y_val, cut = labels_for(returns[(ticker, horizon)], fold,
                                                 args.label_mode)
                unit = f"{fold['name']}|{ticker}"
                numeric_units[str(horizon)][unit] = numeric_metrics(
                    windows[ticker], fold, y_train, y_val, args.seeds[0])
                majority_units[str(horizon)][unit] = majority_metrics(y_train, y_val, n_classes)
                thresholds[f"{horizon}|{unit}"] = cut

    summary, comparisons, by_fold_auc = {}, {}, {}
    for head in args.heads:
        summary[head], comparisons[head], by_fold_auc[head] = {}, {}, {}
        for horizon in args.horizons:
            blocks = {config: results[f"{head}|{horizon}|{config}"] for config in configs}
            summary[head][str(horizon)] = {c: unit_mean(units) for c, units in blocks.items()}
            comparisons[head][str(horizon)] = [
                {"candidate": c, "baseline": b,
                 "roc_auc": paired(blocks[c], blocks[b], "roc_auc"),
                 "macro_f1": paired(blocks[c], blocks[b], "macro_f1")}
                for c, b in pairs]
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
        "configs": configs, "heads": args.heads,
        "head_settings": {h: {"lr": HEADS[h][0], "epochs": HEADS[h][1]} for h in args.heads},
        "seeds": args.seeds, "model_id": args.model_id, "test_was_opened": False,
        "label_thresholds": thresholds,
        "summary": summary, "comparisons": comparisons, "by_fold_auc": by_fold_auc,
        "numeric": {h: unit_mean(units) for h, units in numeric_units.items()},
        "majority": {h: unit_mean(units) for h, units in majority_units.items()},
        "label_mode": args.label_mode, "n_classes": n_classes,
        "cache_from": args.cache_from,
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
