"""Frozen ViT：第二張圖改用 SMA 圖（K20＋SMA）對照現行 K20＋Volume。

與 `run_k20_k60_comparison.py` 使用完全相同的樣本（926 Train／202 Validation）、標籤、
凍結的 ViT-B/16、分類頭設定與 seeds，因此可與該實驗的數字直接並列：

- single_k20      ：20 日 K 線
- dual_k20_volume ：20 日 K 線＋Volume（正式 pipeline 現行雙圖）
- dual_k20_k60    ：20 日 K 線＋60 日 K 線
- dual_k20_sma    ：20 日 K 線＋SMA 圖（收盤價線＋SMA5／10／20，正式 chart_generator 的樣式）

K20／Volume／K60 的 Frozen CLS 直接讀 K20＋K60 實驗的快取，只新畫 SMA 圖並抽其特徵；
SMA 圖畫到本實驗的輸出目錄，不寫入 data/raw/charts。Test 完全不抽特徵、不評估。

用法（需先跑過 run_k20_k60_comparison.py）：
    python experiments/vision/run_k20_sma_comparison.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import paths
from shared.temporal_split import load_temporal_manifest
from experiments.vision import run_k20_k60_comparison as base
from experiments.vision.run_pixel_render_comparison import (
    SPLITS, TICKERS, load_ohlcv, metric_payload, stats, train_head,
)

PROTOCOL = "vit_k20_sma_comparison_v1"
OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "vit_k20_sma"
BASE_FEATURES = base.OUTPUT_ROOT / "k20_k60" / "features"
CONFIGS = {
    "single_k20": ["k20"],
    "dual_k20_volume": ["k20", "volume"],
    "dual_k20_k60": ["k20", "k60"],
    "dual_k20_sma": ["k20", "sma"],
}
COMPARISONS = [("dual_k20_sma", "dual_k20_volume"), ("dual_k20_sma", "single_k20"),
               ("dual_k20_sma", "dual_k20_k60")]
METRICS = base.METRICS
GATE_DELTA, GATE_WINS = base.GATE_DELTA, base.GATE_WINS


def sma_path(output: Path, ticker: str, date: str) -> Path:
    return output / "charts" / ticker / "sma" / f"{date}.png"


def render_sma_charts(data: dict, output: Path) -> None:
    """用正式 chart_generator 的 SMA panel 樣式畫 20 日窗口；已存在的跳過。"""
    import matplotlib.pyplot as plt
    from module_a_data.preprocess.chart_generator import (
        _compute_indicators, _plot_indicator, _save_figure,
    )

    for ticker, payload in data.items():
        computed = _compute_indicators(load_ohlcv(ticker))
        made = 0
        for split in SPLITS:
            rows = payload["rows"][split]
            for date, window in zip(rows["dates"], rows["windows"]):
                target = sma_path(output, ticker, date)
                if target.exists():
                    continue
                values = computed.loc[window.index]
                if values[["sma_5", "sma_10", "sma_20"]].isna().any().any():
                    raise ValueError(f"{ticker} {date} 的 SMA 窗口不完整")
                fig, ax = plt.subplots(figsize=(3, 3))
                _plot_indicator(ax, "sma", values)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                _save_figure(fig, target, (224, 224))
                made += 1
        print(f"[k20_sma:charts] {ticker} 新增 {made} 張 SMA", flush=True)


def save_samples(data: dict, output: Path, n: int = 3) -> list[str]:
    """每檔挑 Train 期間等距的 n 天，輸出 K20｜SMA｜Volume 並排對照圖。"""
    out_dir = output / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ticker, payload in data.items():
        dates = payload["rows"]["train"]["dates"]
        for i in np.linspace(0, len(dates) - 1, n, dtype=int):
            sheet = Image.new("RGB", (224 * 3 + 16, 224 + 8), (128, 128, 128))
            tiles = [paths.RAW_CHARTS / ticker / f"{dates[i]}.png",
                     sma_path(output, ticker, dates[i]),
                     paths.RAW_CHARTS / ticker / "volume" / f"{dates[i]}.png"]
            for position, tile_path in enumerate(tiles):
                with Image.open(tile_path) as tile:
                    sheet.paste(tile.convert("RGB"), (4 + position * 228, 4))
            target = out_dir / f"{ticker}_{dates[i]}.png"
            sheet.save(target)
            saved.append(str(target.relative_to(ROOT)))
    return saved


def load_features(data: dict, output: Path, model_id: str, device: str, batch: int) -> dict:
    """K20／Volume／K60 讀既有快取並核對筆數；SMA 新抽 Frozen CLS。"""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    cache = output / "features"
    cache.mkdir(parents=True, exist_ok=True)
    processor = backbone = None
    features = {}
    for ticker, payload in data.items():
        for split in SPLITS:
            dates = payload["rows"][split]["dates"]
            for kind in ("k20", "volume", "k60"):
                source = BASE_FEATURES / f"{ticker}_{kind}_{split}.npy"
                if not source.exists():
                    raise SystemExit(f"找不到 {source}，先跑 run_k20_k60_comparison.py")
                array = np.load(source)
                if array.shape != (len(dates), 768):
                    raise ValueError(f"{source} 筆數與目前樣本不一致：{array.shape}")
                features[(ticker, kind, split)] = array
            target = cache / f"{ticker}_sma_{split}.npy"
            if not target.exists():
                if backbone is None:
                    processor = AutoImageProcessor.from_pretrained(model_id)
                    backbone = AutoModel.from_pretrained(model_id).to(device).eval()
                parts = []
                for start in range(0, len(dates), batch):
                    images = []
                    for date in dates[start:start + batch]:
                        with Image.open(sma_path(output, ticker, date)) as image:
                            images.append(image.convert("RGB"))
                    with torch.no_grad():
                        pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                        parts.append(backbone(pixel_values=pixels.to(device))
                                     .last_hidden_state[:, 0].cpu().numpy())
                np.save(target, np.concatenate(parts).astype(np.float32))
                print(f"[k20_sma:features] {ticker} sma {split}", flush=True)
            features[(ticker, "sma", split)] = np.load(target)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="k20_sma")
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    output = OUTPUT_ROOT / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = load_temporal_manifest("AAPL")
    data = {ticker: base.build_rows(ticker, manifest) for ticker in TICKERS}
    render_sma_charts(data, output)
    samples = save_samples(data, output)
    features = load_features(data, output, args.model_id, device, args.image_batch)

    def stacked(ticker: str, config: str, split: str) -> np.ndarray:
        return np.concatenate([features[(ticker, kind, split)] for kind in CONFIGS[config]], axis=1)

    results = {name: {} for name in CONFIGS}
    for name in CONFIGS:
        for seed in args.seeds:
            by_ticker = {}
            for ticker in TICKERS:
                rows = data[ticker]["rows"]
                probabilities = train_head(
                    stacked(ticker, name, "train"), rows["train"]["labels"],
                    stacked(ticker, name, "validation"), seed, args.epochs, args.head_lr)
                by_ticker[ticker] = metric_payload(rows["validation"]["labels"], probabilities)
            results[name][str(seed)] = {
                "by_ticker": by_ticker,
                "mean_ticker": {key: float(np.mean([by_ticker[t][key] for t in TICKERS]))
                                for key in METRICS},
            }
        print(f"[k20_sma] {name} 完成", flush=True)

    summary = {}
    for name, by_seed in results.items():
        summary[name] = {key: stats([item["mean_ticker"][key] for item in by_seed.values()])
                         for key in METRICS}
        summary[name]["by_ticker"] = {
            ticker: {key: float(np.mean([item["by_ticker"][ticker][key]
                                         for item in by_seed.values()]))
                     for key in ("macro_f1", "roc_auc")}
            for ticker in TICKERS
        }
    comparisons = []
    for candidate, baseline in COMPARISONS:
        deltas = [results[candidate][str(seed)]["mean_ticker"]["macro_f1"]
                  - results[baseline][str(seed)]["mean_ticker"]["macro_f1"] for seed in args.seeds]
        wins = int(sum(delta > 0 for delta in deltas))
        comparisons.append({
            "candidate": candidate, "baseline": baseline,
            "delta_macro_f1": stats(deltas), "per_seed_delta": deltas, "wins": wins,
            "delta_roc_auc": summary[candidate]["roc_auc"]["mean"]
                             - summary[baseline]["roc_auc"]["mean"],
            "gate_passed": bool(np.mean(deltas) >= GATE_DELTA and wins >= GATE_WINS),
        })

    numeric = {t: base.numeric_for_window(data[t], "windows", args.seeds[0]) for t in TICKERS}
    report = {
        "protocol": PROTOCOL, "tickers": TICKERS, "short_window": base.SHORT,
        "long_window": base.LONG, "model_id": args.model_id, "seeds": args.seeds,
        "epochs": args.epochs, "head_lr": args.head_lr, "configs": CONFIGS,
        "split_counts": {s: len(data[TICKERS[0]]["rows"][s]["dates"]) for s in SPLITS},
        "dropped": data[TICKERS[0]]["dropped"],
        "split_manifest": str(paths.temporal_split_path("AAPL").relative_to(ROOT)),
        "label_thresholds": {ticker: data[ticker]["thresholds"] for ticker in TICKERS},
        "gate": {"min_delta_macro_f1": GATE_DELTA, "min_wins": GATE_WINS},
        "test_was_opened": False,
        "results": results, "summary": summary, "comparisons": comparisons,
        "numeric_baselines": {"numeric_logistic_w20": {
            "by_ticker": numeric,
            "mean_ticker": {key: float(np.mean([numeric[t][key] for t in TICKERS]))
                            for key in METRICS},
        }},
        "samples": samples,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "limitations": [
            "AAPL／NVDA／MSFT 的這段 Validation 已在先前多個實驗中被看過，結果可能偏樂觀。",
            "每檔約 926 筆 Train、202 筆 Validation；先前觀察到同一設定因 Train 少 40 天變動 0.037。",
            "同一份 Validation 已比較過像素畫法與 K60，這是第三輪比較，未做多重比較校正。",
            "MSFT 沒有正式 manifest，這裡以 AAPL 的日期切分與 MSFT 自己的 Train 報酬計算標籤。",
            "只比較 Frozen CLS 串接＋線性頭，不代表正式 Fusion（token 級融合）會有相同結果。",
            "樣本沿用 K20＋K60 實驗（剔除 Train 最前面 40 天），以便數字直接並列。",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    base.write_markdown(report, output)
    text = (output / "report.md").read_text(encoding="utf-8").replace(
        "# Frozen ViT：K20＋K60 vs K20＋Volume", "# Frozen ViT：K20＋SMA vs K20＋Volume").replace(
        "因不足 60 根 K 棒而剔除", "沿用 K20＋K60 實驗的共同樣本而剔除")
    (output / "report.md").write_text(text, encoding="utf-8")
    print(f"[k20_sma] report -> {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
