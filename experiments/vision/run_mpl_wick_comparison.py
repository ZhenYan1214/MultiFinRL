"""Frozen ViT：直接在正式 mplfinance K 線圖上加粗影線的對照。

與正式 chart_generator 完全相同的畫法、樣式與縮圖流程（3×3 吋、dpi 100、tight bbox、
LANCZOS 縮到 224×224），只在 mplfinance 畫完後修改影線的線寬與顏色，實體不動：

- mpl          ：正式 pipeline 現有圖（影線 1pt，縮圖後約 1 像素）。
- mpl_wick3x   ：影線線寬 3 倍、顏色不變。
- mpl_range    ：影線線寬等於實體寬度，改用同色系淺色（上下影線變成與實體同寬的淺色塊）。

樣本、標籤、凍結 ViT、分類頭與 seeds 沿用像素畫法實驗（每檔 966 Train／202 Validation），
mpl 與 Volume 的 Frozen CLS 直接讀其快取；雙圖一律串接正式 Volume CLS。新圖畫到本實驗
輸出目錄，不寫入 data/raw/charts。Test 不評估。

用法（需先跑過 run_pixel_render_comparison.py）：
    python experiments/vision/run_mpl_wick_comparison.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import paths
from shared.temporal_split import load_temporal_manifest
from experiments.vision import run_pixel_render_comparison as base
from experiments.vision.run_pixel_render_comparison import (
    SPLITS, TICKERS, metric_payload, numeric_baseline, stats, train_head,
)

PROTOCOL = "vit_mpl_wick_comparison_v1"
OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "vit_mpl_wick"
BASE_FEATURES = base.OUTPUT_ROOT / "px_vs_mpl_w20" / "features"
NEW_KINDS = ("mpl_wick3x", "mpl_range")
CONFIGS = {
    "mpl_single": ["mpl_candlestick"],
    "wick3x_single": ["mpl_wick3x"],
    "range_single": ["mpl_range"],
    "mpl_dual": ["mpl_candlestick", "mpl_volume"],
    "wick3x_dual": ["mpl_wick3x", "mpl_volume"],
    "range_dual": ["mpl_range", "mpl_volume"],
}
COMPARISONS = [("wick3x_single", "mpl_single"), ("range_single", "mpl_single"),
               ("wick3x_dual", "mpl_dual"), ("range_dual", "mpl_dual")]
LIGHT = {"up": (150 / 255, 210 / 255, 150 / 255), "down": (1.0, 160 / 255, 160 / 255)}


def chart_path(output: Path, ticker: str, kind: str, date: str) -> Path:
    return output / "charts" / ticker / kind / f"{date}.png"


def render(window: pd.DataFrame, kind: str, target: Path) -> None:
    import mplfinance as mpf
    from matplotlib.collections import LineCollection, PolyCollection
    from module_a_data.preprocess.chart_generator import _STYLE, _save_figure

    fig, axes = mpf.plot(
        window, type="candle", style=_STYLE, volume=False, axisoff=True,
        returnfig=True, figsize=(3, 3), scale_padding=0,
    )
    ax = axes[0]
    wicks = next(c for c in ax.collections if isinstance(c, LineCollection))
    bodies = next(c for c in ax.collections if isinstance(c, PolyCollection))
    wicks.set_capstyle("butt")            # 加粗後端點不可延伸進實體或超出最高／最低價
    if kind == "mpl_wick3x":
        wicks.set_linewidth(3 * float(wicks.get_linewidths()[0]))
    elif kind == "mpl_range":
        vertices = bodies.get_paths()[0].vertices
        body_units = float(vertices[:, 0].max() - vertices[:, 0].min())
        x0, x1 = ax.get_xlim()
        points_per_unit = ax.get_window_extent().width / (x1 - x0) * 72 / fig.dpi
        wicks.set_linewidth(body_units * points_per_unit)
        wicks.set_color([LIGHT["up"] if color[1] > color[0] else LIGHT["down"]
                         for color in wicks.get_colors()])
    else:
        raise ValueError(f"不支援的圖別：{kind}")
    _save_figure(fig, target, (224, 224))


def render_charts(data: dict, output: Path) -> None:
    for ticker, payload in data.items():
        made = 0
        for kind in NEW_KINDS:
            for split in SPLITS:
                rows = payload["rows"][split]
                for date, window in zip(rows["dates"], rows["windows"]):
                    target = chart_path(output, ticker, kind, date)
                    if not target.exists():
                        render(window, kind, target)
                        made += 1
        print(f"[mpl_wick:charts] {ticker} 新增 {made} 張", flush=True)


def save_samples(data: dict, output: Path, n: int = 3) -> list[str]:
    """每檔挑 Train 等距 n 天：mpl｜mpl_wick3x｜mpl_range 並排。"""
    out_dir = output / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ticker, payload in data.items():
        dates = payload["rows"]["train"]["dates"]
        for i in np.linspace(0, len(dates) - 1, n, dtype=int):
            tiles = [paths.RAW_CHARTS / ticker / f"{dates[i]}.png"] + [
                chart_path(output, ticker, kind, dates[i]) for kind in NEW_KINDS]
            sheet = Image.new("RGB", (224 * 3 + 16, 224 + 8), (128, 128, 128))
            for position, tile_path in enumerate(tiles):
                with Image.open(tile_path) as tile:
                    sheet.paste(tile.convert("RGB"), (4 + position * 228, 4))
            target = out_dir / f"{ticker}_{dates[i]}.png"
            sheet.save(target)
            saved.append(str(target.relative_to(ROOT)))
    return saved


def load_features(data: dict, output: Path, model_id: str, device: str, batch: int) -> dict:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    cache = output / "features"
    cache.mkdir(parents=True, exist_ok=True)
    processor = backbone = None
    features = {}
    for ticker, payload in data.items():
        for split in SPLITS:
            dates = payload["rows"][split]["dates"]
            for kind in ("mpl_candlestick", "mpl_volume"):
                source = BASE_FEATURES / f"{ticker}_{kind}_{split}.npy"
                if not source.exists():
                    raise SystemExit(f"找不到 {source}，先跑 run_pixel_render_comparison.py")
                array = np.load(source)
                if array.shape != (len(dates), 768):
                    raise ValueError(f"{source} 筆數與目前樣本不一致：{array.shape}")
                features[(ticker, kind, split)] = array
            for kind in NEW_KINDS:
                target = cache / f"{ticker}_{kind}_{split}.npy"
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
                    print(f"[mpl_wick:features] {ticker} {kind} {split}", flush=True)
                features[(ticker, kind, split)] = np.load(target)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="mpl_wick_w20")
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
    data = {ticker: base.build_rows(ticker, manifest, 20, 5) for ticker in TICKERS}
    render_charts(data, output)
    samples = save_samples(data, output)
    features = load_features(data, output, args.model_id, device, args.image_batch)

    def stacked(ticker: str, config: str, split: str) -> np.ndarray:
        return np.concatenate([features[(ticker, kind, split)] for kind in CONFIGS[config]], axis=1)

    keys = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")
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
                "mean_ticker": {k: float(np.mean([by_ticker[t][k] for t in TICKERS])) for k in keys},
            }
        print(f"[mpl_wick] {name} 完成", flush=True)

    summary = {}
    for name, by_seed in results.items():
        summary[name] = {k: stats([item["mean_ticker"][k] for item in by_seed.values()])
                         for k in keys}
        summary[name]["by_ticker"] = {
            t: float(np.mean([item["by_ticker"][t]["macro_f1"] for item in by_seed.values()]))
            for t in TICKERS
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
            "gate_passed": bool(np.mean(deltas) >= base.GATE_DELTA and wins >= base.GATE_WINS),
        })

    numeric = {ticker: numeric_baseline(data[ticker], args.seeds[0]) for ticker in TICKERS}
    report = {
        "protocol": PROTOCOL, "tickers": TICKERS, "window_days": 20,
        "model_id": args.model_id, "seeds": args.seeds, "epochs": args.epochs,
        "head_lr": args.head_lr, "configs": CONFIGS,
        "split_counts": {split: len(manifest["dates"][split]) for split in SPLITS},
        "split_manifest": str(paths.temporal_split_path("AAPL").relative_to(ROOT)),
        "gate": {"min_delta_macro_f1": base.GATE_DELTA, "min_wins": base.GATE_WINS},
        "test_was_opened": False,
        "results": results, "summary": summary, "comparisons": comparisons,
        "numeric_baseline": {
            "by_ticker": numeric,
            "mean_ticker": {k: float(np.mean([numeric[t][k] for t in TICKERS]))
                            for k in ("macro_f1", "balanced_accuracy", "roc_auc", "log_loss")},
        },
        "samples": samples,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "limitations": [
            "AAPL／NVDA／MSFT 的這段 Validation 已在先前多輪實驗中被看過，未做多重比較校正。",
            "每檔 966 筆 Train、202 筆 Validation；先前觀察到同一設定因樣本小幅變動可差 0.037。",
            "MSFT 沒有正式 manifest，這裡以 AAPL 的日期切分與 MSFT 自己的 Train 報酬計算標籤。",
            "只比較 Frozen CLS＋線性頭，不代表正式 Fusion（token 級融合）會有相同結果。",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    base.write_markdown(report, output)
    text = (output / "report.md").read_text(encoding="utf-8").replace(
        "# Frozen ViT：matplotlib vs 像素精確畫法", "# Frozen ViT：mplfinance K 線影線加粗對照").replace(
        "## 配對比較（候選 − mpl，", "## 配對比較（候選 − 基準，")
    (output / "report.md").write_text(text, encoding="utf-8")
    print(f"[mpl_wick] report -> {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
