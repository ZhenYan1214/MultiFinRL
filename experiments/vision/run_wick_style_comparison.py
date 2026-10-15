"""Frozen ViT：K 線影線加粗、換色的畫法對照。

沿用像素畫法實驗（run_pixel_render_comparison.py）的樣本、標籤、凍結 ViT、分類頭與 seeds，
並直接讀取其 Frozen CLS 快取；本實驗只新增兩種影線樣式的 K 線圖：

- mpl       ：正式 pipeline 的 mplfinance 圖（參考組）。
- pxw       ：白底像素圖，影線 1 像素、與實體同色（對照基準：版面與下面兩種完全相同）。
- pxw_wick3 ：影線加粗為 3 像素，改用同色系淺色（漲淺綠、跌淺紅）。
- pxw_range ：最高～最低畫成與實體同寬的淺色棒，開盤～收盤疊上深色實體。

雙圖一律串接同一份白底像素 Volume CLS（mpl 參考組用正式 Volume 圖）。Test 不評估。

用法（需先跑過 run_pixel_render_comparison.py）：
    python experiments/vision/run_wick_style_comparison.py
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

PROTOCOL = "vit_wick_style_comparison_v1"
OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "vit_wick_style"
BASE_FEATURES = base.OUTPUT_ROOT / "px_vs_mpl_w20" / "features"
NEW_KINDS = {"pxw_wick3": 3, "pxw_range": None}   # 影線寬度（像素）；None = 與實體同寬
CONFIGS = {
    "mpl_single": ["mpl_candlestick"],
    "pxw_single": ["pxw_candlestick"],
    "wick3_single": ["pxw_wick3"],
    "range_single": ["pxw_range"],
    "mpl_dual": ["mpl_candlestick", "mpl_volume"],
    "pxw_dual": ["pxw_candlestick", "pxw_volume"],
    "wick3_dual": ["pxw_wick3", "pxw_volume"],
    "range_dual": ["pxw_range", "pxw_volume"],
}
COMPARISONS = [
    ("wick3_single", "pxw_single"), ("range_single", "pxw_single"),
    ("wick3_single", "mpl_single"), ("range_single", "mpl_single"),
    ("wick3_dual", "pxw_dual"), ("range_dual", "pxw_dual"),
    ("wick3_dual", "mpl_dual"), ("range_dual", "mpl_dual"),
]
BODY = {True: (0, 128, 0), False: (255, 0, 0)}            # 與 mplfinance 的 green／red 相同
WICK = {True: (150, 210, 150), False: (255, 160, 160)}    # 同色系淺色


def render_candles_wick(window: pd.DataFrame, wick_width: int | None,
                        size: int = 224, pad: int = 4) -> np.ndarray:
    """白底 K 線；版面與 render_candles_px 相同，只改影線寬度與顏色。"""
    col, x0, body = base._layout(len(window), size)
    width = body if wick_width is None else min(wick_width, body)
    lo, hi = float(window["Low"].min()), float(window["High"].max())
    span = max(hi - lo, 1e-9)

    def y(price: float) -> int:
        return int(round((size - 1 - pad) - (price - lo) / span * (size - 1 - 2 * pad)))

    img = np.full((size, size, 3), 255, dtype=np.uint8)
    for i, (o, h, l, c) in enumerate(window[["Open", "High", "Low", "Close"]].to_numpy(float)):
        left = x0 + i * col
        up = bool(c >= o)
        wick_left = left + (body - width) // 2
        img[y(h):y(l) + 1, wick_left:wick_left + width] = WICK[up]
        top, bottom = sorted((y(o), y(c)))
        img[top:bottom + 1, left:left + body] = BODY[up]     # 實體蓋在影線上
    return img


def new_image(kind: str, window: pd.DataFrame) -> Image.Image:
    return Image.fromarray(render_candles_wick(window, NEW_KINDS[kind]))


def save_samples(data: dict, output: Path, n: int = 3) -> list[str]:
    """每檔挑 Train 等距 n 天：mpl｜pxw｜pxw_wick3｜pxw_range 並排。"""
    out_dir = output / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ticker, payload in data.items():
        train = payload["rows"]["train"]
        for i in np.linspace(0, len(train["dates"]) - 1, n, dtype=int):
            date, window = train["dates"][i], train["windows"][i]
            tiles = [base.load_image(ticker, "mpl_candlestick", date, window),
                     base.load_image(ticker, "pxw_candlestick", date, window),
                     new_image("pxw_wick3", window), new_image("pxw_range", window)]
            sheet = Image.new("RGB", (224 * 4 + 20, 224 + 8), (128, 128, 128))
            for position, tile in enumerate(tiles):
                sheet.paste(tile, (4 + position * 228, 4))
            target = out_dir / f"{ticker}_{date}.png"
            sheet.save(target)
            saved.append(str(target.relative_to(ROOT)))
    return saved


def load_features(data: dict, output: Path, model_id: str, device: str, batch: int) -> dict:
    """既有圖別讀像素畫法實驗的快取並核對筆數；兩種新影線樣式新抽 Frozen CLS。"""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    cache = output / "features"
    cache.mkdir(parents=True, exist_ok=True)
    cached_kinds = sorted({k for kinds in CONFIGS.values() for k in kinds} - set(NEW_KINDS))
    processor = backbone = None
    features = {}
    for ticker, payload in data.items():
        for split in SPLITS:
            rows = payload["rows"][split]
            for kind in cached_kinds:
                source = BASE_FEATURES / f"{ticker}_{kind}_{split}.npy"
                if not source.exists():
                    raise SystemExit(f"找不到 {source}，先跑 run_pixel_render_comparison.py")
                array = np.load(source)
                if array.shape != (len(rows["dates"]), 768):
                    raise ValueError(f"{source} 筆數與目前樣本不一致：{array.shape}")
                features[(ticker, kind, split)] = array
            for kind in NEW_KINDS:
                target = cache / f"{ticker}_{kind}_{split}.npy"
                if not target.exists():
                    if backbone is None:
                        processor = AutoImageProcessor.from_pretrained(model_id)
                        backbone = AutoModel.from_pretrained(model_id).to(device).eval()
                    parts = []
                    for start in range(0, len(rows["windows"]), batch):
                        images = [new_image(kind, w) for w in rows["windows"][start:start + batch]]
                        with torch.no_grad():
                            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                            parts.append(backbone(pixel_values=pixels.to(device))
                                         .last_hidden_state[:, 0].cpu().numpy())
                    np.save(target, np.concatenate(parts).astype(np.float32))
                    print(f"[wick_style:features] {ticker} {kind} {split}", flush=True)
                features[(ticker, kind, split)] = np.load(target)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="wick_w20")
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
        print(f"[wick_style] {name} 完成", flush=True)

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
        "wick_styles": {"pxw_wick3": "3 像素、淺色", "pxw_range": "與實體同寬、淺色"},
        "colors": {"body": {"up": BODY[True], "down": BODY[False]},
                   "wick": {"up": WICK[True], "down": WICK[False]}},
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
            "乾淨的對照是 wick3／range 對 pxw（只差影線）；對 mpl 的比較同時包含版面與縮圖差異。",
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
        "# Frozen ViT：matplotlib vs 像素精確畫法", "# Frozen ViT：K 線影線加粗、換色對照").replace(
        "## 配對比較（候選 − mpl，", "## 配對比較（候選 − 基準，")
    (output / "report.md").write_text(text, encoding="utf-8")
    print(f"[wick_style] report -> {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
