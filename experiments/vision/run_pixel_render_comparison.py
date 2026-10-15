"""Frozen ViT：matplotlib K 線圖 vs 像素精確 K 線圖的畫法對照。

只改變「怎麼畫」，其餘全部固定：同一份 OHLCV、同一個 20 日窗口、同一個凍結的
ViT-B/16、同一組日期切分與 Train-only 分位數三分類標籤、同一型態的分類頭。

- mpl：正式 pipeline 現有的 mplfinance PNG（白底、畫完再 LANCZOS 縮到 224×224）。
- px ：直接填像素陣列（黑底、漲填 G／跌填 R、每根 K 棒固定像素欄、影線 1 像素、不縮圖）。
- pxw：與 px 相同的像素配置，但改回白底與 mplfinance 的綠／紅，用來區分底色與銳利度的影響。
- single 只用 K 線 CLS；dual 串接 K 線 CLS 與 Volume CLS（與已封存的 Frozen 實驗相同）。

股票為 AAPL／NVDA／MSFT，沿用 AAPL 的 `strict_temporal_v1` 日期切分（三檔共同日期）；
每檔各自只用 Train 的 5 日報酬算 q33／q67。Test 完全不抽特徵、不評估。分類頭固定訓練
epochs 數並回報最後一個 epoch，不用 Validation 挑 epoch。

用法：
    python experiments/vision/run_pixel_render_comparison.py --samples-only   # 只輸出對照圖
    python experiments/vision/run_pixel_render_comparison.py                  # 完整對照
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

PROTOCOL = "vit_pixel_render_comparison_v1"
OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "vit_pixel_render"
TICKERS = ["AAPL", "NVDA", "MSFT"]
SPLITS = ("train", "validation")
LABEL_NAMES = ["BEARISH", "NEUTRAL", "BULLISH"]
CONFIGS = {
    "mpl_single": ["mpl_candlestick"],
    "px_single": ["px_candlestick"],
    "pxw_single": ["pxw_candlestick"],
    "mpl_dual": ["mpl_candlestick", "mpl_volume"],
    "px_dual": ["px_candlestick", "px_volume"],
    "pxw_dual": ["pxw_candlestick", "pxw_volume"],
}
COMPARISONS = [("px_single", "mpl_single"), ("pxw_single", "mpl_single"),
               ("px_dual", "mpl_dual"), ("pxw_dual", "mpl_dual")]
# 預先設定的採用門檻，沿用 decisions.md #88～#90 的標準。
GATE_DELTA, GATE_WINS = 0.02, 4


# --- 像素繪圖 ---

def to_white_background(img: np.ndarray) -> np.ndarray:
    """黑底 G／R 圖 -> 白底、mplfinance 同色（green=(0,128,0)、red=(255,0,0)）。"""
    out = np.full_like(img, 255)
    out[img[..., 1] > 0] = (0, 128, 0)
    out[img[..., 0] > 0] = (255, 0, 0)
    return out


def _layout(n: int, size: int) -> tuple[int, int, int]:
    col = size // n                      # 每根 K 棒佔的像素欄數
    return col, (size - col * n) // 2, max(col - 2, 1)


def render_candles_px(window: pd.DataFrame, size: int = 224, pad: int = 4) -> np.ndarray:
    """窗口 OHLC -> [size,size,3] uint8。黑底；漲填 G、跌填 R；影線固定 1 像素寬。"""
    col, x0, body = _layout(len(window), size)
    lo, hi = float(window["Low"].min()), float(window["High"].max())
    span = max(hi - lo, 1e-9)

    def y(price: float) -> int:          # 價格 -> 像素列，上方為高價
        return int(round((size - 1 - pad) - (price - lo) / span * (size - 1 - 2 * pad)))

    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i, (o, h, l, c) in enumerate(window[["Open", "High", "Low", "Close"]].to_numpy(float)):
        left = x0 + i * col
        channel = 1 if c >= o else 0
        img[y(h):y(l) + 1, left + body // 2, channel] = 255
        top, bottom = sorted((y(o), y(c)))
        img[top:bottom + 1, left:left + body, channel] = 255   # 十字線至少 1 像素高
    return img


def render_volume_px(window: pd.DataFrame, size: int = 224, pad: int = 4) -> np.ndarray:
    """窗口成交量 -> [size,size,3] uint8；像素欄與 K 線圖逐日對齊。"""
    col, x0, body = _layout(len(window), size)
    volume = window["Volume"].to_numpy(float)
    vmax = max(float(volume.max()), 1e-9)
    up = window["Close"].to_numpy(float) >= window["Open"].to_numpy(float)

    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i, value in enumerate(volume):
        height = max(int(round(value / vmax * (size - 2 * pad))), 1)
        left = x0 + i * col
        img[size - pad - height:size - pad, left:left + body, 1 if up[i] else 0] = 255
    return img


# --- 資料 ---

def load_ohlcv(ticker: str) -> pd.DataFrame:
    frame = pd.read_csv(paths.RAW / "ohlcv" / f"{ticker}.csv", parse_dates=["Date"])
    return frame.set_index("Date").sort_index()


def build_rows(ticker: str, manifest: dict, window_days: int, horizon: int) -> dict:
    """回傳各 split 的日期、窗口、5 日報酬與 Train-only 分位數標籤。"""
    frame = load_ohlcv(ticker)
    position = {date.strftime("%Y-%m-%d"): i for i, date in enumerate(frame.index)}
    close = frame["Close"].to_numpy(float)
    rows = {}
    for split in SPLITS:
        dates = manifest["dates"][split]
        index = [position[date] for date in dates]
        if min(index) < window_days - 1 or max(index) + horizon >= len(frame):
            raise ValueError(f"{ticker} {split} 的 OHLCV 不足以涵蓋窗口或 {horizon} 日報酬")
        rows[split] = {
            "dates": dates,
            "windows": [frame.iloc[i - window_days + 1:i + 1] for i in index],
            "returns": np.asarray([close[i + horizon] / close[i] - 1 for i in index]),
        }
    lower, upper = np.quantile(rows["train"]["returns"], [0.3333, 0.6667])
    for split in SPLITS:
        values = rows[split]["returns"]
        rows[split]["labels"] = np.where(values < lower, 0, np.where(values > upper, 2, 1))
    return {"rows": rows, "thresholds": {"lower": float(lower), "upper": float(upper)}}


def load_image(ticker: str, kind: str, date: str, window: pd.DataFrame) -> Image.Image:
    if kind == "mpl_candlestick":
        return Image.open(paths.RAW_CHARTS / ticker / f"{date}.png").convert("RGB")
    if kind == "mpl_volume":
        return Image.open(paths.RAW_CHARTS / ticker / "volume" / f"{date}.png").convert("RGB")
    if kind == "px_candlestick":
        return Image.fromarray(render_candles_px(window))
    if kind == "px_volume":
        return Image.fromarray(render_volume_px(window))
    if kind == "pxw_candlestick":
        return Image.fromarray(to_white_background(render_candles_px(window)))
    if kind == "pxw_volume":
        return Image.fromarray(to_white_background(render_volume_px(window)))
    raise ValueError(f"不支援的圖別：{kind}")


def save_samples(data: dict, output: Path, n: int = 3) -> list[str]:
    """每檔挑 Train 期間等距的 n 天，輸出 mpl／px／pxw 並排對照圖（上：K 線，下：Volume）。"""
    out_dir = output / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ticker, payload in data.items():
        train = payload["rows"]["train"]
        for i in np.linspace(0, len(train["dates"]) - 1, n, dtype=int):
            date, window = train["dates"][i], train["windows"][i]
            sheet = Image.new("RGB", (224 * 3 + 16, 224 * 2 + 12), (128, 128, 128))
            for position, kind in enumerate(
                ["mpl_candlestick", "px_candlestick", "pxw_candlestick",
                 "mpl_volume", "px_volume", "pxw_volume"]
            ):
                tile = load_image(ticker, kind, date, window)
                sheet.paste(tile, (4 + (position % 3) * 228, 4 + (position // 3) * 228))
            target = out_dir / f"{ticker}_{date}.png"
            sheet.save(target)
            saved.append(str(target.relative_to(ROOT)))
    return saved


# --- Frozen 特徵 ---

def extract_features(data: dict, output: Path, model_id: str, device: str, batch: int) -> dict:
    """各 ticker／圖別／split 的 Frozen CLS [N,768]；已存在的快取直接讀取。"""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    cache = output / "features"
    cache.mkdir(parents=True, exist_ok=True)
    kinds = sorted({kind for kinds in CONFIGS.values() for kind in kinds})
    processor = backbone = None
    features = {}
    for ticker, payload in data.items():
        for kind in kinds:
            for split in SPLITS:
                rows = payload["rows"][split]
                target = cache / f"{ticker}_{kind}_{split}.npy"
                if target.exists():
                    features[(ticker, kind, split)] = np.load(target)
                    continue
                if backbone is None:
                    processor = AutoImageProcessor.from_pretrained(model_id)
                    backbone = AutoModel.from_pretrained(model_id).to(device).eval()
                parts = []
                for start in range(0, len(rows["dates"]), batch):
                    images = [
                        load_image(ticker, kind, date, window)
                        for date, window in zip(rows["dates"][start:start + batch],
                                                rows["windows"][start:start + batch])
                    ]
                    with torch.no_grad():
                        pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                        parts.append(backbone(pixel_values=pixels.to(device))
                                     .last_hidden_state[:, 0].cpu().numpy())
                array = np.concatenate(parts).astype(np.float32)
                np.save(target, array)
                features[(ticker, kind, split)] = array
                print(f"[pixel_render:features] {ticker} {kind} {split} {array.shape}", flush=True)
    return features


# --- 分類頭與指標 ---

def metric_payload(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss,
                                 roc_auc_score)
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro",
                                   zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities, multi_class="ovr",
                                       average="macro", labels=[0, 1, 2])),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "predicted_counts": np.bincount(predictions, minlength=3).tolist(),
    }


def train_head(x_train, y_train, x_val, seed: int, epochs: int, lr: float) -> np.ndarray:
    """LayerNorm＋Dropout＋Linear，class-weighted；回傳最後一個 epoch 的 Validation 機率。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    head = nn.Sequential(nn.LayerNorm(x_train.shape[1]), nn.Dropout(0.1),
                         nn.Linear(x_train.shape[1], 3))
    counts = np.bincount(y_train, minlength=3)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(len(y_train) / (3 * counts), dtype=torch.float32))
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    features = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train).long()
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


def numeric_baseline(payload: dict, seed: int) -> dict:
    """不看圖的對照：窗口內逐日報酬＋相對成交量 -> Logistic Regression。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def numeric(window: pd.DataFrame) -> np.ndarray:
        close = window["Close"].to_numpy(float)
        volume = window["Volume"].to_numpy(float)
        return np.concatenate([np.diff(np.log(close)), volume / volume.mean() - 1])

    rows = payload["rows"]
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=seed))
    model.fit(np.stack([numeric(w) for w in rows["train"]["windows"]]), rows["train"]["labels"])
    probabilities = model.predict_proba(
        np.stack([numeric(w) for w in rows["validation"]["windows"]]))
    return metric_payload(rows["validation"]["labels"], probabilities)


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def write_markdown(report: dict, output: Path) -> None:
    lines = [
        "# Frozen ViT：matplotlib vs 像素精確畫法", "",
        f"- Protocol：`{report['protocol']}`；Test 未開啟。",
        f"- 股票：{'／'.join(report['tickers'])}；每檔 Train／Validation = "
        f"{report['split_counts']['train']}／{report['split_counts']['validation']}。",
        f"- Seeds：{report['seeds']}；分類頭固定 {report['epochs']} epochs，回報最後一個 epoch。",
        "", "## Validation（五 seed 平均，mean-ticker）", "",
        "| 設定 | Macro-F1 | Balanced acc | ROC-AUC | Log loss |", "|---|---:|---:|---:|---:|",
    ]
    for name, summary in report["summary"].items():
        lines.append(
            f"| {name} | {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f} "
            f"| {summary['balanced_accuracy']['mean']:.4f} | {summary['roc_auc']['mean']:.4f} "
            f"| {summary['log_loss']['mean']:.4f} |")
    baseline = report["numeric_baseline"]["mean_ticker"]
    lines.append(f"| numeric_logistic | {baseline['macro_f1']:.4f} | "
                 f"{baseline['balanced_accuracy']:.4f} | {baseline['roc_auc']:.4f} | "
                 f"{baseline['log_loss']:.4f} |")
    lines += ["", "## 配對比較（候選 − mpl，mean-ticker Macro-F1）", "",
              "| 比較 | 平均差 | 勝出 seeds | ROC-AUC 差 | Gate |", "|---|---:|---:|---:|---|"]
    for item in report["comparisons"]:
        lines.append(f"| {item['candidate']} vs {item['baseline']} | "
                     f"{item['delta_macro_f1']['mean']:+.4f} ± {item['delta_macro_f1']['std']:.4f} "
                     f"| {item['wins']}/{len(report['seeds'])} | {item['delta_roc_auc']:+.4f} "
                     f"| {'PASS' if item['gate_passed'] else 'FAIL'} |")
    lines += ["", "## 各股票 Macro-F1（五 seed 平均）", "",
              "| 設定 | " + " | ".join(report["tickers"]) + " |",
              "|---|" + "---:|" * len(report["tickers"])]
    for name, summary in report["summary"].items():
        lines.append(f"| {name} | " + " | ".join(
            f"{summary['by_ticker'][ticker]:.4f}" for ticker in report["tickers"]) + " |")
    lines += ["", "## 限制", ""] + [f"- {text}" for text in report["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="px_vs_mpl_w20")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--device", default=None)
    parser.add_argument("--samples-only", action="store_true", help="只輸出畫法對照圖，不跑模型")
    args = parser.parse_args()

    output = OUTPUT_ROOT / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_temporal_manifest("AAPL")   # 三檔共同日期，沿用 AAPL 的嚴格切分
    data = {ticker: build_rows(ticker, manifest, args.window, 5) for ticker in TICKERS}
    samples = save_samples(data, output)
    print(f"[pixel_render] 對照圖 -> {output / 'samples'}（{len(samples)} 張）", flush=True)
    if args.samples_only:
        return

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    features = extract_features(data, output, args.model_id, device, args.image_batch)

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
                "mean_ticker": {
                    key: float(np.mean([by_ticker[t][key] for t in TICKERS]))
                    for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")
                },
            }
        print(f"[pixel_render] {name} 完成", flush=True)

    summary = {}
    for name, by_seed in results.items():
        summary[name] = {
            key: stats([item["mean_ticker"][key] for item in by_seed.values()])
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")
        }
        summary[name]["by_ticker"] = {
            ticker: float(np.mean([item["by_ticker"][ticker]["macro_f1"]
                                   for item in by_seed.values()]))
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

    numeric = {ticker: numeric_baseline(data[ticker], args.seeds[0]) for ticker in TICKERS}
    report = {
        "protocol": PROTOCOL, "tickers": TICKERS, "window_days": args.window,
        "model_id": args.model_id, "seeds": args.seeds, "epochs": args.epochs,
        "head_lr": args.head_lr, "configs": CONFIGS,
        "split_counts": {split: len(manifest["dates"][split]) for split in SPLITS},
        "split_manifest": str(paths.temporal_split_path("AAPL").relative_to(ROOT)),
        "label_thresholds": {ticker: data[ticker]["thresholds"] for ticker in TICKERS},
        "gate": {"min_delta_macro_f1": GATE_DELTA, "min_wins": GATE_WINS},
        "test_was_opened": False,
        "results": results, "summary": summary, "comparisons": comparisons,
        "numeric_baseline": {
            "by_ticker": numeric,
            "mean_ticker": {key: float(np.mean([numeric[t][key] for t in TICKERS]))
                            for key in ("macro_f1", "balanced_accuracy", "roc_auc", "log_loss")},
        },
        "samples": samples,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "limitations": [
            "px 同時改變底色（黑）與銳利度（不縮圖）；pxw 只改銳利度與版面，用來拆開兩者。",
            "同一份 Validation 上比較了多個畫法，個別差異未做多重比較校正。",
            "AAPL／NVDA／MSFT 的這段 Validation 已在先前多個實驗中被看過，結果可能偏樂觀。",
            "每檔只有 966 筆 Train、202 筆 Validation，單一 seed 的差異多屬雜訊。",
            "MSFT 沒有正式 manifest，這裡以 AAPL 的日期切分與 MSFT 自己的 Train 報酬計算標籤。",
            "只比較 Frozen CLS＋線性頭，不代表正式 Fusion（token 級融合）會有相同結果。",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output)
    print(f"[pixel_render] report -> {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
