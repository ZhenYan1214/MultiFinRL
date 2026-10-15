"""Frozen ViT：第二張圖改用 60 日 K 線（K20＋K60）對照現行 K20＋Volume。

所有設定共用同一批樣本、標籤、凍結的 ViT-B/16 與分類頭設定，只改變輸入的圖：

- single_k20      ：20 日 K 線
- single_k60      ：60 日 K 線
- dual_k20_volume ：20 日 K 線＋20 日 Volume（正式 pipeline 現行雙圖）
- dual_k20_k60    ：20 日 K 線＋60 日 K 線

K20 與 Volume 直接讀正式 pipeline 的 PNG；K60 以相同 mplfinance 樣式另外畫到本實驗的輸出
目錄，不寫入 data/raw/charts。雙圖沿用已封存 Frozen 實驗的做法：同一個 ViT 各取 CLS 後串接。

樣本：沿用 AAPL 的 `strict_temporal_v1` 日期切分（AAPL／NVDA／MSFT 共同日期）。OHLCV 從
2021-01-04 起，Train 最前面不足 60 根 K 棒的日期會從「所有設定」一併剔除；標籤門檻仍用
完整 Train 的 5 日報酬計算，與正式 manifest 一致。Test 完全不抽特徵、不評估。

用法：
    python experiments/vision/run_k20_k60_comparison.py
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
from experiments.vision.run_pixel_render_comparison import (
    SPLITS, TICKERS, load_ohlcv, metric_payload, numeric_baseline, stats, train_head,
)

PROTOCOL = "vit_k20_k60_comparison_v1"
OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "vit_k20_k60"
SHORT, LONG, HORIZON = 20, 60, 5
CONFIGS = {
    "single_k20": ["k20"],
    "single_k60": ["k60"],
    "dual_k20_volume": ["k20", "volume"],
    "dual_k20_k60": ["k20", "k60"],
}
COMPARISONS = [("dual_k20_k60", "dual_k20_volume"), ("dual_k20_k60", "single_k20"),
               ("single_k60", "single_k20")]
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")
# 預先設定的採用門檻，沿用 decisions.md #88～#90 的標準。
GATE_DELTA, GATE_WINS = 0.02, 4


def build_rows(ticker: str, manifest: dict) -> dict:
    frame = load_ohlcv(ticker)
    position = {date.strftime("%Y-%m-%d"): i for i, date in enumerate(frame.index)}
    close = frame["Close"].to_numpy(float)

    def forward_return(i: int) -> float:
        return close[i + HORIZON] / close[i] - 1

    # 門檻用完整 Train（含之後因 60 日窗口不足而剔除的日期），與正式 manifest 一致。
    full_train = [forward_return(position[date]) for date in manifest["dates"]["train"]]
    lower, upper = np.quantile(full_train, [0.3333, 0.6667])
    rows, dropped = {}, {}
    for split in SPLITS:
        dates = [date for date in manifest["dates"][split] if position[date] >= LONG - 1]
        dropped[split] = len(manifest["dates"][split]) - len(dates)
        index = [position[date] for date in dates]
        returns = np.asarray([forward_return(i) for i in index])
        rows[split] = {
            "dates": dates,
            "windows": [frame.iloc[i - SHORT + 1:i + 1] for i in index],
            "long_windows": [frame.iloc[i - LONG + 1:i + 1] for i in index],
            "labels": np.where(returns < lower, 0, np.where(returns > upper, 2, 1)),
        }
    return {"rows": rows, "dropped": dropped,
            "thresholds": {"lower": float(lower), "upper": float(upper)}}


def k60_path(output: Path, ticker: str, date: str) -> Path:
    return output / "charts" / ticker / "k60" / f"{date}.png"


def render_k60_charts(data: dict, output: Path) -> None:
    """用正式 K 線圖相同的 mplfinance 樣式與正規化流程畫 60 日 K 線；已存在的跳過。"""
    import mplfinance as mpf
    from module_a_data.preprocess.chart_generator import _STYLE, _save_figure

    for ticker, payload in data.items():
        made = 0
        for split in SPLITS:
            rows = payload["rows"][split]
            for date, window in zip(rows["dates"], rows["long_windows"]):
                target = k60_path(output, ticker, date)
                if target.exists():
                    continue
                fig, _ = mpf.plot(
                    window, type="candle", style=_STYLE, volume=False, axisoff=True,
                    returnfig=True, figsize=(3, 3), scale_padding=0,
                )
                _save_figure(fig, target, (224, 224))
                made += 1
        print(f"[k20_k60:charts] {ticker} 新增 {made} 張 K60", flush=True)


def image_path(output: Path, ticker: str, kind: str, date: str) -> Path:
    if kind == "k20":
        return paths.RAW_CHARTS / ticker / f"{date}.png"
    if kind == "volume":
        return paths.RAW_CHARTS / ticker / "volume" / f"{date}.png"
    if kind == "k60":
        return k60_path(output, ticker, date)
    raise ValueError(f"不支援的圖別：{kind}")


def save_samples(data: dict, output: Path, n: int = 3) -> list[str]:
    """每檔挑 Train 期間等距的 n 天，輸出 K20｜K60｜Volume 並排對照圖。"""
    out_dir = output / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ticker, payload in data.items():
        dates = payload["rows"]["train"]["dates"]
        for i in np.linspace(0, len(dates) - 1, n, dtype=int):
            sheet = Image.new("RGB", (224 * 3 + 16, 224 + 8), (128, 128, 128))
            for position, kind in enumerate(["k20", "k60", "volume"]):
                with Image.open(image_path(output, ticker, kind, dates[i])) as tile:
                    sheet.paste(tile.convert("RGB"), (4 + position * 228, 4))
            target = out_dir / f"{ticker}_{dates[i]}.png"
            sheet.save(target)
            saved.append(str(target.relative_to(ROOT)))
    return saved


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
                dates = payload["rows"][split]["dates"]
                target = cache / f"{ticker}_{kind}_{split}.npy"
                if target.exists():
                    features[(ticker, kind, split)] = np.load(target)
                    continue
                if backbone is None:
                    processor = AutoImageProcessor.from_pretrained(model_id)
                    backbone = AutoModel.from_pretrained(model_id).to(device).eval()
                parts = []
                for start in range(0, len(dates), batch):
                    images = []
                    for date in dates[start:start + batch]:
                        with Image.open(image_path(output, ticker, kind, date)) as image:
                            images.append(image.convert("RGB"))
                    with torch.no_grad():
                        pixels = processor(images=images, return_tensors="pt")["pixel_values"]
                        parts.append(backbone(pixel_values=pixels.to(device))
                                     .last_hidden_state[:, 0].cpu().numpy())
                array = np.concatenate(parts).astype(np.float32)
                np.save(target, array)
                features[(ticker, kind, split)] = array
                print(f"[k20_k60:features] {ticker} {kind} {split} {array.shape}", flush=True)
    return features


def numeric_for_window(payload: dict, key: str, seed: int) -> dict:
    """不看圖的對照；key 選 20 日或 60 日窗口的逐日報酬＋相對成交量。"""
    rows = {split: {"windows": payload["rows"][split][key],
                    "labels": payload["rows"][split]["labels"]} for split in SPLITS}
    return numeric_baseline({"rows": rows}, seed)


def write_markdown(report: dict, output: Path) -> None:
    counts = report["split_counts"]
    lines = [
        "# Frozen ViT：K20＋K60 vs K20＋Volume", "",
        f"- Protocol：`{report['protocol']}`；Test 未開啟。",
        f"- 股票：{'／'.join(report['tickers'])}；每檔 Train／Validation = "
        f"{counts['train']}／{counts['validation']}"
        f"（Train 最前面 {report['dropped']['train']} 天因不足 60 根 K 棒而剔除）。",
        f"- Seeds：{report['seeds']}；分類頭固定 {report['epochs']} epochs，回報最後一個 epoch。",
        "", "## Validation（五 seed 平均，mean-ticker）", "",
        "| 設定 | Macro-F1 | Balanced acc | ROC-AUC | Log loss |", "|---|---:|---:|---:|---:|",
    ]
    for name, summary in report["summary"].items():
        lines.append(
            f"| {name} | {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f} "
            f"| {summary['balanced_accuracy']['mean']:.4f} | {summary['roc_auc']['mean']:.4f} "
            f"| {summary['log_loss']['mean']:.4f} |")
    for name, baseline in report["numeric_baselines"].items():
        values = baseline["mean_ticker"]
        lines.append(f"| {name} | {values['macro_f1']:.4f} | {values['balanced_accuracy']:.4f} "
                     f"| {values['roc_auc']:.4f} | {values['log_loss']:.4f} |")
    lines += ["", "## 配對比較（候選 − 基準，mean-ticker Macro-F1）", "",
              "| 比較 | 平均差 | 勝出 seeds | ROC-AUC 差 | Gate |", "|---|---:|---:|---:|---|"]
    for item in report["comparisons"]:
        lines.append(f"| {item['candidate']} vs {item['baseline']} | "
                     f"{item['delta_macro_f1']['mean']:+.4f} ± {item['delta_macro_f1']['std']:.4f} "
                     f"| {item['wins']}/{len(report['seeds'])} | {item['delta_roc_auc']:+.4f} "
                     f"| {'PASS' if item['gate_passed'] else 'FAIL'} |")
    lines += ["", "## 各股票 Macro-F1／ROC-AUC（五 seed 平均）", "",
              "| 設定 | " + " | ".join(report["tickers"]) + " |",
              "|---|" + "---:|" * len(report["tickers"])]
    for name, summary in report["summary"].items():
        lines.append(f"| {name} | " + " | ".join(
            f"{summary['by_ticker'][t]['macro_f1']:.4f}／{summary['by_ticker'][t]['roc_auc']:.4f}"
            for t in report["tickers"]) + " |")
    lines += ["", "## 限制", ""] + [f"- {text}" for text in report["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="k20_k60")
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
    manifest = load_temporal_manifest("AAPL")   # 三檔共同日期，沿用 AAPL 的嚴格切分
    data = {ticker: build_rows(ticker, manifest) for ticker in TICKERS}
    render_k60_charts(data, output)
    samples = save_samples(data, output)
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
                "mean_ticker": {key: float(np.mean([by_ticker[t][key] for t in TICKERS]))
                                for key in METRICS},
            }
        print(f"[k20_k60] {name} 完成", flush=True)

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

    numeric_baselines = {}
    for name, key in (("numeric_logistic_w20", "windows"), ("numeric_logistic_w60", "long_windows")):
        by_ticker = {t: numeric_for_window(data[t], key, args.seeds[0]) for t in TICKERS}
        numeric_baselines[name] = {
            "by_ticker": by_ticker,
            "mean_ticker": {key_: float(np.mean([by_ticker[t][key_] for t in TICKERS]))
                            for key_ in METRICS},
        }

    report = {
        "protocol": PROTOCOL, "tickers": TICKERS, "short_window": SHORT, "long_window": LONG,
        "model_id": args.model_id, "seeds": args.seeds, "epochs": args.epochs,
        "head_lr": args.head_lr, "configs": CONFIGS,
        "split_counts": {s: len(data[TICKERS[0]]["rows"][s]["dates"]) for s in SPLITS},
        "dropped": data[TICKERS[0]]["dropped"],
        "split_manifest": str(paths.temporal_split_path("AAPL").relative_to(ROOT)),
        "label_thresholds": {ticker: data[ticker]["thresholds"] for ticker in TICKERS},
        "gate": {"min_delta_macro_f1": GATE_DELTA, "min_wins": GATE_WINS},
        "test_was_opened": False,
        "results": results, "summary": summary, "comparisons": comparisons,
        "numeric_baselines": numeric_baselines, "samples": samples,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "limitations": [
            "AAPL／NVDA／MSFT 的這段 Validation 已在先前多個實驗中被看過，結果可能偏樂觀。",
            "每檔約 926 筆 Train、202 筆 Validation，單一 seed 的差異多屬雜訊。",
            "同一份 Validation 上同時比較多組輸入，個別差異未做多重比較校正。",
            "MSFT 沒有正式 manifest，這裡以 AAPL 的日期切分與 MSFT 自己的 Train 報酬計算標籤。",
            "只比較 Frozen CLS 串接＋線性頭，不代表正式 Fusion（token 級融合）會有相同結果。",
            "60 日 K 線縮到 224×224 後每根 K 棒不到 4 像素寬，細節比 20 日圖模糊。",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output)
    print(f"[k20_k60] report -> {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
