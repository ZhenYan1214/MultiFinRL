"""文獻中的金融圖像 CNN 方法，套用本專案的三分類標籤與 walk-forward 協定。

目的：檢查已發表的 K 線／價格圖像 CNN 做法，在 AAPL／NVDA／MSFT、t+5 Train-only q33／q67
三分類下，能否勝過 Frozen ViT 與 Numeric Logistic。與 run_cnn_baseline.py 的差別是：圖不是
mplfinance 圖，而是照各論文的方式由 OHLCV 直接產生。

方法（皆依原文；原文未寫或與本協定衝突處在下方註明）：
- jkx：Jiang, Kelly & Xiu (2023, Journal of Finance)「(Re-)Imag(in)ing Price Trends」的 I20 模型。
  64×60 黑底白線灰階圖：每天 3 像素寬（開盤橫線、高低直線、收盤橫線），20 日均線，
  底部 1/5 為成交量柱；價格以視窗內 OHLC 最高／最低撐滿高度。CNN 為 3 個區塊
  （5×3 卷積、BatchNorm、LeakyReLU、2×1 MaxPool；64→128→256；第一層 stride (3,1)、
  dilation (2,1)），FC 前 50% dropout，Xavier 初始化，Adam lr 1e-5、batch 128，
  驗證 loss 連續 2 個 epoch 未改善即停止，5 次獨立訓練的機率平均。兩類輸出改為三類。
- cnn_ta：Sezer & Ozbayoglu (2018, Applied Soft Computing)「CNN-TA」。15 種技術指標 ×
  15 種參數（6～20 日）組成 15×15 單通道圖。CNN：Conv3×3(32)→Conv3×3(64)→MaxPool→
  Dropout 0.25→FC128→Dropout 0.5→輸出。原文以價格高低點標 Buy／Sell／Hold，這裡改用本專案標籤；
  原文的 Parabolic SAR 沒有期間參數，以 Stochastic %K 取代；均線類指標除以收盤價使其無單位。
- gaf：Barra et al. (2020, IEEE/CAA JAS) 的多解析度 Gramian Angular Field。收盤價在
  20／40／60 日窗口下各取 20 點（逐日、2 日、3 日平均），轉成 GASF，疊成 3×20×20 圖；
  CNN 同 cnn_ta 的結構（原文為多個 CNN 的 ensemble、架構細節不同）。

共同協定：
- 資料、folds、標籤、統計單位與 run_horizon_period_grid.py 的 grid_2016 相同
  （2016-01-04～2025-10-01、7 個逐年 folds、5 日間隔），可與該 run 的 Frozen ViT 逐單位配對。
- 三檔股票的 Train 合併訓練。Early stopping 用 Train 最後 20% 期間（與前段留 5 日間隔），
  不碰 Validation；JKX 原文是在訓練期內隨機抽 30%，但相鄰窗口幾乎重疊，隨機抽會讓
  early stopping 過度樂觀，因此改用時間切分。
- 正規化統計量（像素均值／標準差、指標標準化）只用該 fold 的 Train 計算。
- 正式 Test 期間不建立任何樣本。

用法：
    python experiments/vision/run_cnn_literature.py --smoke
    python experiments/vision/run_cnn_literature.py --methods jkx cnn_ta gaf
中斷後同指令續跑：已完成的 method × fold 會跳過。
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import paths
from shared.temporal_split import load_temporal_manifest
from experiments.vision.run_horizon_period_grid import (
    GAP, OUTPUT_ROOT as GRID_ROOT, WINDOW, build_folds, classify_metrics, labels_for,
    load_long_ohlcv, majority_metrics, numeric_metrics, paired, sample_dates, unit_mean,
)

PROTOCOL = "cnn_literature_walk_forward_v1"
OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "cnn_literature"
METHODS = ("jkx", "cnn_ta", "gaf")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")
VIT_KEYS = {"vit_single_k20": "lr1e-4_e100|{h}|single_k20",
            "vit_dual_k20_volume": "lr1e-4_e100|{h}|dual_k20_volume"}
TRAINING = {   # method: (lr, batch, max_epochs, patience)
    "jkx": (1e-5, 128, 100, 2),
    "cnn_ta": (1e-3, 128, 100, 5),
    "gaf": (1e-3, 128, 100, 5),
}


# --- 輸入：照各論文由 OHLCV 產生 ---

def jkx_image(frame: pd.DataFrame, end: int, ma: np.ndarray) -> np.ndarray:
    """JKX I20：64×60，上 51 列畫 OHLC＋均線、空 1 列、下 12 列畫成交量；黑底 0、白 255。"""
    height, price_rows, volume_rows = 64, 51, 12
    image = np.zeros((height, 3 * WINDOW), dtype=np.uint8)
    rows = slice(end - WINDOW + 1, end + 1)
    o, h, l, c = (frame[k].to_numpy(float)[rows] for k in ("Open", "High", "Low", "Close"))
    v, m = frame["Volume"].to_numpy(float)[rows], ma[rows]
    low, high = l.min(), h.max()          # 原文：以 OHLC 路徑的最高／最低撐滿；均線超出則截斷
    span = high - low if high > low else 1.0

    def to_row(price: np.ndarray) -> np.ndarray:
        scaled = np.clip((price - low) / span, 0, 1)
        return (price_rows - 1 - np.round(scaled * (price_rows - 1))).astype(int)

    r_o, r_h, r_l, r_c, r_m = to_row(o), to_row(h), to_row(l), to_row(c), to_row(m)
    for day in range(WINDOW):
        x = 3 * day
        image[r_o[day], x] = 255
        image[r_h[day]:r_l[day] + 1, x + 1] = 255
        image[r_c[day], x + 2] = 255
        if day > 0:                        # 均線：每天中間欄一點，相鄰兩點以直線相連
            steps = 4
            ys = np.linspace(r_m[day - 1], r_m[day], steps)
            xs = np.linspace(x - 2, x + 1, steps)
            image[np.round(ys).astype(int), np.round(xs).astype(int)] = 255
        else:
            image[r_m[day], x + 1] = 255
    peak = v.max() if v.max() > 0 else 1.0
    bars = np.round(v / peak * volume_rows).astype(int)
    for day in range(WINDOW):
        if bars[day] > 0:
            image[height - bars[day]:, 3 * day + 1] = 255
    return image[None]


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def _wma(x: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return x.rolling(n).apply(lambda w: np.dot(w, weights) / weights.sum(), raw=True)


def cnn_ta_features(frame: pd.DataFrame) -> np.ndarray:
    """每天一張 15（指標）×15（期間 6～20）的矩陣；回傳 [日數, 15, 15]，前期不足處為 NaN。"""
    c, h, l, v = frame["Close"], frame["High"], frame["Low"], frame["Volume"]
    out = np.full((len(frame), 15, 15), np.nan)
    for j, n in enumerate(range(6, 21)):
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        hh, ll = h.rolling(n).max(), l.rolling(n).min()
        williams = -100 * (hh - c) / (hh - ll)
        stoch = 100 * (c - ll) / (hh - ll)
        sma, ema, wma = c.rolling(n).mean(), _ema(c, n), _wma(c, n)
        hma = _wma(2 * _wma(c, max(n // 2, 1)) - _wma(c, n), max(int(np.sqrt(n)), 1))
        e1 = _ema(c, n); e2 = _ema(e1, n); e3 = _ema(e2, n)
        tema = 3 * e1 - 3 * e2 + e3
        tp = (h + l + c) / 3
        mad = tp.rolling(n).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True)
        cci = (tp - tp.rolling(n).mean()) / (0.015 * mad)
        up_sum = delta.clip(lower=0).rolling(n).sum()
        down_sum = (-delta.clip(upper=0)).rolling(n).sum()
        cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum)
        macd = (_ema(c, n) - _ema(c, 2 * n)) / c
        ppo = 100 * (_ema(c, n) - _ema(c, 2 * n)) / _ema(c, 2 * n)
        roc = 100 * (c / c.shift(n) - 1)
        mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        cmf = (mfm * v).rolling(n).sum() / v.rolling(n).sum()
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        plus_dm = (h.diff()).where((h.diff() > -l.diff()) & (h.diff() > 0), 0.0)
        minus_dm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0.0)
        atr = tr.ewm(alpha=1 / n, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
        dmi = (plus_di - minus_di)
        rows = [rsi, williams, wma / c, ema / c, sma / c, hma / c, tema / c, cci, cmo, macd,
                ppo, roc, cmf, dmi, stoch]
        for i, series in enumerate(rows):
            out[:, i, j] = series.to_numpy(float)
    return out


def gasf(series: np.ndarray) -> np.ndarray:
    lo, hi = series.min(), series.max()
    scaled = np.clip(2 * (series - lo) / (hi - lo) - 1, -1, 1) if hi > lo else np.zeros_like(series)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])


def gaf_image(close: np.ndarray, end: int) -> np.ndarray:
    """3×20×20：20 日逐日、40 日每 2 日平均、60 日每 3 日平均的收盤 GASF。"""
    channels = []
    for step in (1, 2, 3):
        window = close[end - 20 * step + 1:end + 1].reshape(20, step).mean(axis=1)
        channels.append(gasf(window))
    return np.stack(channels).astype(np.float32)


# --- 模型 ---

def build_model(method: str, n_classes: int = 3):
    import torch.nn as nn

    if method == "jkx":
        def block(c_in, c_out, **kw):
            return [nn.Conv2d(c_in, c_out, (5, 3), **kw), nn.BatchNorm2d(c_out),
                    nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
        model = nn.Sequential(
            *block(1, 64, stride=(3, 1), dilation=(2, 1), padding=(12, 1)),   # 64×60 → 13×60
            *block(64, 128, padding=(2, 1)),                                  # → 6×60
            *block(128, 256, padding=(2, 1)),                                 # → 3×60
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 3 * 60, n_classes))
        for layer in model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        return model
    channels = 1 if method == "cnn_ta" else 3
    size = 15 if method == "cnn_ta" else 20
    pooled = (size - 4) // 2                       # 兩個 3×3 valid 卷積後 2×2 pooling
    return nn.Sequential(
        nn.Conv2d(channels, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.25), nn.Flatten(),
        nn.Linear(64 * pooled * pooled, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, n_classes))


def train_one(method: str, x_fit, y_fit, x_stop, y_stop, x_val, seed: int, device: str) -> tuple:
    """固定 lr／batch；以 x_stop 的 loss 做 early stopping，回傳最佳 epoch 權重下的 Validation 機率。"""
    import torch
    import torch.nn as nn

    lr, batch, max_epochs, patience = TRAINING[method]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(method).to(device)
    counts = np.bincount(y_fit, minlength=3)
    weight = torch.tensor(len(y_fit) / (3 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    to = lambda a: torch.from_numpy(a).to(device)

    def predict(array):
        model.eval()
        with torch.no_grad():
            return torch.cat([torch.softmax(model(to(array[s:s + 512])), 1).cpu()
                              for s in range(0, len(array), 512)]).numpy()

    best, best_state, best_epoch, waited, history = np.inf, None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(y_fit), generator=generator).numpy()
        total = 0.0
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(to(x_fit[index])), to(y_fit[index]).long())
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        stop_prob = predict(x_stop)
        stop_loss = float(criterion(torch.log(torch.from_numpy(stop_prob).clamp_min(1e-7)).to(device),
                                    to(y_stop).long()))
        history.append({"epoch": epoch, "train_loss": total / len(order), "stop_loss": stop_loss})
        if stop_loss < best - 1e-6:
            best, best_epoch, waited = stop_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    probabilities = predict(x_val)
    train_accuracy = float((predict(x_fit).argmax(1) == y_fit).mean())
    del model, optimizer, best_state
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return probabilities, {"best_epoch": best_epoch, "epochs_run": len(history),
                           "best_stop_loss": best, "train_accuracy_at_best": train_accuracy,
                           "history": history}


def write_markdown(report: dict, output: Path) -> None:
    lines = [
        "# 文獻 CNN 方法 vs Frozen ViT（三分類 walk-forward）", "",
        f"- Protocol：`{report['protocol']}`；正式 Test 期間未建立樣本。",
        f"- 股票：{'／'.join(report['tickers'])}；folds：{'、'.join(report['fold_names'])}；"
        f"統計單位 = fold × 股票，共 {report['n_units']} 個。",
        f"- 標籤：t+{report['horizon']} 報酬的 Train-only q33／q67 三分類。",
        f"- 每個方法在每個 fold 訓練 {len(report['seeds'])} 次（seeds {report['seeds']}），"
        "Validation 機率取平均（JKX 原文的 ensemble 做法）；early stopping 用 Train 最後 20% 期間。",
        f"- ViT 為 `{report['vit_source']}` 的 Frozen ViT CLS＋線性頭（lr1e-4×100）。",
        "", "## Validation（fold × 股票平均 ± 標準誤）", "",
        "| 方法 | ROC-AUC | Balanced acc | Accuracy | Macro-F1 | Log loss |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, values in report["summary"].items():
        lines.append(f"| {name} | {values['roc_auc']['mean']:.4f} ± {values['roc_auc']['se']:.4f} | "
                     f"{values['balanced_accuracy']['mean']:.4f} | {values['accuracy']['mean']:.4f} | "
                     f"{values['macro_f1']['mean']:.4f} | {values['log_loss']['mean']:.4f} |")
    lines += ["", "## 配對比較（單位差值平均 ± 標準誤，勝出單位數）", "",
              "| 候選 vs 基準 | ROC-AUC 差 | Macro-F1 差 |", "|---|---|---|"]
    for item in report["comparisons"]:
        auc, f1 = item["roc_auc"], item["macro_f1"]
        lines.append(f"| {item['candidate']} vs {item['baseline']} | "
                     f"{auc['mean']:+.4f} ± {auc['se']:.4f}（{auc['wins']}/{auc['units']}） | "
                     f"{f1['mean']:+.4f} ± {f1['se']:.4f}（{f1['wins']}/{f1['units']}） |")
    lines += ["", "## 各 fold 的 ROC-AUC（各股票平均）", "",
              "| 方法 | " + " | ".join(report["fold_names"]) + " |",
              "|---|" + "---:|" * len(report["fold_names"])]
    for name, by_fold in report["by_fold_auc"].items():
        lines.append(f"| {name} | " + " | ".join(f"{by_fold[f]:.3f}" for f in report["fold_names"]) + " |")
    lines += ["", "## 訓練診斷（各 fold × seed 平均）", "",
              "| 方法 | 停止時的 epoch | 實際跑的 epochs | 最佳 epoch 的 Train accuracy | Validation 最大類別占比 |",
              "|---|---:|---:|---:|---:|"]
    for name, values in report["diagnostics"].items():
        lines.append(f"| {name} | {values['best_epoch']:.1f} | {values['epochs_run']:.1f} | "
                     f"{values['train_accuracy_at_best']:.4f} | {values['val_max_class_share']:.2f} |")
    lines += ["", "## 限制", ""] + [f"- {text}" for text in report["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="lit_2016_t5")
    parser.add_argument("--grid-run", default="grid_2016", help="讀取 OHLCV 與 ViT 結果的 grid run")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "NVDA", "MSFT"])
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--download-start", default="2015-09-01")
    parser.add_argument("--fold-years", type=int, nargs="+",
                        default=[2019, 2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42, 43, 44])
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true", help="工程短測：2 folds、1 seed、最多 2 epochs")
    args = parser.parse_args()
    if args.smoke:
        args.run_name, args.fold_years, args.seeds = "smoke", [2024, 2025], [40]
        for key, (lr, batch, _, patience) in TRAINING.items():
            TRAINING[key] = (lr, batch, 2, patience)

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    output = OUTPUT_ROOT / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    grid_output = GRID_ROOT / args.grid_run / "run"
    started = time.perf_counter()

    manifest = load_temporal_manifest("AAPL")
    last_sample = manifest["dates"]["validation"][-1]
    download_end = (pd.Timestamp(manifest["dates"]["gap_validation_test"][-1])
                    + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames = {t: load_long_ohlcv(t, grid_output, args.download_start, download_end)
              for t in args.tickers}
    dates = sample_dates(frames[args.tickers[0]], args.start, last_sample)
    folds = build_folds(dates, args.fold_years)

    position = {t: {d: i for i, d in enumerate(frames[t].index)} for t in args.tickers}
    index = {t: np.asarray([position[t][d] for d in dates]) for t in args.tickers}
    returns, windows = {}, {}
    for t in args.tickers:
        close = frames[t]["Close"].to_numpy(float)
        returns[t] = close[index[t] + args.horizon] / close[index[t]] - 1
        windows[t] = [frames[t].iloc[position[t][d] - WINDOW + 1:position[t][d] + 1] for d in dates]

    inputs = {}
    for method in args.methods:
        for t in args.tickers:
            frame = frames[t]
            if method == "jkx":
                ma = frame["Close"].rolling(WINDOW).mean().to_numpy(float)
                inputs[(method, t)] = np.stack([jkx_image(frame, i, ma) for i in index[t]])
            elif method == "cnn_ta":
                inputs[(method, t)] = cnn_ta_features(frame)[index[t]][:, None].astype(np.float32)
            else:
                close = frame["Close"].to_numpy(float)
                inputs[(method, t)] = np.stack([gaf_image(close, i) for i in index[t]])
            if np.isnan(inputs[(method, t)].astype(float)).any():
                raise ValueError(f"{method} {t} 的輸入有 NaN（前期資料不足）")
        print(f"[cnn_literature:inputs] {method} {inputs[(method, args.tickers[0])].shape}", flush=True)
    if "jkx" in args.methods:
        np.save(output / "jkx_sample_AAPL.npy", inputs[("jkx", args.tickers[0])][-5:])

    results_path = output / "results.json"
    saved = json.loads(results_path.read_text()) if results_path.exists() else {}
    for method in args.methods:
        for fold in folds:
            key = f"{method}|{fold['name']}"
            if key in saved:
                continue
            block_started = time.perf_counter()
            labels = {t: labels_for(returns[t], fold, "tercile") for t in args.tickers}
            cut = int(len(fold["train"]) * 0.8)
            fit_idx, stop_idx = fold["train"][:cut - GAP], fold["train"][cut:]
            gather = lambda idx: np.concatenate([inputs[(method, t)][idx] for t in args.tickers])
            x_fit, x_stop = gather(fit_idx), gather(stop_idx)
            x_val = gather(fold["validation"])
            y_all = {t: labels[t][0] for t in args.tickers}         # Train 段標籤，依 fold["train"] 排列
            y_fit = np.concatenate([y_all[t][:cut - GAP] for t in args.tickers])
            y_stop = np.concatenate([y_all[t][cut:] for t in args.tickers])
            # 正規化：只用 fit 段的統計量
            if method == "jkx":
                mean, std = float(x_fit.mean()), float(x_fit.std()) or 1.0
                norm = lambda a: ((a.astype(np.float32) - mean) / std).astype(np.float32)
            elif method == "cnn_ta":
                mean = x_fit.mean(axis=0, keepdims=True)
                std = x_fit.std(axis=0, keepdims=True) + 1e-8
                norm = lambda a: np.clip((a - mean) / std, -5, 5).astype(np.float32)
            else:
                norm = lambda a: a.astype(np.float32)
            x_fit, x_stop, x_val = norm(x_fit), norm(x_stop), norm(x_val)
            runs = [train_one(method, x_fit, y_fit, x_stop, y_stop, x_val, seed, device)
                    for seed in args.seeds]
            probabilities = np.mean([p for p, _ in runs], axis=0)
            per_ticker, offset = {}, 0
            for t in args.tickers:
                y_val = labels[t][1]
                metrics = classify_metrics(y_val, probabilities[offset:offset + len(y_val)])
                metrics.setdefault("predicted_counts", np.bincount(
                    probabilities[offset:offset + len(y_val)].argmax(1), minlength=3).tolist())
                per_ticker[t] = metrics
                offset += len(y_val)
            saved[key] = {"metrics": per_ticker, "runs": [info for _, info in runs]}
            results_path.write_text(json.dumps(saved), encoding="utf-8")
            epochs = [info["best_epoch"] for _, info in runs]
            print(f"[cnn_literature] {key} 完成（{time.perf_counter() - block_started:.0f} 秒；"
                  f"best epochs {epochs}；Train acc "
                  f"{np.mean([i['train_accuracy_at_best'] for _, i in runs]):.3f}）", flush=True)

    units = {m: {f"{fold['name']}|{t}": {k: saved[f"{m}|{fold['name']}"]["metrics"][t][k]
                                          for k in METRICS}
                 for fold in folds for t in args.tickers} for m in args.methods}
    vit_results = json.loads((grid_output / "results.json").read_text())
    if not args.smoke:
        for name, template in VIT_KEYS.items():
            units[name] = {u: v for u, v in vit_results[template.format(h=args.horizon)].items()
                           if u in units[args.methods[0]]}
    numeric_units, majority_units = {}, {}
    for fold in folds:
        for t in args.tickers:
            y_train, y_val, _ = labels_for(returns[t], fold, "tercile")
            unit = f"{fold['name']}|{t}"
            numeric_units[unit] = numeric_metrics(windows[t], fold, y_train, y_val, args.seeds[0])
            majority_units[unit] = majority_metrics(y_train, y_val, 3)
    units["numeric_logistic_w20"] = numeric_units
    units["majority"] = majority_units

    summary = {name: unit_mean(per_unit) for name, per_unit in units.items()}
    baselines = [b for b in ("vit_single_k20", "numeric_logistic_w20") if b in units]
    comparisons = [{"candidate": m, "baseline": b, "roc_auc": paired(units[m], units[b], "roc_auc"),
                    "macro_f1": paired(units[m], units[b], "macro_f1")}
                   for m in args.methods for b in baselines]
    by_fold_auc = {name: {fold["name"]: float(np.mean([per_unit[f"{fold['name']}|{t}"]["roc_auc"]
                                                       for t in args.tickers])) for fold in folds}
                   for name, per_unit in units.items() if name != "majority"}
    diagnostics = {}
    for m in args.methods:
        entries = [saved[f"{m}|{fold['name']}"] for fold in folds]
        runs = [r for e in entries for r in e["runs"]]
        diagnostics[m] = {k: float(np.mean([r[k] for r in runs]))
                          for k in ("best_epoch", "epochs_run", "train_accuracy_at_best")}
        diagnostics[m]["val_max_class_share"] = float(np.mean([
            max(x["predicted_counts"]) / sum(x["predicted_counts"])
            for e in entries for x in e["metrics"].values()]))
    parameters = {m: int(sum(p.numel() for p in build_model(m).parameters())) for m in args.methods}

    report = {
        "protocol": PROTOCOL, "smoke": args.smoke, "tickers": args.tickers, "horizon": args.horizon,
        "period": [f"{dates[0]:%Y-%m-%d}", f"{dates[-1]:%Y-%m-%d}"],
        "fold_names": [fold["name"] for fold in folds], "n_units": len(folds) * len(args.tickers),
        "methods": args.methods, "parameters": parameters, "training": TRAINING,
        "seeds": args.seeds, "vit_source": f"{args.grid_run}/run/results.json",
        "test_was_opened": False, "summary": summary, "comparisons": comparisons,
        "by_fold_auc": by_fold_auc, "diagnostics": diagnostics,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "limitations": [
            "fold × 股票單位彼此相關（同一年各股票常同漲同跌），標準誤偏樂觀。",
            "原文多以數千檔股票訓練（JKX 為 1993～2000 年全美股），這裡每個 fold 只有三檔、約 2,000～5,000 筆，"
            "資料量差兩到三個數量級；JKX 也指出只用最大的 500 檔時效果明顯減弱。",
            "原文的標籤、切分或 ensemble 細節與本協定不同（見 runner 說明），屬於「方法移植」而非逐字重現。",
            "同一批 Validation 上比較多個方法，最好的那個會因多重比較而偏樂觀。",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output)
    print(f"[cnn_literature] report -> {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
