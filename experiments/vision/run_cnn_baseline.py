"""從頭訓練的小型 CNN 基線：直接在 K 線圖上學特徵，與凍結 ImageNet ViT 對照。

目的：區分「K 線圖本身沒有方向訊號」與「凍結 ViT 看不懂 K 線圖（領域落差）」。

- 資料與評估完全沿用 run_horizon_period_grid.py 的 grid_2016 run：同一批 mplfinance 圖
  （只讀取）、同一份 OHLCV、同樣 7 個逐年 walk-forward folds、t+5 報酬的 Train-only
  q33／q67 三分類、同樣的統計單位（fold × 股票），因此可與該 run 的 Frozen ViT 結果逐單位配對。
- 每個 fold、每個 seed 從頭訓練一個 CNN（三檔股票的 Train 樣本合併訓練），固定 epochs、
  回報最後一個 epoch，不以 Validation 挑 epoch；Validation 只用來評估。
- 架構刻意很小（約 6 萬參數）：4 個卷積區塊＋全域平均池化＋線性層。Chen（2025）的 CNN
  是兩層卷積後直接攤平接 128 維全連接層，在 224×224 輸入下約 2,500 萬參數，以每 fold
  幾千張圖訓練會嚴重過擬合，因此不照抄。
- 雙圖（K 線＋Volume）時，兩張圖共用同一個 CNN，池化後的特徵串接再分類，與 ViT 的
  雙圖做法對應。
- 不做資料增強：K 線圖旋轉或水平翻轉會改變價格與時間的意義。
- 正規化層預設 GroupNorm（--norm group）。第一版 cnn_2016_t5 用 BatchNorm，診斷發現 eval
  模式的 running statistics 與訓練時的 batch statistics 落差很大（K 線圖幾乎全白、啟動值很稀疏），
  eval 時連 Train 樣本都塌成單一類別，因此 Macro-F1／log loss 失真；GroupNorm 不存 running
  statistics，train／eval 行為一致。--norm batch 保留以重現第一版。
- 其他變體（2026-10-18 起，皆只看 Train 端行為判斷能否學習，不以 Validation 挑設定）：
  - --bn-recal：訓練完後把 BatchNorm 的 running statistics 歸零，以整個 Train 集重新累計
    （累計平均而非 EMA）再進 eval，修正第一版的 train／eval 落差。
  - --invert：像素反轉成黑底亮 K 棒，且不平移，讓背景為 0、只有 K 棒有啟動值。
  - --model paper：照 Chen（2025）的 CNN──Conv(32, 3×3)+ReLU+MaxPool → Conv(64, 3×3)+ReLU+MaxPool
    → 攤平 → FC128+ReLU → 輸出；論文是二元 sigmoid，這裡改成三類 softmax。224×224 輸入下
    約 2,400 萬參數，論文未寫 epochs／lr，沿用本 runner 的設定。
  - --model mobilenetv2：ImageNet 預訓練的 torchvision MobileNetV2 全部微調（論文的另一個影像基線），
    輸入用 ImageNet 均值／標準差正規化；它內含 BatchNorm，建議搭配 --bn-recal。
- 二分類（2026-10-18 起）：--label-mode binary（第 h 天收盤是否高於今日）／binary_anyup（h 天內
  任一天收盤高於今日，推測為 Chen 2025 的定義），定義與 run_horizon_period_grid.py 相同；以
  --vit-run 指向同標籤的 grid run（grid_2016_binary／grid_2016_binary_anyup）做 ViT 配對。
  cnn_single_kvol 讀 grid_2016_kvol 的 K 線＋成交量合併圖（論文的輸入格式）。二分類報告另列
  「漲」類別的 precision／recall／F1，對應論文表格；--no-class-weight 取消類別加權（論文未提加權）。
- 每個模型訓練完都會在 eval 模式下對 Train 集再預測一次，記錄 Train 準確率與最大類別占比，
  用來偵測「eval 時塌成單一類別」這類問題。
- 正式 Test 期間不建立任何樣本。

用法（需先跑過 run_horizon_period_grid.py --run-name grid_2016）：
    python experiments/vision/run_cnn_baseline.py --smoke
    python experiments/vision/run_cnn_baseline.py 2>&1 | tee /tmp/cnn_baseline.log
    python experiments/vision/run_cnn_baseline.py --norm batch --run-name cnn_2016_t5   # 重現第一版
    python experiments/vision/run_cnn_baseline.py --model paper --configs cnn_single_k20 --run-name cnn_paper
    python experiments/vision/run_cnn_baseline.py --model mobilenetv2 --bn-recal --lr 1e-4 \
        --configs cnn_single_k20 --run-name cnn_mnv2
中斷後用相同指令續跑：已完成的 config × fold × seed 會跳過。
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
from experiments.vision.run_horizon_period_grid import (
    LABEL_MODES, OUTPUT_ROOT as GRID_ROOT, WINDOW, build_folds, chart_path, classify_metrics,
    labels_for,
    load_long_ohlcv, majority_metrics, numeric_metrics, paired, sample_dates, unit_mean,
)

PROTOCOL = "cnn_baseline_walk_forward_v1"
OUTPUT_ROOT = paths.OUTPUTS / "experiments" / "cnn_baseline"
CONFIGS = {"cnn_single_k20": ["k20"], "cnn_dual_k20_volume": ["k20", "volume"],
           "cnn_single_kvol": ["kvol"]}
VIT_KEYS = {"vit_single_k20": "lr1e-4_e100|{h}|single_k20",
            "vit_single_kvol": "lr1e-4_e100|{h}|single_kvol",
            "vit_dual_k20_volume": "lr1e-4_e100|{h}|dual_k20_volume"}
COMPARISONS = [("cnn_single_k20", "vit_single_k20"),
               ("cnn_single_kvol", "vit_single_kvol"),
               ("cnn_dual_k20_volume", "vit_dual_k20_volume"),
               ("cnn_dual_k20_volume", "cnn_single_k20")]
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "log_loss")
UP_METRICS = ("up_precision", "up_recall", "up_f1", "up_share")   # 只在二分類時存在


def build_cnn(n_images: int, n_classes: int = 3, norm: str = "group", model: str = "small"):
    import torch.nn as nn

    if model == "paper":
        return build_paper_cnn(n_images, n_classes)
    if model == "mobilenetv2":
        return build_mobilenetv2(n_images, n_classes)

    def block(c_in: int, c_out: int, kernel: int, stride: int) -> list:
        layer = nn.GroupNorm(8, c_out) if norm == "group" else nn.BatchNorm2d(c_out)
        return [nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2, bias=False),
                layer, nn.ReLU(inplace=True)]

    class ChartCNN(nn.Module):
        """224×224 RGB -> 64 維特徵；多張圖共用權重，特徵串接後分類。"""

        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                *block(3, 16, 5, 2), nn.MaxPool2d(2),      # 56×56
                *block(16, 32, 3, 1), nn.MaxPool2d(2),     # 28×28
                *block(32, 64, 3, 1), nn.MaxPool2d(2),     # 14×14
                *block(64, 64, 3, 1),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 * n_images, n_classes))

        def forward(self, images):              # images: [B, n_images, 3, H, W]
            b, n = images.shape[:2]
            features = self.encoder(images.flatten(0, 1)).view(b, n * 64)
            return self.head(features)

    return ChartCNN()


def build_paper_cnn(n_images: int, n_classes: int = 3):
    """Chen（2025）的 CNN：兩層卷積後攤平接 FC128；多張圖共用卷積，攤平特徵串接。"""
    import torch.nn as nn

    class PaperCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 111×111
                nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 54×54
                nn.Flatten(),
            )
            self.head = nn.Sequential(nn.Linear(64 * 54 * 54 * n_images, 128), nn.ReLU(inplace=True),
                                      nn.Linear(128, n_classes))

        def forward(self, images):
            b = images.shape[0]
            return self.head(self.encoder(images.flatten(0, 1)).view(b, -1))

    return PaperCNN()


def build_mobilenetv2(n_images: int, n_classes: int = 3):
    """ImageNet 預訓練 MobileNetV2，全部微調；多張圖共用 backbone，池化特徵串接。"""
    import torch.nn as nn
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    class ChartMobileNet(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280 * n_images, n_classes))

        def forward(self, images):
            b, n = images.shape[:2]
            return self.head(self.encoder(images.flatten(0, 1)).view(b, n * 1280))

    return ChartMobileNet()


def recalibrate_batchnorm(model, x_train: np.ndarray, batch: int, to_tensor) -> None:
    """把 BatchNorm running statistics 換成整個 Train 集的累計平均（權重不變）。"""
    import torch
    import torch.nn as nn

    layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not layers:
        return
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
    model.train()
    with torch.no_grad():
        for start in range(0, len(x_train), batch):
            model(to_tensor(x_train[start:start + batch]))


def load_images(tickers, dates, kinds, grid_output: Path) -> dict:
    """讀 grid run 已畫好的圖為 uint8 陣列 [N, 3, 224, 224]；只讀取，不寫入 grid 目錄。"""
    images = {}
    for ticker in tickers:
        for kind in kinds:
            stack = np.empty((len(dates), 3, 224, 224), dtype=np.uint8)
            for i, date in enumerate(dates):
                path = chart_path(grid_output, ticker, kind, date)
                if not path.exists():
                    raise SystemExit(f"找不到 {path}，先跑 run_horizon_period_grid.py --run-name grid_2016")
                with Image.open(path) as image:
                    stack[i] = np.asarray(image.convert("RGB")).transpose(2, 0, 1)
            images[(ticker, kind)] = stack
            print(f"[cnn_baseline:images] {ticker} {kind} {stack.shape}", flush=True)
    return images


def train_cnn(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, seed: int,
              epochs: int, lr: float, batch: int, device: str,
              norm: str = "group", model_name: str = "small", invert: bool = False,
              bn_recal: bool = False, n_classes: int = 3,
              class_weight: bool = True) -> tuple[np.ndarray, list, dict]:
    """x_*: uint8 [N, n_images, 3, 224, 224]；回傳最後一個 epoch 的 Validation 機率、訓練紀錄，
    以及 eval 模式下對 Train 集的診斷（準確率、最大類別占比）。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_cnn(x_train.shape[1], n_classes, norm=norm, model=model_name).to(device)
    counts = np.bincount(y_train, minlength=n_classes)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(
        len(y_train) / (n_classes * counts), dtype=torch.float32, device=device)
        if class_weight else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    labels = torch.from_numpy(y_train).long()
    generator = torch.Generator().manual_seed(seed)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    def to_tensor(array: np.ndarray):
        x = torch.from_numpy(array).to(device).float().div_(255.0)
        if invert:
            x = 1.0 - x                       # 黑底：背景 0、K 棒為正
        if model_name == "mobilenetv2":
            return (x - mean) / std
        return x if invert else x.sub_(0.5)

    def predict(array: np.ndarray) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(array), batch):
                outputs.append(torch.softmax(model(to_tensor(array[start:start + batch])), 1).cpu())
        return torch.cat(outputs).numpy()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(labels), generator=generator).numpy()
        total, correct = 0.0, 0
        for start in range(0, len(order), batch):
            index = np.sort(order[start:start + batch])
            inputs, targets = to_tensor(x_train[index]), labels[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
            correct += int((logits.argmax(1) == targets).sum())
        history.append({"epoch": epoch, "train_loss": total / len(order),
                        "train_accuracy": correct / len(order)})
    if bn_recal:
        recalibrate_batchnorm(model, x_train, batch, to_tensor)
    train_pred = predict(x_train).argmax(1)
    train_eval = {"accuracy": float(np.mean(train_pred == y_train)),
                  "max_class_share": float(np.bincount(train_pred, minlength=n_classes).max()
                                           / len(train_pred))}
    probabilities = predict(x_val)
    # MPS 不會自動歸還快取：MobileNetV2 連跑多個模型時記憶體一路累積、觸發 swap，
    # 每個模型從 2 分鐘變成 1.5 小時，因此每個模型結束後主動釋放。
    del model, optimizer
    import gc
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return probabilities, history, train_eval


def cnn_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """沿用 grid 的指標；二分類另補預測類別數與「漲」類別的 precision／recall／F1（論文表格）。"""
    metrics = classify_metrics(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    metrics.setdefault("predicted_counts",
                       np.bincount(predictions, minlength=probabilities.shape[1]).tolist())
    if probabilities.shape[1] == 2:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, labels=[1], average=None, zero_division=0)
        metrics.update({"up_precision": float(precision[0]), "up_recall": float(recall[0]),
                        "up_f1": float(f1[0]), "up_share": float(labels.mean())})
    return metrics


def write_markdown(report: dict, output: Path) -> None:
    lines = [
        "# 從頭訓練 CNN 基線 vs 凍結 ViT（walk-forward）", "",
        f"- Protocol：`{report['protocol']}`；正式 Test 期間未建立樣本。",
        f"- 股票：{'／'.join(report['tickers'])}；樣本期間 {report['period'][0]} ～ "
        f"{report['period'][1]}；Walk-forward folds：{'、'.join(report['fold_names'])}；"
        f"統計單位 = fold × 股票，共 {report['n_units']} 個。",
        f"- 標籤：" + {"tercile": f"t+{report['horizon']} 報酬的 Train-only q33／q67 三分類。",
                     "binary": f"二分類：第 {report['horizon']} 天收盤是否高於今日。",
                     "binary_anyup": f"二分類：未來 {report['horizon']} 天內任一天收盤是否高於今日"
                                     "（推測為 Chen 2025 的定義）。"}[report.get("label_mode", "tercile")]
        + ("" if report.get("class_weight", True) else "CE 不做類別加權。"),
        f"- CNN：{report['cnn']['architecture']}；"
        f"{next(iter(report['cnn']['parameters'].values())):,} 參數（{next(iter(report['cnn']['parameters']))}）；"
        f"輸入 {report['cnn']['input']}"
        f"{'；訓練後以 Train 集重算 BatchNorm 統計值' if report['cnn'].get('bn_recal') else ''}。",
        f"- 每個 fold × seed 重新訓練，{report['epochs']} epochs、lr {report['lr']}、"
        f"batch {report['batch']}、seeds {report['seeds']}，回報最後一個 epoch。",
        f"- ViT 為 `{report['vit_source']}` 的 Frozen ViT CLS＋線性頭（lr1e-4×100、5 seeds），同一批單位。",
        "", "## Validation（fold × 股票平均 ± 標準誤）", "",
        "| 輸入 | ROC-AUC | Balanced acc | Accuracy | Macro-F1 | Log loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["summary"].items():
        lines.append(
            f"| {name} | {values['roc_auc']['mean']:.4f} ± {values['roc_auc']['se']:.4f} | "
            f"{values['balanced_accuracy']['mean']:.4f} | {values['accuracy']['mean']:.4f} | "
            f"{values['macro_f1']['mean']:.4f} | {values['log_loss']['mean']:.4f} |")
    lines += ["", "## 配對比較（單位差值平均 ± 標準誤，勝出單位數）", "",
              "| 候選 vs 基準 | ROC-AUC 差 | Macro-F1 差 |", "|---|---|---|"]
    for item in report["comparisons"]:
        auc, f1 = item["roc_auc"], item["macro_f1"]
        lines.append(f"| {item['candidate']} vs {item['baseline']} | "
                     f"{auc['mean']:+.4f} ± {auc['se']:.4f}（{auc['wins']}/{auc['units']}） | "
                     f"{f1['mean']:+.4f} ± {f1['se']:.4f}（{f1['wins']}/{f1['units']}） |")
    if report.get("up_summary"):
        lines += ["", "## 「漲」類別（對應 Chen 2025 表格；各單位平均）", "",
                  "| 輸入 | Precision | Recall | F1 | Validation 漲的比例 |", "|---|---:|---:|---:|---:|"]
        for name, values in report["up_summary"].items():
            lines.append(f"| {name} | {values['up_precision']:.4f} | {values['up_recall']:.4f} | "
                         f"{values['up_f1']:.4f} | {values['up_share']:.4f} |")
        lines.append("")
        lines.append("majority 為永遠猜 Train 多數類（二分類下即全猜漲）。")
    lines += ["", "## 各 fold 的 ROC-AUC（各股票平均）", "",
              "| 輸入 | " + " | ".join(report["fold_names"]) + " |",
              "|---|" + "---:|" * len(report["fold_names"])]
    for name, by_fold in report["by_fold_auc"].items():
        lines.append(f"| {name} | " + " | ".join(
            f"{by_fold[fold]:.3f}" for fold in report["fold_names"]) + " |")
    lines += ["", "## CNN 訓練末期（各 fold × seed 平均）", "",
              "| 輸入 | 最後一個 epoch 的 Train loss | Train accuracy（訓練中） | "
              "Train accuracy（eval 模式） | Validation 最大類別占比 |", "|---|---:|---:|---:|---:|"]
    for name, values in report["final_train"].items():
        eval_acc = values.get("eval_train_accuracy")
        lines.append(f"| {name} | {values['train_loss']:.4f} | {values['train_accuracy']:.4f} | "
                     f"{'—' if eval_acc is None else f'{eval_acc:.4f}'} | "
                     f"{values['val_max_class_share']:.2f} |")
    lines += ["", "## 限制", ""] + [f"- {text}" for text in report["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="cnn_2016_t5_gn")
    parser.add_argument("--grid-run", default="grid_2016", help="讀取圖與 OHLCV 的 grid run")
    parser.add_argument("--vit-run", default=None,
                        help="讀取 ViT 結果的 grid run（預設同 --grid-run；二分類需指向同標籤的 run）")
    parser.add_argument("--label-mode", choices=list(LABEL_MODES), default="tercile")
    parser.add_argument("--no-class-weight", action="store_true", help="CE 不做類別加權")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "NVDA", "MSFT"])
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--download-start", default="2015-09-01")
    parser.add_argument("--fold-years", type=int, nargs="+",
                        default=[2019, 2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--configs", nargs="+", default=["cnn_single_k20", "cnn_dual_k20_volume"],
                        choices=list(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--norm", choices=["group", "batch"], default="group",
                        help="小型 CNN 的正規化層；batch 只用來重現第一版 cnn_2016_t5")
    parser.add_argument("--model", choices=["small", "paper", "mobilenetv2"], default="small")
    parser.add_argument("--invert", action="store_true", help="像素反轉成黑底、背景為 0")
    parser.add_argument("--bn-recal", action="store_true",
                        help="訓練後以整個 Train 集重算 BatchNorm running statistics")
    parser.add_argument("--smoke", action="store_true",
                        help="工程短測：2 folds、1 seed、2 epochs；輸出不作效果證據")
    args = parser.parse_args()
    if args.smoke:
        args.run_name, args.fold_years, args.seeds, args.epochs = "smoke", [2024, 2025], [40], 2

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
    configs = {name: CONFIGS[name] for name in args.configs}
    kinds = sorted({kind for kinds_ in configs.values() for kind in kinds_})
    images = load_images(args.tickers, dates, kinds, grid_output)

    position = {t: {date: i for i, date in enumerate(frames[t].index)} for t in args.tickers}
    returns, windows = {}, {}
    for ticker in args.tickers:
        close = frames[ticker]["Close"].to_numpy(float)
        index = np.asarray([position[ticker][date] for date in dates])
        if args.label_mode == "binary_anyup":
            future = np.max([close[index + k] for k in range(1, args.horizon + 1)], axis=0)
        else:
            future = close[index + args.horizon]
        returns[ticker] = future / close[index] - 1
        windows[ticker] = [frames[ticker].iloc[position[ticker][d] - WINDOW + 1:
                                               position[ticker][d] + 1] for d in dates]

    n_classes = 3 if args.label_mode == "tercile" else 2
    results_path = output / "results.json"
    saved = json.loads(results_path.read_text()) if results_path.exists() else {}
    for config, config_kinds in configs.items():
        for fold in folds:
            labels = {t: labels_for(returns[t], fold, args.label_mode) for t in args.tickers}
            x_train = np.concatenate([np.stack([images[(t, k)][fold["train"]] for k in config_kinds], 1)
                                      for t in args.tickers])
            y_train = np.concatenate([labels[t][0] for t in args.tickers])
            for seed in args.seeds:
                key = f"{config}|{fold['name']}|{seed}"
                if key in saved:
                    continue
                block_started = time.perf_counter()
                x_val = np.concatenate([np.stack([images[(t, k)][fold["validation"]]
                                                  for k in config_kinds], 1)
                                        for t in args.tickers])
                probabilities, history, train_eval = train_cnn(
                    x_train, y_train, x_val, seed, args.epochs, args.lr, args.batch, device,
                    args.norm, args.model, args.invert, args.bn_recal, n_classes,
                    not args.no_class_weight)
                per_ticker, offset = {}, 0
                for ticker in args.tickers:
                    y_val = labels[ticker][1]
                    per_ticker[ticker] = cnn_metrics(y_val, probabilities[offset:offset + len(y_val)])
                    offset += len(y_val)
                saved[key] = {"metrics": per_ticker, "history": history, "train_eval": train_eval}
                results_path.write_text(json.dumps(saved), encoding="utf-8")
                val_share = np.mean([max(m["predicted_counts"]) / sum(m["predicted_counts"])
                                     for m in per_ticker.values()])
                print(f"[cnn_baseline] {key} 完成（{time.perf_counter() - block_started:.0f} 秒；"
                      f"train acc {history[-1]['train_accuracy']:.3f}，eval 模式 train acc "
                      f"{train_eval['accuracy']:.3f}；val 最大類別占比 {val_share:.2f}）", flush=True)

    # fold × 股票單位：CNN 先在 seeds 間平均；ViT 直接讀 grid run 的單位結果。
    units = {}
    for config in configs:
        units[config] = {}
        for fold in folds:
            for ticker in args.tickers:
                per_seed = [saved[f"{config}|{fold['name']}|{s}"]["metrics"][ticker]
                            for s in args.seeds]
                units[config][f"{fold['name']}|{ticker}"] = {
                    m: float(np.mean([item[m] for item in per_seed]))
                    for m in METRICS + (UP_METRICS if n_classes == 2 else ())}
    vit_run = args.vit_run or args.grid_run
    vit_results_path = GRID_ROOT / vit_run / "run" / "results.json"
    vit_results = json.loads(vit_results_path.read_text()) if vit_results_path.exists() else {}
    for name, template in VIT_KEYS.items():
        key = template.format(h=args.horizon)
        if key in vit_results and not args.smoke:
            units[name] = {u: v for u, v in vit_results[key].items() if u in units[args.configs[0]]}
    numeric_units, majority_units = {}, {}
    for fold in folds:
        for ticker in args.tickers:
            y_train, y_val, _ = labels_for(returns[ticker], fold, args.label_mode)
            unit = f"{fold['name']}|{ticker}"
            numeric_units[unit] = numeric_metrics(windows[ticker], fold, y_train, y_val,
                                                  args.seeds[0])
            majority_units[unit] = majority_metrics(y_train, y_val, n_classes)
            if n_classes == 2:            # 多數類為「漲」時：precision = 漲的比例、recall = 1
                up, guess_up = float(y_val.mean()), bool(np.bincount(y_train, minlength=2).argmax())
                majority_units[unit].update({
                    "up_precision": up if guess_up else 0.0, "up_recall": float(guess_up),
                    "up_f1": 2 * up / (1 + up) if guess_up else 0.0, "up_share": up})
    units["numeric_logistic_w20"] = numeric_units
    units["majority"] = majority_units

    summary = {name: unit_mean(per_unit) for name, per_unit in units.items()}
    up_summary = {name: {m: float(np.mean([u[m] for u in per_unit.values()])) for m in UP_METRICS}
                  for name, per_unit in units.items()
                  if all(m in u for u in per_unit.values() for m in UP_METRICS)}
    comparisons = [{"candidate": c, "baseline": b,
                    "roc_auc": paired(units[c], units[b], "roc_auc"),
                    "macro_f1": paired(units[c], units[b], "macro_f1")}
                   for c, b in COMPARISONS if c in units and b in units]
    by_fold_auc = {name: {fold["name"]: float(np.mean(
        [per_unit[f"{fold['name']}|{t}"]["roc_auc"] for t in args.tickers])) for fold in folds}
        for name, per_unit in units.items() if name != "majority"}
    final_train = {}
    for config in configs:
        entries = [saved[f"{config}|{fold['name']}|{s}"] for fold in folds for s in args.seeds]
        final_train[config] = {key: float(np.mean([e["history"][-1][key] for e in entries]))
                               for key in ("train_loss", "train_accuracy")}
        final_train[config]["eval_train_accuracy"] = (
            float(np.mean([e["train_eval"]["accuracy"] for e in entries]))
            if all("train_eval" in e for e in entries) else None)
        final_train[config]["val_max_class_share"] = float(np.mean([
            max(m["predicted_counts"]) / sum(m["predicted_counts"])
            for e in entries for m in e["metrics"].values()]))
    parameters = {config: int(sum(p.numel() for p in build_cnn(
        len(k), norm=args.norm, model=args.model).parameters())) for config, k in configs.items()}

    report = {
        "protocol": PROTOCOL, "smoke": args.smoke, "tickers": args.tickers,
        "period": [f"{dates[0]:%Y-%m-%d}", f"{dates[-1]:%Y-%m-%d}"], "horizon": args.horizon,
        "fold_names": [fold["name"] for fold in folds],
        "fold_sizes": {fold["name"]: {"train": int(len(fold["train"])),
                                      "validation": int(len(fold["validation"]))}
                       for fold in folds},
        "n_units": len(folds) * len(args.tickers), "configs": configs,
        "cnn": {"parameters": parameters, "input": (
                    "224×224 RGB，ImageNet 均值／標準差正規化" if args.model == "mobilenetv2"
                    else "224×224 RGB，" + ("像素反轉（黑底、背景 0）" if args.invert else "像素 /255 − 0.5"))
                    + ("；先反轉成黑底" if args.invert and args.model == "mobilenetv2" else ""),
                "norm": args.norm, "model": args.model, "invert": args.invert,
                "bn_recal": args.bn_recal, "architecture": {
                    "small": f"4×(Conv-{'GN' if args.norm == 'group' else 'BN'}-ReLU)，"
                             "前三個後接 MaxPool；GAP；Dropout 0.3；Linear",
                    "paper": "Chen（2025）：Conv32-ReLU-MaxPool → Conv64-ReLU-MaxPool → 攤平 → "
                             "FC128-ReLU → Linear(3)",
                    "mobilenetv2": "ImageNet 預訓練 MobileNetV2 全部微調；GAP；Dropout 0.2；Linear",
                }[args.model]},
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "seeds": args.seeds,
        "vit_source": f"{vit_run}/run/results.json", "test_was_opened": False,
        "label_mode": args.label_mode, "n_classes": n_classes,
        "class_weight": not args.no_class_weight, "grid_run": args.grid_run,
        "summary": summary, "comparisons": comparisons, "by_fold_auc": by_fold_auc,
        "up_summary": up_summary, "final_train": final_train, "elapsed_seconds": round(time.perf_counter() - started, 1),
        "limitations": [
            "fold × 股票單位彼此相關（同一年各股票常同漲同跌），標準誤偏樂觀。",
            "CNN 超參數（架構、epochs、lr）為事前設定，未以 Validation 調整；不同設定可能得到不同結果。",
            "同一批 Validation 上比較了多個 CNN 變體，最好的那個會因多重比較而偏樂觀。",
            "CNN 用三檔股票合併訓練，ViT 線性頭則是每檔各自訓練；兩者的訓練資料量不同。",
            "MPS 上的卷積訓練不保證位元級可重現，重跑數字可能有極小差異。",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    write_markdown(report, output)
    print(f"[cnn_baseline] report -> {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
