"""科技巨頭股票留出 ViT 實驗的圖表與 Markdown。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.vision.visualize_comparison_report import configure_style, save_figure


LABELS = ["DOWN", "NEUTRAL", "UP"]
MODE_LABELS = {
    "single": "K-line",
    "dual_volume": "K-line + Volume",
    "dual_sma": "K-line + SMA",
}


def matrix_for(report: dict, field: str) -> np.ndarray:
    return np.asarray([
        [report["aggregate"][f"w{window}_{mode}"][field]["mean"]
         for mode in report["input_modes"]]
        for window in report["windows"]
    ])


def annotate_heatmap(axis, matrix: np.ndarray):
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, f"{matrix[row, column]:.3f}",
                ha="center", va="center",
                color="white" if matrix[row, column] < np.nanmedian(matrix) else "black",
            )


def generate_figures(report: dict, output_dir: Path, dpi: int) -> list[Path]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    windows = report["windows"]
    modes = report["input_modes"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)
    definitions = (
        ("mean_ticker_macro_f1", "Mean-ticker Macro-F1", "A  Primary metric"),
        ("balanced_accuracy", "Balanced accuracy", "B  Balanced accuracy"),
        ("roc_auc_ovr_macro", "Macro OvR AUC", "C  Ranking quality"),
    )
    for axis, (field, colorbar_label, title) in zip(axes, definitions):
        matrix = matrix_for(report, field)
        image = axis.imshow(matrix, cmap="cividis", aspect="auto")
        annotate_heatmap(axis, matrix)
        axis.set_xticks(range(len(modes)), [MODE_LABELS[mode] for mode in modes], rotation=20)
        axis.set_yticks(range(len(windows)), windows)
        axis.set_xlabel("Input mode")
        axis.set_ylabel("K-bar window")
        axis.set_title(title, loc="left")
        fig.colorbar(image, ax=axis, shrink=0.78, label=colorbar_label)
    fig.suptitle("Tech-giant Frozen ViT ablation — stock-held-out folds, Test LOCKED")
    paths.append(save_figure(fig, output_dir, "overview", dpi))

    selected = report["selected"]
    selected_value = report["aggregate"][selected]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    matrix = np.asarray(selected_value["confusion_matrix_sum"], dtype=float)
    normalized = np.divide(
        matrix, matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    axes[0].imshow(normalized, cmap="cividis", vmin=0, vmax=1)
    for row in range(3):
        for column in range(3):
            value = normalized[row, column]
            axes[0].text(
                column, row, f"{int(matrix[row, column])}\n{value:.0%}",
                ha="center", va="center", color="white" if value < 0.45 else "black",
            )
    axes[0].set_xticks(range(3), LABELS, rotation=15)
    axes[0].set_yticks(range(3), LABELS)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"A  {selected} pooled confusion matrix", loc="left")

    ticker_stats = selected_value["by_ticker_macro_f1"]
    tickers = list(ticker_stats)
    means = [ticker_stats[ticker]["mean"] for ticker in tickers]
    stds = [ticker_stats[ticker]["std"] for ticker in tickers]
    axes[1].bar(np.arange(len(tickers)), means, yerr=stds, capsize=3, color="#0072B2")
    axes[1].axhline(1 / 3, color="#888888", linestyle="--", linewidth=0.9)
    axes[1].set_xticks(np.arange(len(tickers)), tickers, rotation=35)
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("B  Held-out ticker results across seeds", loc="left")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    paths.append(save_figure(fig, output_dir, "selected_diagnostics", dpi))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    x = np.arange(len(windows))
    width = 0.8 / len(modes)
    for index, mode in enumerate(modes):
        offset = (index - (len(modes) - 1) / 2) * width
        means = [report["aggregate"][f"w{window}_{mode}"]["mean_ticker_macro_f1"]["mean"]
                 for window in windows]
        stds = [report["aggregate"][f"w{window}_{mode}"]["mean_ticker_macro_f1"]["std"]
                for window in windows]
        axes[0].bar(x + offset, means, width, yerr=stds, capsize=2, label=MODE_LABELS[mode])
    axes[0].axhline(1 / 3, color="#888888", linestyle="--", linewidth=0.9)
    axes[0].set_xticks(x, windows)
    axes[0].set_xlabel("K-bar window")
    axes[0].set_ylabel("Mean-ticker Macro-F1")
    axes[0].set_title("A  Input-mode comparison", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)

    timing = report.get("feature_extraction", {})
    labels, values = [], []
    for window in windows:
        for image_type in ("candlestick", "volume", "sma"):
            key = f"w{window}_{image_type}"
            if key in timing:
                labels.append(f"w{window}\n{image_type}")
                values.append(timing[key]["seconds"] / 60)
    axes[1].bar(np.arange(len(labels)), values, color="#009E73")
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=25)
    axes[1].set_ylabel("Minutes")
    axes[1].set_title("B  Frozen feature extraction", loc="left")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    paths.append(save_figure(fig, output_dir, "input_and_cost", dpi))
    return paths


def append_markdown(report: dict, lines: list[str]) -> None:
    from experiments.vision.render_experiment_report import markdown_table

    lines.extend([
        "## 實驗設定", "",
        "- 狀態：科技巨頭 Frozen ViT screening；Test 全程 locked。",
        f"- 股票：{', '.join(item['ticker'] for item in report['tickers'])}。",
        "- 股票驗證：4 folds；每 fold 6 檔 Train、2 檔完全未見 Validation。",
        f"- 窗口：{', '.join(str(value) for value in report['windows'])}。",
        "- 輸入：K 線單圖、K 線＋Volume、K 線＋SMA5/10/20。",
        f"- Seeds：{', '.join(str(value) for value in report['seeds'])}；每組 {report['epochs']} epochs。",
        "- Label：每 fold 只依該 fold Train 股票的 Train-period 5 日報酬 q33／q67 定義三分類。",
        f"- 目前選定：`{report['selected']}`。", "",
        "## 股票留出 folds", "",
    ])
    lines.extend(markdown_table(
        ["Fold", "Held-out Validation stocks"],
        [[fold["name"], ", ".join(fold["holdout"])] for fold in report["folds"]],
    ))

    lines.extend(["", "## Validation unseen 彙整", ""])
    rows = []
    for candidate in report["candidates"]:
        value = report["aggregate"][candidate]
        def mean_std(field):
            return f"{value[field]['mean']:.4f} ± {value[field]['std']:.4f}"
        rows.append([
            candidate + (" ★" if candidate == report["selected"] else ""),
            mean_std("mean_ticker_macro_f1"), mean_std("macro_f1"),
            mean_std("accuracy"), mean_std("balanced_accuracy"),
            mean_std("roc_auc_ovr_macro"), mean_std("log_loss"),
        ])
    lines.extend(markdown_table(
        ["Candidate", "Mean-ticker F1", "Pooled F1", "Accuracy", "Balanced acc.",
         "Macro OvR AUC", "Log loss"], rows,
    ))

    lines.extend(["", "## 固定 Baseline（跨 fold）", ""])
    baseline_rows = []
    for name, value in report["baseline_aggregate"].items():
        def mean_std(field):
            return f"{value[field]['mean']:.4f} ± {value[field]['std']:.4f}"
        baseline_rows.append([
            name, mean_std("mean_ticker_macro_f1"), mean_std("macro_f1"),
            mean_std("accuracy"), mean_std("balanced_accuracy"),
            mean_std("roc_auc_ovr_macro"), mean_std("log_loss"),
        ])
    lines.extend(markdown_table(
        ["Baseline", "Mean-ticker F1", "Pooled F1", "Accuracy", "Balanced acc.",
         "Macro OvR AUC", "Log loss"], baseline_rows,
    ))

    selected = report["aggregate"][report["selected"]]
    lines.extend(["", "## 選定設定的 held-out 股票", ""])
    ticker_rows = [
        [ticker, value["mean"], value["std"], value["min"], value["max"]]
        for ticker, value in selected["by_ticker_macro_f1"].items()
    ]
    lines.extend(markdown_table(
        ["Ticker", "Mean F1", "Std", "Min", "Max"], ticker_rows,
    ))
