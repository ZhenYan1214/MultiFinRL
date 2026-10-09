"""Frozen ViT window × input-mode 消融的圖表與 Markdown 區段。"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.vision.visualize_comparison_report import configure_style, save_figure


MODE_COLORS = {"single": "#0072B2", "dual": "#D55E00"}


def _ordered_results(report: dict):
    for window in report["windows"]:
        for mode in report["input_modes"]:
            name = f"w{window}_{mode}"
            if name in report["results"]:
                yield name, report["results"][name]


def generate_figures(report: dict, output_dir: Path, dpi: int) -> list[Path]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    windows = report["windows"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), constrained_layout=True)
    definitions = (
        ("mean_ticker_macro_f1", "Mean-ticker Macro-F1", "A  Primary selection metric"),
        ("roc_auc", "ROC-AUC", "B  Ranking quality"),
        ("balanced_accuracy", "Balanced accuracy", "C  Class-balanced accuracy"),
    )
    for axis, (metric, ylabel, title) in zip(axes.flat[:3], definitions):
        for mode in report["input_modes"]:
            x_values, y_values = [], []
            for window in windows:
                result = report["results"].get(f"w{window}_{mode}")
                if result:
                    x_values.append(window)
                    y_values.append(
                        result["best_validation"]["validation_unseen"].get(metric, np.nan)
                    )
            axis.plot(
                x_values, y_values, marker="o", linewidth=1.8,
                color=MODE_COLORS[mode], label=mode,
            )
        axis.axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
        axis.set_xticks(windows)
        axis.set_xlabel("K-bar window")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.grid(color="#DDDDDD", linewidth=0.6)
        axis.legend(frameon=False)

    deltas = report.get("comparisons", {})
    delta_x = [window for window in windows if str(window) in deltas]
    delta_y = [
        deltas[str(window)]["dual_minus_single_mean_ticker_macro_f1"]
        for window in delta_x
    ]
    axes[1, 1].bar(
        delta_x, delta_y,
        color=["#009E73" if value >= 0 else "#CC79A7" for value in delta_y],
    )
    axes[1, 1].axhline(0, color="#333333", linewidth=0.8)
    axes[1, 1].set_xticks(windows)
    axes[1, 1].set_xlabel("K-bar window")
    axes[1, 1].set_ylabel("Dual − Single Macro-F1")
    axes[1, 1].set_title("D  Volume-image contribution", loc="left")
    axes[1, 1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle(
        "Frozen ViT window × input-mode ablation — "
        + ("SMOKE / not efficacy evidence" if report["smoke"] else "Test LOCKED"),
        fontsize=13,
    )
    paths.append(save_figure(fig, output_dir, "overview", dpi))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    histories = 0
    for _name, result in _ordered_results(report):
        history = result["history"]
        epochs = [entry["epoch"] for entry in history]
        mode = result["input_mode"]
        label = f"w{result['window_days']} {mode}"
        axes[0].plot(
            epochs, [entry["train_loss"] for entry in history], marker="o",
            color=MODE_COLORS[mode], alpha=0.35 + result["window_days"] / 50,
            label=label,
        )
        axes[1].plot(
            epochs,
            [entry["validation"]["validation_unseen"]["mean_ticker_macro_f1"]
             for entry in history],
            marker="o", color=MODE_COLORS[mode],
            alpha=0.35 + result["window_days"] / 50, label=label,
        )
        histories += 1
    for axis, title, ylabel in (
        (axes[0], "A  Classification-head training", "Weighted loss"),
        (axes[1], "B  Unseen validation", "Mean-ticker Macro-F1"),
    ):
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.grid(color="#DDDDDD", linewidth=0.6)
    if histories:
        axes[1].legend(frameon=False, fontsize=7, ncol=2)
        paths.append(save_figure(fig, output_dir, "training_curves", dpi))
    else:
        plt.close(fig)

    panels = list(_ordered_results(report))
    if panels:
        columns = min(5, len(panels))
        rows = math.ceil(len(panels) / columns)
        fig, axes = plt.subplots(
            rows, columns, figsize=(3.1 * columns, 3.0 * rows),
            squeeze=False, constrained_layout=True,
        )
        for axis, (name, result) in zip(axes.flat, panels):
            counts = np.asarray(
                result["best_validation"]["validation_unseen"]["confusion_matrix"],
                dtype=float,
            )
            totals = counts.sum(axis=1, keepdims=True)
            normalized = np.divide(
                counts, totals, out=np.zeros_like(counts), where=totals > 0,
            )
            axis.imshow(normalized, cmap="cividis", vmin=0, vmax=1)
            for row in range(2):
                for column in range(2):
                    value = normalized[row, column]
                    axis.text(
                        column, row, f"{int(counts[row, column])}\n{value:.0%}",
                        ha="center", va="center",
                        color="white" if value < 0.45 else "black",
                    )
            axis.set_xticks([0, 1], ["DOWN", "UP"])
            axis.set_yticks([0, 1], ["DOWN", "UP"])
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            axis.set_title(name)
        for axis in axes.flat[len(panels):]:
            axis.axis("off")
        fig.suptitle("Unseen-validation confusion matrices", fontsize=13)
        paths.append(save_figure(fig, output_dir, "confusion_matrices", dpi))

    timings = report.get("feature_extraction", {})
    if timings:
        fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        x = np.arange(len(windows))
        width = 0.34
        for index, image_type in enumerate(("candlestick", "volume")):
            values = [
                timings.get(f"w{window}_{image_type}", {}).get("seconds", np.nan)
                for window in windows
            ]
            axis.bar(
                x + (index - 0.5) * width, np.asarray(values) / 60, width,
                color=MODE_COLORS["single" if image_type == "candlestick" else "dual"],
                label=image_type,
            )
        axis.set_xticks(x, windows)
        axis.set_xlabel("K-bar window")
        axis.set_ylabel("Feature extraction minutes")
        axis.set_title("Frozen ViT feature extraction cost")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.legend(frameon=False)
        paths.append(save_figure(fig, output_dir, "feature_extraction_cost", dpi))
    return paths


def append_markdown(report: dict, lines: list[str]) -> None:
    from experiments.vision.render_experiment_report import markdown_table

    source = report["source_config"]
    lines.extend([
        "## 實驗設定", "",
        f"- 狀態：{'SMOKE，不能判斷效果' if report['smoke'] else '正式 Frozen screening'}；Test 全程 locked。",
        "- 搜尋類型：有限範圍 grid search，同時也是 window × input-mode ablation。",
        f"- K 棒窗口：{', '.join(str(value) for value in report['windows'])}。",
        "- 單圖：K 線；雙圖：K 線＋Volume；兩張圖共用相同 Frozen ViT。",
        f"- Seed：{report['seed']}；epochs：{report['epochs']}；主指標：`{report['selection_metric']}`。",
        f"- 共用樣本：{report['common_sample_universe']}",
        f"- 日期：train [{source['train_start']}, {source['validation_start']})；"
        f"validation [{source['validation_start']}, {source['test_start']})。",
        f"- 目前選定：`{report.get('selected') or '尚未完成'}`。", "",
        "## Validation 結果", "",
    ])
    result_rows = []
    for name, result in _ordered_results(report):
        value = result["best_validation"]["validation_unseen"]
        result_rows.append([
            name + (" ★" if name == report.get("selected") else ""),
            result["best_epoch"], value["mean_ticker_macro_f1"], value["macro_f1"],
            value["accuracy"], value["balanced_accuracy"], value.get("roc_auc"),
            value["log_loss"], result["seconds"],
        ])
    lines.extend(markdown_table(
        ["設定", "Best epoch", "Mean-ticker F1", "Pooled F1", "Accuracy",
         "Balanced acc.", "ROC-AUC", "Log loss", "Head 秒數"],
        result_rows,
    ))

    lines.extend(["", "## 雙圖相對單圖", ""])
    lines.extend(markdown_table(
        ["Window", "Dual − Single mean-ticker Macro-F1"],
        [[window, values["dual_minus_single_mean_ticker_macro_f1"]]
         for window, values in report.get("comparisons", {}).items()],
    ))

    lines.extend(["", "## 固定 Baseline", ""])
    baseline_rows = []
    for name, groups in report.get("baselines", {}).items():
        for group, value in groups.items():
            baseline_rows.append([
                name, group, value["mean_ticker_macro_f1"], value["accuracy"],
                value["balanced_accuracy"], value.get("roc_auc"), value["log_loss"],
            ])
    lines.extend(markdown_table(
        ["Baseline", "Split", "Mean-ticker F1", "Accuracy", "Balanced acc.",
         "ROC-AUC", "Log loss"], baseline_rows,
    ))

    lines.extend(["", "## Frozen 特徵抽取", ""])
    timing_rows = []
    for key, value in sorted(report.get("feature_extraction", {}).items()):
        misses = sum(item["ticker_cache_misses"] for item in value["splits"].values())
        hits = sum(item["ticker_cache_hits"] for item in value["splits"].values())
        timing_rows.append([key, value["seconds"] / 60, hits, misses])
    lines.extend(markdown_table(
        ["特徵", "本次分鐘", "Ticker cache hits", "Ticker cache misses"], timing_rows,
    ))
    lines.extend([
        "",
        "`result.json`、最佳 head checkpoint、逐筆 validation predictions 與訓練 history "
        "皆保存在各設定子目錄。Frozen feature cache 可供相同協定續跑使用。",
        "",
    ])
