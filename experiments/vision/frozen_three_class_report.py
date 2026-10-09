"""Train-only quantile 三分類 Frozen ViT 實驗的圖表與 Markdown。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.vision.visualize_comparison_report import configure_style, save_figure


LABELS = ["DOWN", "NEUTRAL", "UP"]
COLORS = ["#0072B2", "#E69F00", "#009E73"]


def generate_figures(report: dict, output_dir: Path, dpi: int) -> list[Path]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = report["windows"]
    aggregate = report["aggregate"]
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    definitions = (
        ("mean_ticker_macro_f1", "Mean-ticker Macro-F1", "A  Primary metric"),
        ("balanced_accuracy", "Balanced accuracy", "B  Class-balanced accuracy"),
        ("roc_auc_ovr_macro", "Macro OvR ROC-AUC", "C  Ranking quality"),
    )
    for axis, (field, ylabel, title) in zip(axes, definitions):
        means = [aggregate[str(window)][field]["mean"] for window in windows]
        stds = [aggregate[str(window)][field]["std"] for window in windows]
        axis.errorbar(
            windows, means, yerr=stds, marker="o", capsize=4,
            color="#0072B2", linewidth=1.8,
        )
        axis.axhline(1 / 3 if field != "roc_auc_ovr_macro" else 0.5,
                     color="#888888", linestyle="--", linewidth=0.9)
        baseline = report["baselines"]["numeric_logistic_window20"]["validation_unseen"]
        baseline_field = field
        axis.axhline(
            baseline[baseline_field], color="#D55E00", linestyle=":", linewidth=1.2,
            label="Numeric Logistic",
        )
        axis.set_xticks(windows)
        axis.set_xlabel("K-bar window")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.grid(color="#DDDDDD", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    mode = report.get("input_mode", "single")
    fig.suptitle(
        f"Frozen ViT train-only quantile three-class experiment ({mode}) — Test LOCKED"
    )
    paths.append(save_figure(fig, output_dir, "overview", dpi))

    selected = str(report["selected_window"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    matrix = np.asarray(aggregate[selected]["confusion_matrix_sum"], dtype=float)
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
    axes[0].set_title(f"A  w{selected}, seed-pooled confusion matrix", loc="left")

    x = np.arange(len(LABELS))
    means = [aggregate[selected]["class_f1"][label]["mean"] for label in LABELS]
    stds = [aggregate[selected]["class_f1"][label]["std"] for label in LABELS]
    axes[1].bar(x, means, yerr=stds, capsize=4, color=COLORS)
    axes[1].set_xticks(x, LABELS)
    axes[1].set_ylabel("F1")
    axes[1].set_title(f"B  w{selected} class F1 across seeds", loc="left")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    paths.append(save_figure(fig, output_dir, "selected_window_diagnostics", dpi))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for seed in report["seeds"]:
        result = report["results"][f"w{selected}_seed{seed}"]
        epochs = [entry["epoch"] for entry in result["history"]]
        axes[0].plot(
            epochs, [entry["train_loss"] for entry in result["history"]],
            marker="o", alpha=0.75, label=f"seed{seed}",
        )
        axes[1].plot(
            epochs,
            [entry["validation"]["validation_unseen"]["mean_ticker_macro_f1"]
             for entry in result["history"]],
            marker="o", alpha=0.75, label=f"seed{seed}",
        )
    for axis, title, ylabel in (
        (axes[0], "A  Classification-head loss", "Weighted loss"),
        (axes[1], "B  Unseen-validation selection metric", "Mean-ticker Macro-F1"),
    ):
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.grid(color="#DDDDDD", linewidth=0.6)
        axis.legend(frameon=False, fontsize=8)
    paths.append(save_figure(fig, output_dir, "training_curves", dpi))

    counts = report["label_policy"]["counts"]
    fig, axis = plt.subplots(figsize=(8.3, 4.2), constrained_layout=True)
    split_names = ["train", "validation_seen", "validation_unseen"]
    x = np.arange(len(split_names))
    bottom = np.zeros(len(split_names))
    for class_id, (label, color) in enumerate(zip(LABELS, COLORS)):
        values = np.asarray([counts[split][class_id] for split in split_names], dtype=float)
        totals = np.asarray([sum(counts[split]) for split in split_names], dtype=float)
        fractions = values / totals
        axis.bar(x, fractions, bottom=bottom, color=color, label=label)
        bottom += fractions
    axis.set_xticks(x, ["Train", "Validation seen", "Validation unseen"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Class proportion")
    axis.set_title("Train-only thresholds applied without validation rebalancing")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    paths.append(save_figure(fig, output_dir, "label_distribution", dpi))
    return paths


def append_markdown(report: dict, lines: list[str]) -> None:
    from experiments.vision.render_experiment_report import markdown_table

    policy = report["label_policy"]
    mode = report.get("input_mode", "single")
    input_description = "K 線圖" if mode == "single" else "K 線圖＋Volume 圖（CLS 串接）"
    lines.extend([
        "## 實驗設定", "",
        "- 狀態：正式 Frozen 三分類 screening；Test 全程 locked。",
        "- Label：只以 Train 的 5 日 forward return q33／q67 決定 DOWN／NEUTRAL／UP。",
        f"- Train-only 門檻：DOWN < {policy['lower']:.6%}；UP > {policy['upper']:.6%}；其餘為 NEUTRAL。",
        f"- K 棒窗口：{', '.join(str(value) for value in report['windows'])}；輸入：{input_description}。",
        f"- Seeds：{', '.join(str(value) for value in report['seeds'])}；每組 {report['epochs']} epochs。",
        f"- Input mode：`{mode}`；Frozen CLS 直接沿用二分類窗口實驗的 feature cache，沒有重新執行 ViT。",
        f"- 目前選定窗口：`w{report['selected_window']}`（依五 seed 平均 unseen mean-ticker Macro-F1）。",
        "", "## Label 分布", "",
    ])
    count_rows = []
    for split, values in policy["counts"].items():
        total = sum(values)
        count_rows.append([
            split, total,
            f"{values[0]} ({values[0] / total:.1%})",
            f"{values[1]} ({values[1] / total:.1%})",
            f"{values[2]} ({values[2] / total:.1%})",
        ])
    lines.extend(markdown_table(
        ["Split", "N", "DOWN", "NEUTRAL", "UP"], count_rows,
    ))

    lines.extend(["", "## 五 seed Validation unseen 結果", ""])
    rows = []
    for window in report["windows"]:
        value = report["aggregate"][str(window)]
        def mean_std(field):
            return f"{value[field]['mean']:.4f} ± {value[field]['std']:.4f}"
        rows.append([
            f"w{window}" + (" ★" if window == report["selected_window"] else ""),
            mean_std("mean_ticker_macro_f1"), mean_std("macro_f1"),
            mean_std("accuracy"), mean_std("balanced_accuracy"),
            mean_std("roc_auc_ovr_macro"), mean_std("log_loss"),
        ])
    lines.extend(markdown_table(
        ["Window", "Mean-ticker F1", "Pooled F1", "Accuracy",
         "Balanced acc.", "Macro OvR AUC", "Log loss"], rows,
    ))

    selected = report["selected_window"]
    lines.extend(["", f"## w{selected} 各 seed", ""])
    seed_rows = []
    for seed in report["seeds"]:
        result = report["results"][f"w{selected}_seed{seed}"]
        value = result["best_validation"]["validation_unseen"]
        seed_rows.append([
            seed, result["best_epoch"], value["mean_ticker_macro_f1"],
            value["balanced_accuracy"], value["roc_auc_ovr_macro"], value["log_loss"],
        ])
    lines.extend(markdown_table(
        ["Seed", "Best epoch", "Mean-ticker F1", "Balanced acc.",
         "Macro OvR AUC", "Log loss"], seed_rows,
    ))

    lines.extend(["", "## 固定 Baseline", ""])
    baseline_rows = []
    for name, groups in report["baselines"].items():
        for split, value in groups.items():
            baseline_rows.append([
                name, split, value["mean_ticker_macro_f1"], value["macro_f1"],
                value["accuracy"], value["balanced_accuracy"],
                value["roc_auc_ovr_macro"], value["log_loss"],
            ])
    lines.extend(markdown_table(
        ["Baseline", "Split", "Mean-ticker F1", "Pooled F1", "Accuracy",
         "Balanced acc.", "Macro OvR AUC", "Log loss"], baseline_rows,
    ))

    lines.extend([
        "", "## 保存內容", "",
        "每個 window／seed 的最佳 head checkpoint、history、逐筆 Validation predictions、"
        "原始 JSON 與圖表皆保存在本報告目錄。", "",
    ])
