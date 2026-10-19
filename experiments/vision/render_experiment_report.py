"""將 vision 實驗 JSON 自動轉成圖表與可閱讀的 Markdown 報告。

單份報告：

    python experiments/vision/render_experiment_report.py \
        data/outputs/experiments/maxvit_s_comparison/AAPL_NVDA_MSFT/run/report.json

補產全部舊報告：

    python experiments/vision/render_experiment_report.py \
        --all data/outputs/experiments

預設在 JSON 旁建立 ``figures/*.png`` 與 ``report.md``。這支程式只讀取既有
結果，不會重新訓練、重新選模或開啟 Test。
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MPL_CACHE = Path(tempfile.gettempdir()) / "multifinrl-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.vision.visualize_comparison_report import (
    COLORS,
    LABELS,
    configure_style,
    display_name,
    ensure_aggregate_fields,
    generate_visualizations as generate_comparison_visualizations,
    save_figure,
)


def detect_report_type(report: dict) -> str:
    if report.get("protocol") == "cross_stock_shared_head_binary_v1":
        return "cross_stock"
    if report.get("protocol") == "frozen_backbone_validation_only_no_test":
        return "backbone_comparison"
    if "validation_results" in report and "candidates" in report:
        return "auxiliary_vision"
    if "per_seed" in report and "aggregate" in report and "strategies" in report:
        return "multiseed_summary"
    if "results" in report and "strategies" in report:
        return "vit_adaptation"
    raise ValueError("不支援的 vision report 格式")


def primary_metric(metrics: dict) -> float:
    if "mean_ticker_macro_f1" in metrics:
        return float(metrics["mean_ticker_macro_f1"])
    return float(metrics["macro_f1"])


def normalized_confusion(metrics: dict) -> np.ndarray:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def draw_confusion(ax, metrics: dict, title: str):
    matrix = normalized_confusion(metrics)
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="cividis")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column, row, f"{value:.0%}", ha="center", va="center",
                color="white" if value > 0.48 else "black", fontweight="bold",
            )
    ax.set_xticks(range(len(LABELS)), [value.title() for value in LABELS], rotation=25)
    ax.set_yticks(range(len(LABELS)), [value.title() for value in LABELS])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    return image


def generate_adaptation_figures(
    report: dict, output_dir: Path, dpi: int
) -> list[Path]:
    strategies = report["strategies"]
    results = report["results"]
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.3))
    x = np.arange(len(strategies))
    colors = [COLORS[i % len(COLORS)] for i in range(len(strategies))]
    validations = [results[name]["best_validation"] for name in strategies]
    values = [primary_metric(value) for value in validations]
    bars = axes[0, 0].bar(x, values, color=colors)
    axes[0, 0].set_xticks(x, strategies, rotation=20)
    axes[0, 0].set_ylabel("Selection metric")
    axes[0, 0].set_title("A  Best validation result", loc="left")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for bar, value, strategy in zip(bars, values, strategies):
        marker = " ★" if strategy == report.get("selected") else ""
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}{marker}",
            ha="center", va="bottom", fontsize=8,
        )

    metrics = ("accuracy", "balanced_accuracy", "macro_f1")
    width = 0.8 / len(strategies)
    metric_x = np.arange(len(metrics))
    for index, (strategy, validation) in enumerate(zip(strategies, validations)):
        offset = (index - (len(strategies) - 1) / 2) * width
        axes[0, 1].bar(
            metric_x + offset,
            [validation.get(metric, np.nan) for metric in metrics],
            width, color=colors[index], label=strategy,
        )
    axes[0, 1].set_xticks(metric_x, ["Accuracy", "Balanced acc.", "Macro F1"])
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].set_title("B  Validation metrics", loc="left")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", color="#DDDDDD", linewidth=0.6)

    class_x = np.arange(len(LABELS))
    for index, (strategy, validation) in enumerate(zip(strategies, validations)):
        offset = (index - (len(strategies) - 1) / 2) * width
        axes[1, 0].bar(
            class_x + offset,
            [validation["detail"][label]["f1-score"] for label in LABELS],
            width, color=colors[index], label=strategy,
        )
    axes[1, 0].set_xticks(class_x, [value.title() for value in LABELS])
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].set_title("C  Validation F1 by class", loc="left")
    axes[1, 0].grid(axis="y", color="#DDDDDD", linewidth=0.6)

    trainable = [
        results[name].get("parameter_counts", {}).get("trainable", np.nan)
        for name in strategies
    ]
    bars = axes[1, 1].bar(x, np.asarray(trainable) / 1e6, color=colors)
    axes[1, 1].set_xticks(x, strategies, rotation=20)
    axes[1, 1].set_ylabel("Trainable parameters (M)")
    axes[1, 1].set_title("D  Adaptation cost", loc="left")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for bar, value in zip(bars, trainable):
        if np.isfinite(value):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value / 1e6:.2f}M", ha="center", va="bottom", fontsize=8,
            )

    title = "ViT adaptation experiment"
    if report.get("train_tickers"):
        title += f" — {', '.join(report['train_tickers'])}"
    elif report.get("ticker"):
        title += f" — {report['ticker']}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.5)
    paths.append(save_figure(fig, output_dir, "overview", dpi))

    histories = {name: results[name].get("history", []) for name in strategies}
    if any(histories.values()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for index, strategy in enumerate(strategies):
            history = histories[strategy]
            if not history:
                continue
            epochs = [entry["epoch"] for entry in history]
            axes[0].plot(
                epochs, [entry["train_loss"] for entry in history],
                marker="o", markersize=3, color=colors[index], label=strategy,
            )
            axes[1].plot(
                epochs, [primary_metric(entry["validation"]) for entry in history],
                marker="o", markersize=3, color=colors[index], label=strategy,
            )
            best_epoch = results[strategy].get("best_epoch")
            if best_epoch in epochs:
                best_index = epochs.index(best_epoch)
                axes[1].scatter(
                    [best_epoch], [primary_metric(history[best_index]["validation"])],
                    marker="*", s=90, color=colors[index], edgecolor="black", zorder=4,
                )
        axes[0].set_title("Training loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].set_title("Validation selection metric (★ selected checkpoint)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro F1")
        for ax in axes:
            ax.grid(color="#DDDDDD", linewidth=0.6)
            ax.legend(frameon=False)
        fig.tight_layout()
        paths.append(save_figure(fig, output_dir, "training_curves", dpi))

    available = [
        name for name in strategies
        if "confusion_matrix" in results[name].get("best_validation", {})
    ]
    if available:
        columns = min(3, len(available))
        rows = math.ceil(len(available) / columns)
        fig, axes = plt.subplots(
            rows, columns, figsize=(4.3 * columns, 3.8 * rows), squeeze=False,
        )
        image = None
        for index, strategy in enumerate(available):
            image = draw_confusion(
                axes.flat[index], results[strategy]["best_validation"], strategy
            )
        for ax in axes.flat[len(available):]:
            ax.axis("off")
        fig.suptitle("Best-validation confusion matrices (row-normalized)", fontsize=12)
        fig.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.86, wspace=0.4, hspace=0.5)
        paths.append(save_figure(fig, output_dir, "confusion_matrices", dpi))

    selected_test = report.get("selected_test")
    frozen_test = report.get("frozen_baseline_test")
    if selected_test and frozen_test:
        selected = report.get("selected", "selected")
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
        test_metrics = ("accuracy", "balanced_accuracy", "macro_f1")
        metric_x = np.arange(len(test_metrics))
        width = 0.34
        axes[0].bar(
            metric_x - width / 2,
            [frozen_test[value] for value in test_metrics], width,
            color=COLORS[0], label="frozen",
        )
        axes[0].bar(
            metric_x + width / 2,
            [selected_test[value] for value in test_metrics], width,
            color=COLORS[1], label=selected,
        )
        axes[0].set_xticks(metric_x, ["Accuracy", "Balanced acc.", "Macro F1"])
        axes[0].set_ylim(bottom=0)
        axes[0].set_title("Held-out Test metrics")
        axes[0].legend(frameon=False)
        axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
        draw_confusion(axes[1], frozen_test, "Frozen Test")
        draw_confusion(axes[2], selected_test, f"{selected} Test")
        fig.suptitle("Test was opened only after validation selection", fontsize=12)
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.82, wspace=0.4)
        paths.append(save_figure(fig, output_dir, "test_comparison", dpi))
    return paths


def generate_multiseed_figures(
    report: dict, output_dir: Path, dpi: int
) -> list[Path]:
    strategies = report["strategies"]
    seeds = [str(seed) for seed in report["seeds"]]
    per_seed = report["per_seed"]
    aggregate = report["aggregate"]
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(strategies))
    for seed in seeds:
        values = [
            per_seed[seed][f"{strategy}_mean_ticker_macro_f1"]
            for strategy in strategies
        ]
        axes[0].plot(x, values, color="#999999", alpha=0.6, linewidth=1)
        axes[0].scatter(x, values, color="#777777", s=18)
    for index, strategy in enumerate(strategies):
        stats = aggregate[strategy]["mean_ticker_macro_f1"]
        axes[0].errorbar(
            index, stats["mean"], yerr=stats["std"], color=COLORS[index],
            marker="o", markersize=7, capsize=4, linewidth=1.5,
        )
    axes[0].set_xticks(x, strategies)
    axes[0].set_ylabel("Mean-ticker macro F1")
    axes[0].set_title(f"A  Paired results (n={len(seeds)} seeds)", loc="left")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)

    tickers = list(aggregate[strategies[0]]["by_ticker_macro_f1"])
    ticker_x = np.arange(len(tickers))
    width = 0.8 / len(strategies)
    for index, strategy in enumerate(strategies):
        stats = aggregate[strategy]["by_ticker_macro_f1"]
        offset = (index - (len(strategies) - 1) / 2) * width
        axes[1].bar(
            ticker_x + offset, [stats[ticker]["mean"] for ticker in tickers],
            width, yerr=[stats[ticker]["std"] for ticker in tickers], capsize=3,
            color=COLORS[index], label=strategy,
        )
    axes[1].set_xticks(ticker_x, tickers)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("B  Results by ticker (mean ± SD)", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle("ViT adaptation multi-seed summary — Test locked", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2)
    return [save_figure(fig, output_dir, "overview", dpi)]


def generate_auxiliary_figures(
    report: dict, output_dir: Path, dpi: int
) -> list[Path]:
    candidates = report["candidates"]
    results = report["validation_results"]
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    x = np.arange(len(candidates))
    colors = [COLORS[i % len(COLORS)] for i in range(len(candidates))]
    macro = [results[name]["validation"]["macro_f1"] for name in candidates]
    accuracy = [results[name]["validation"]["accuracy"] for name in candidates]
    width = 0.36
    axes[0, 0].bar(x - width / 2, macro, width, color=colors, label="Macro F1")
    axes[0, 0].bar(
        x + width / 2, accuracy, width, color=colors, alpha=0.45, label="Accuracy"
    )
    axes[0, 0].set_xticks(x, candidates)
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_title("A  Validation candidate comparison", loc="left")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", color="#DDDDDD", linewidth=0.6)

    for index, candidate in enumerate(candidates):
        losses = results[candidate].get("losses", [])
        axes[0, 1].plot(
            range(1, len(losses) + 1), losses, marker="o", markersize=3,
            color=colors[index], label=candidate,
        )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Fusion training loss")
    axes[0, 1].set_title("B  Training curves", loc="left")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(color="#DDDDDD", linewidth=0.6)

    class_x = np.arange(len(LABELS))
    group_width = 0.8 / len(candidates)
    for index, candidate in enumerate(candidates):
        detail = results[candidate]["validation"]["detail"]
        offset = (index - (len(candidates) - 1) / 2) * group_width
        axes[1, 0].bar(
            class_x + offset, [detail[label]["f1-score"] for label in LABELS],
            group_width, color=colors[index], label=candidate,
        )
    axes[1, 0].set_xticks(class_x, [value.title() for value in LABELS])
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].set_title("C  Validation F1 by class", loc="left")
    axes[1, 0].grid(axis="y", color="#DDDDDD", linewidth=0.6)

    selected_test = report.get("selected_test")
    baseline = report.get("volume_baseline_test")
    if selected_test and baseline:
        test_x = np.arange(2)
        test_values = [baseline["macro_f1"], selected_test["macro_f1"]]
        test_bars = axes[1, 1].bar(
            test_x, test_values, color=(COLORS[0], COLORS[2])
        )
        axes[1, 1].set_xticks(test_x, ["volume", report["selected"]])
        axes[1, 1].set_ylim(bottom=0)
        axes[1, 1].set_ylabel("Test macro F1")
        axes[1, 1].set_title("D  Selected vs predefined baseline", loc="left")
        for bar, value in zip(test_bars, test_values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom",
            )
        axes[1, 1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    else:
        axes[1, 1].axis("off")
    fig.suptitle(
        f"Auxiliary vision comparison — {report.get('ticker', '')}", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2, w_pad=1.5)
    return [save_figure(fig, output_dir, "overview", dpi)]


def generate_figures(
    report_path: Path, report: dict, report_type: str,
    output_dir: Path, dpi: int,
) -> list[Path]:
    if report_type == "cross_stock":
        from experiments.vision.cross_stock_report import generate_figures as cross_stock_figures
        return cross_stock_figures(report, output_dir, dpi)
    if report_type == "backbone_comparison":
        return generate_comparison_visualizations(report_path, output_dir, dpi)
    if report_type == "vit_adaptation":
        return generate_adaptation_figures(report, output_dir, dpi)
    if report_type == "multiseed_summary":
        return generate_multiseed_figures(report, output_dir, dpi)
    if report_type == "auxiliary_vision":
        return generate_auxiliary_figures(report, output_dir, dpi)
    raise AssertionError(report_type)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(fmt(value) for value in row) + " |" for row in rows)
    return lines


def append_common_header(
    lines: list[str], report_path: Path, report_type: str, report: dict
) -> None:
    try:
        source = report_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        source = report_path.resolve()
    if "test_was_opened" in report:
        test_status = "OPENED" if report["test_was_opened"] else "LOCKED"
    elif report.get("selected_test"):
        test_status = "OPENED（由 selected_test 欄位判定）"
    else:
        test_status = "UNKNOWN"
    lines.extend([
        "# Vision Experiment Report",
        "",
        "> 此文件由 `render_experiment_report.py` 從 `report.json` 自動產生；"
        "原始 JSON 才是數值的唯一依據。",
        "",
        f"- 報告類型：`{report_type}`",
        f"- 原始資料：`{source}`",
        f"- Test 狀態：{test_status}",
        "",
    ])


def append_figures(lines: list[str], figure_paths: list[Path], markdown_path: Path) -> None:
    if not figure_paths:
        return
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(["## 圖表", ""])
    titles = {
        "overview.png": "實驗總覽",
        "training_curves.png": "訓練過程",
        "confusion_matrices.png": "混淆矩陣",
        "test_comparison.png": "Test 比較",
    }
    for path in figure_paths:
        relative = path.relative_to(markdown_path.parent)
        lines.extend([f"### {titles.get(path.name, path.stem)}", "", f"![{path.stem}]({relative.as_posix()})", ""])


def comparison_markdown(report: dict, lines: list[str]) -> None:
    backbones = report["backbones"]
    split = report.get("split", {})
    lines.extend([
        "## 實驗設定", "",
        f"- 股票：{', '.join(report.get('train_tickers', []))}",
        f"- Seeds：{', '.join(str(value) for value in report.get('seeds', []))}",
        f"- Train／Validation：{split.get('n_train', '—')}／{split.get('n_validation', '—')}",
        f"- Gap：{split.get('gap', '—')} 個交易日",
        f"- 選模指標：`{report.get('selection_metric', '—')}`",
        "", "## Validation 結果", "",
    ])
    rows = []
    for backbone in backbones:
        aggregate = report["aggregate"][backbone]
        primary = aggregate["mean_ticker_macro_f1"]
        pooled = aggregate["pooled_validation"]
        rows.append([
            display_name(backbone), f"{primary['mean']:.4f} ± {primary['std']:.4f}",
            pooled["accuracy"]["mean"], pooled["balanced_accuracy"]["mean"],
            pooled["macro_f1"]["mean"], pooled["log_loss"]["mean"],
        ])
    lines.extend(markdown_table(
        ["Backbone", "Mean-ticker macro F1", "Accuracy", "Balanced accuracy", "Pooled macro F1", "Log loss"],
        rows,
    ))
    comparisons = report.get("comparisons_against_vit", {})
    if not comparisons and report.get("comparison"):
        comparisons = {"maxvit_t": {
            "delta_candidate_minus_vit": report["comparison"]["delta_maxvit_minus_vit"],
            "candidate_win_count": report["comparison"]["maxvit_win_count"],
            "predefined_gate": report["comparison"].get("predefined_gate", {}),
        }}
    if comparisons:
        lines.extend(["", "## 預設門檻判定", ""])
        rows = []
        for candidate, value in comparisons.items():
            delta = value["delta_candidate_minus_vit"]
            gate = value.get("predefined_gate", {})
            gate_result = (
                "PASS" if gate.get("passed") else "FAIL" if gate else "N/A"
            )
            rows.append([
                display_name(candidate), f"{delta['mean']:.4f} ± {delta['std']:.4f}",
                f"{value['candidate_win_count']}/{len(report['seeds'])}",
                gate_result,
            ])
        lines.extend(markdown_table(
            ["Candidate", "Candidate − ViT", "Seed wins", "Gate"], rows
        ))


def adaptation_markdown(report: dict, lines: list[str]) -> None:
    split = report.get("split", {})
    lines.extend([
        "## 實驗設定", "",
        f"- 訓練股票：{', '.join(report.get('train_tickers', [])) or report.get('ticker', '—')}",
        f"- Held-out 股票：{report.get('held_out_ticker') or '無'}",
        f"- Seed：{report.get('seed', '—')}",
        f"- Train／Validation／Test：{split.get('n_train', '—')}／{split.get('n_validation', '—')}／{split.get('n_test', '—')}",
        f"- 選定策略：`{report.get('selected', '—')}`",
        f"- 選模指標：`{report.get('selection_metric', '—')}`",
        "", "## Validation 結果", "",
    ])
    rows = []
    for strategy in report["strategies"]:
        result = report["results"][strategy]
        value = result["best_validation"]
        rows.append([
            strategy + (" ★" if strategy == report.get("selected") else ""),
            result.get("best_epoch"), primary_metric(value), value.get("accuracy"),
            value.get("balanced_accuracy"), value.get("macro_f1"), value.get("log_loss"),
        ])
    lines.extend(markdown_table(
        ["Strategy", "Best epoch", "Selection metric", "Accuracy", "Balanced accuracy", "Macro F1", "Log loss"],
        rows,
    ))
    if report.get("selected_test"):
        selected = report["selected_test"]
        frozen = report.get("frozen_baseline_test")
        lines.extend(["", "## Held-out Test", ""])
        test_rows = []
        if frozen:
            test_rows.append(["frozen", frozen.get("accuracy"), frozen.get("balanced_accuracy"), frozen.get("macro_f1"), frozen.get("log_loss")])
        test_rows.append([report.get("selected", "selected"), selected.get("accuracy"), selected.get("balanced_accuracy"), selected.get("macro_f1"), selected.get("log_loss")])
        lines.extend(markdown_table(
            ["Model", "Accuracy", "Balanced accuracy", "Macro F1", "Log loss"], test_rows
        ))


def multiseed_markdown(report: dict, lines: list[str]) -> None:
    strategies = report["strategies"]
    lines.extend([
        "## 實驗設定", "",
        f"- Seeds：{', '.join(str(value) for value in report['seeds'])}",
        f"- 選模指標：`{report.get('selection_metric', '—')}`",
        "- Test：全程 locked",
        "", "## 多 seed 彙整", "",
    ])
    rows = []
    for strategy in strategies:
        stats = report["aggregate"][strategy]["mean_ticker_macro_f1"]
        rows.append([strategy, f"{stats['mean']:.4f} ± {stats['std']:.4f}", stats["min"], stats["max"]])
    lines.extend(markdown_table(["Strategy", "Mean ± SD", "Min", "Max"], rows))
    lines.extend(["", "## 每個 seed", ""])
    per_seed_rows = []
    for seed in (str(value) for value in report["seeds"]):
        item = report["per_seed"][seed]
        per_seed_rows.append([
            seed, item.get("selected"),
            *[item[f"{strategy}_mean_ticker_macro_f1"] for strategy in strategies],
            item.get("delta_lora_r4_minus_frozen"),
        ])
    lines.extend(markdown_table(
        ["Seed", "Selected", *strategies, "Candidate delta"], per_seed_rows
    ))


def auxiliary_markdown(report: dict, lines: list[str]) -> None:
    split = report.get("split", {})
    lines.extend([
        "## 實驗設定", "",
        f"- 股票：{report.get('ticker', '—')}",
        f"- Seed：{report.get('seed', '—')}",
        f"- Train／Validation／Test：{split.get('n_train', '—')}／{split.get('n_validation', '—')}／{split.get('n_test', '—')}",
        f"- 選定候選：`{report.get('selected', '—')}`",
        "", "## Validation 結果", "",
    ])
    rows = []
    for candidate in report["candidates"]:
        metrics = report["validation_results"][candidate]["validation"]
        rows.append([
            candidate + (" ★" if candidate == report.get("selected") else ""),
            metrics.get("accuracy"), metrics.get("macro_f1"),
            *[metrics["detail"][label]["f1-score"] for label in LABELS],
        ])
    lines.extend(markdown_table(
        ["Candidate", "Accuracy", "Macro F1", "Bearish F1", "Neutral F1", "Bullish F1"], rows
    ))
    if report.get("selected_test"):
        lines.extend(["", "## Test 結果", ""])
        test_rows = []
        if report.get("volume_baseline_test"):
            value = report["volume_baseline_test"]
            test_rows.append(["volume", value["accuracy"], value["macro_f1"]])
        value = report["selected_test"]
        test_rows.append([report["selected"], value["accuracy"], value["macro_f1"]])
        lines.extend(markdown_table(["Candidate", "Accuracy", "Macro F1"], test_rows))


def format_markdown(
    report_path: Path, report: dict, report_type: str,
    figure_paths: list[Path], output_path: Path,
) -> Path:
    lines: list[str] = []
    append_common_header(lines, report_path, report_type, report)
    if report_type == "cross_stock":
        from experiments.vision.cross_stock_report import append_markdown
        append_markdown(report, lines)
    elif report_type == "backbone_comparison":
        comparison_markdown(report, lines)
    elif report_type == "vit_adaptation":
        adaptation_markdown(report, lines)
    elif report_type == "multiseed_summary":
        multiseed_markdown(report, lines)
    elif report_type == "auxiliary_vision":
        auxiliary_markdown(report, lines)
    append_figures(lines, figure_paths, output_path)
    limitations = report.get("limitations") or []
    if limitations:
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(["## 限制", ""])
        lines.extend(f"- {value}" for value in limitations)
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def render_report_bundle(
    report_path: Path,
    output_dir: Path | None = None,
    markdown_path: Path | None = None,
    dpi: int = 220,
) -> dict:
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_type = detect_report_type(report)
    if report_type == "backbone_comparison":
        ensure_aggregate_fields(report)
    output_dir = Path(output_dir) if output_dir else report_path.parent / "figures"
    markdown_path = Path(markdown_path) if markdown_path else report_path.parent / "report.md"
    figures = generate_figures(report_path, report, report_type, output_dir, dpi)
    formatted = format_markdown(report_path, report, report_type, figures, markdown_path)
    return {
        "source": report_path,
        "type": report_type,
        "figures": figures,
        "markdown": formatted,
    }


def render_all(root: Path, dpi: int) -> tuple[list[dict], list[tuple[Path, str]]]:
    completed = []
    failed = []
    for report_path in sorted(root.rglob("report.json")):
        try:
            result = render_report_bundle(report_path, dpi=dpi)
            completed.append(result)
            print(f"[render] {report_path} -> {result['markdown']}", flush=True)
        except Exception as error:  # 批次模式需繼續處理其他歷史報告。
            failed.append((report_path, str(error)))
            print(f"[render] SKIP {report_path}: {error}", flush=True)

    index = root / "experiment_reports_index.md"
    lines = ["# Experiment Reports Index", "", "自動產生的實驗報告索引。", ""]
    for item in completed:
        relative = item["markdown"].relative_to(root)
        lines.append(f"- [{item['source'].parent.relative_to(root)}]({relative.as_posix()}) — `{item['type']}`")
    if failed:
        lines.extend(["", "## Skipped", ""])
        lines.extend(f"- `{path.relative_to(root)}`：{reason}" for path, reason in failed)
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[render] index -> {index}", flush=True)
    return completed, failed


def auto_render_after_training(report_path: Path) -> None:
    """訓練 runner 用的非致命包裝；報告繪製失敗不抹掉訓練成果。"""
    try:
        result = render_report_bundle(report_path)
        print(f"[report] markdown -> {result['markdown']}", flush=True)
        print(f"[report] figures -> {report_path.parent / 'figures'}", flush=True)
    except Exception as error:
        print(
            f"[report] WARNING: JSON 已保存，但自動產圖／Markdown 失敗：{error}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", type=Path)
    parser.add_argument(
        "--all", dest="all_root", type=Path,
        help="遞迴處理目錄內所有 report.json，並建立索引",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if bool(args.report) == bool(args.all_root):
        parser.error("請指定單一 report，或使用 --all 指定實驗根目錄")
    if args.dpi < 100:
        parser.error("--dpi 至少 100")
    if args.all_root:
        completed, failed = render_all(args.all_root, args.dpi)
        print(f"[render] completed={len(completed)}, skipped={len(failed)}")
        return
    result = render_report_bundle(
        args.report, args.output_dir, args.markdown, args.dpi
    )
    for path in result["figures"]:
        print(path)
    print(result["markdown"])


if __name__ == "__main__":
    main()
