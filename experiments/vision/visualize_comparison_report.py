"""將已保存的 vision backbone comparison JSON 轉成靜態圖表。

支援 ViT-B/16、MaxViT 與 DINOv3 比較報告。圖表只讀既有 JSON，
不依賴已封存的訓練 runner，不會重新訓練，也不會開啟 Test。

範例：

    python experiments/vision/visualize_comparison_report.py \
        data/outputs/experiments/maxvit_s_comparison/AAPL_NVDA_MSFT/run/report.json
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

_MPL_CACHE = Path(tempfile.gettempdir()) / "multifinrl-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


LABELS = ("BEARISH", "NEUTRAL", "BULLISH")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "^", "D")
DISPLAY_NAMES = {
    "vit_b16": "ViT-B/16",
    "maxvit_t": "MaxViT-T",
    "maxvit_s": "MaxViT-S",
    "dinov3_vitb16": "DINOv3 ViT-B/16",
}


def display_name(backbone: str) -> str:
    return DISPLAY_NAMES.get(backbone, backbone)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def validate_report(report: dict) -> None:
    required = {
        "backbones", "seeds", "results", "aggregate",
        "feature_extraction", "test_was_opened",
    }
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"report 缺少必要欄位：{', '.join(missing)}")
    if report["test_was_opened"]:
        raise ValueError("此工具預期 validation-only report，但 report 顯示 Test 已開啟")
    if len(report["backbones"]) < 2:
        raise ValueError("至少需要兩個 backbone 才能產生比較圖")


def ensure_aggregate_fields(report: dict) -> None:
    """由逐 seed 結果補算舊版 report 尚未保存的 aggregate 欄位。"""
    seeds = [str(seed) for seed in report["seeds"]]

    def stats(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "min": float(array.min()),
            "max": float(array.max()),
        }

    for backbone in report["backbones"]:
        validations = [
            report["results"][backbone][seed]["best_validation"]
            for seed in seeds
        ]
        aggregate = report["aggregate"].setdefault(backbone, {})
        aggregate.setdefault(
            "mean_ticker_macro_f1",
            stats([value["mean_ticker_macro_f1"] for value in validations]),
        )
        aggregate.setdefault("pooled_validation", {
            metric: stats([value[metric] for value in validations])
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss")
        })
        aggregate.setdefault("class_f1", {
            label: stats([
                value["detail"][label]["f1-score"] for value in validations
            ])
            for label in LABELS
        })
        aggregate.setdefault("by_ticker_macro_f1", {
            ticker: stats([
                value["by_ticker"][ticker]["macro_f1"] for value in validations
            ])
            for ticker in report.get("train_tickers", [])
        })


def save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overview(report: dict, output_dir: Path, dpi: int) -> Path:
    backbones = report["backbones"]
    seeds = [str(seed) for seed in report["seeds"]]
    tickers = report.get("train_tickers", [])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    # A: 保留配對關係，直接顯示每個 seed，而不只畫平均柱狀圖。
    ax = axes[0, 0]
    x = np.arange(len(backbones))
    for seed in seeds:
        values = [
            report["results"][backbone][seed]["best_validation"][
                "mean_ticker_macro_f1"
            ]
            for backbone in backbones
        ]
        ax.plot(x, values, color="#999999", alpha=0.55, linewidth=1)
    for index, backbone in enumerate(backbones):
        stats = report["aggregate"][backbone]["mean_ticker_macro_f1"]
        values = [
            report["results"][backbone][seed]["best_validation"][
                "mean_ticker_macro_f1"
            ]
            for seed in seeds
        ]
        ax.scatter(
            np.full(len(values), index), values, s=28,
            color=COLORS[index % len(COLORS)], marker=MARKERS[index % len(MARKERS)],
            edgecolor="white", linewidth=0.5, zorder=3,
        )
        ax.errorbar(
            index, stats["mean"], yerr=stats["std"], color="black",
            marker="_", markersize=14, capsize=4, linewidth=1.3, zorder=4,
        )
    ax.set_xticks(x, [display_name(value) for value in backbones])
    ax.set_ylabel("Mean-ticker macro F1")
    ax.set_title(f"A  Paired validation result (n={len(seeds)} seeds)", loc="left")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    # B: 股票別 macro F1，error bar 是跨 seed 的標準差。
    ax = axes[0, 1]
    width = 0.8 / len(backbones)
    ticker_x = np.arange(len(tickers))
    for index, backbone in enumerate(backbones):
        stats = report["aggregate"][backbone]["by_ticker_macro_f1"]
        means = [stats[ticker]["mean"] for ticker in tickers]
        stds = [stats[ticker]["std"] for ticker in tickers]
        offset = (index - (len(backbones) - 1) / 2) * width
        ax.bar(
            ticker_x + offset, means, width, yerr=stds, capsize=3,
            color=COLORS[index % len(COLORS)], alpha=0.9,
            label=display_name(backbone),
        )
    ax.set_xticks(ticker_x, tickers)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Macro F1")
    ax.set_title("B  Performance by ticker (mean ± SD)", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    # C: 類別別 F1。
    ax = axes[1, 0]
    class_x = np.arange(len(LABELS))
    for index, backbone in enumerate(backbones):
        stats = report["aggregate"][backbone]["class_f1"]
        means = [stats[label]["mean"] for label in LABELS]
        stds = [stats[label]["std"] for label in LABELS]
        offset = (index - (len(backbones) - 1) / 2) * width
        ax.bar(
            class_x + offset, means, width, yerr=stds, capsize=3,
            color=COLORS[index % len(COLORS)], alpha=0.9,
            label=display_name(backbone),
        )
    ax.set_xticks(class_x, [label.title() for label in LABELS])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("F1")
    ax.set_title("C  Performance by class (mean ± SD)", loc="left")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    # D: 工程成本。
    ax = axes[1, 1]
    elapsed = [
        report["feature_extraction"][backbone]["elapsed_seconds"]
        for backbone in backbones
    ]
    bars = ax.bar(
        x, elapsed, color=[COLORS[i % len(COLORS)] for i in range(len(backbones))],
        alpha=0.9,
    )
    ax.set_xticks(x, [display_name(value) for value in backbones])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Feature extraction time (s)")
    ax.set_title("D  Frozen feature extraction cost", loc="left")
    for index, bar in enumerate(bars):
        params = report["feature_extraction"][backbones[index]]["parameter_count"] / 1e6
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{bar.get_height():.1f}s\n{params:.1f}M params",
            ha="center", va="bottom", fontsize=8,
        )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    split = report.get("split", {})
    fig.suptitle(
        "Frozen vision backbone comparison — validation only\n"
        f"{', '.join(tickers)} | {split.get('n_validation', '?')} validation samples | "
        "Test locked",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0, w_pad=1.7)
    return save_figure(fig, output_dir, "overview", dpi)


def plot_training_curves(report: dict, output_dir: Path, dpi: int) -> Path:
    backbones = report["backbones"]
    seeds = [str(seed) for seed in report["seeds"]]
    fig, axes = plt.subplots(
        2, len(backbones), figsize=(5.2 * len(backbones), 6.6), squeeze=False,
        sharex=False,
    )
    seed_colors = plt.get_cmap("cividis")(np.linspace(0.12, 0.9, len(seeds)))

    for column, backbone in enumerate(backbones):
        loss_ax = axes[0, column]
        score_ax = axes[1, column]
        for color, seed in zip(seed_colors, seeds):
            result = report["results"][backbone][seed]
            history = result["history"]
            epochs = [entry["epoch"] for entry in history]
            losses = [entry["train_loss"] for entry in history]
            scores = [
                entry["validation"]["mean_ticker_macro_f1"]
                for entry in history
            ]
            loss_ax.plot(
                epochs, losses, color=color, marker="o", markersize=3,
                linewidth=1.2, label=f"seed {seed}",
            )
            score_ax.plot(
                epochs, scores, color=color, marker="o", markersize=3,
                linewidth=1.2, label=f"seed {seed}",
            )
            best_epoch = result["best_epoch"]
            best_index = epochs.index(best_epoch)
            score_ax.scatter(
                [best_epoch], [scores[best_index]], marker="*", s=70,
                color=color, edgecolor="black", linewidth=0.4, zorder=4,
            )

        loss_ax.set_title(f"{display_name(backbone)} — training loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Balanced cross-entropy")
        loss_ax.grid(color="#DDDDDD", linewidth=0.6)
        score_ax.set_title(f"{display_name(backbone)} — validation selection metric")
        score_ax.set_xlabel("Epoch")
        score_ax.set_ylabel("Mean-ticker macro F1")
        score_ax.grid(color="#DDDDDD", linewidth=0.6)
        score_ax.legend(frameon=False, ncol=2)

    fig.suptitle(
        "Classification-head training trajectories\n"
        "Stars mark the checkpoint selected within each seed",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0, w_pad=1.5)
    return save_figure(fig, output_dir, "training_curves", dpi)


def mean_normalized_confusion(report: dict, backbone: str) -> np.ndarray:
    matrices = []
    for seed in (str(value) for value in report["seeds"]):
        matrix = np.asarray(
            report["results"][backbone][seed]["best_validation"][
                "confusion_matrix"
            ],
            dtype=float,
        )
        row_totals = matrix.sum(axis=1, keepdims=True)
        matrices.append(
            np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)
        )
    return np.mean(matrices, axis=0)


def plot_confusion_matrices(report: dict, output_dir: Path, dpi: int) -> Path:
    backbones = report["backbones"]
    fig, axes = plt.subplots(
        1, len(backbones), figsize=(4.6 * len(backbones), 4.2), squeeze=False,
    )
    image = None
    for index, backbone in enumerate(backbones):
        ax = axes[0, index]
        matrix = mean_normalized_confusion(report, backbone)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="cividis")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                ax.text(
                    column, row, f"{value:.0%}", ha="center", va="center",
                    color="white" if value > 0.48 else "black", fontweight="bold",
                )
        ax.set_xticks(range(len(LABELS)), [label.title() for label in LABELS], rotation=25)
        ax.set_yticks(range(len(LABELS)), [label.title() for label in LABELS])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(display_name(backbone))

    fig.suptitle(
        f"Validation confusion matrices averaged across {len(report['seeds'])} seeds\n"
        "Each row sums to 100%; this is not additional independent sample size",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.80, wspace=0.35)
    return save_figure(fig, output_dir, "confusion_matrices", dpi)


def generate_visualizations(
    report_path: Path,
    output_dir: Path | None = None,
    dpi: int = 220,
) -> list[Path]:
    """產生 backbone comparison 圖表，供 CLI 與自動報告流程共用。"""
    report_path = Path(report_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"找不到 report：{report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report)
    ensure_aggregate_fields(report)
    output_dir = Path(output_dir) if output_dir else report_path.parent / "figures"
    configure_style()
    return [
        plot_overview(report, output_dir, dpi),
        plot_training_curves(report, output_dir, dpi),
        plot_confusion_matrices(report, output_dir, dpi),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="vision comparison report.json")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="預設輸出到 report 同目錄的 figures/",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi < 100:
        parser.error("--dpi 至少 100")
    if not args.report.is_file():
        parser.error(f"找不到 report：{args.report}")

    output_dir = args.output_dir or args.report.parent / "figures"
    report = json.loads(args.report.read_text(encoding="utf-8"))
    paths = generate_visualizations(args.report, output_dir, args.dpi)
    summary = {
        "source_report": str(args.report),
        "backbones": report["backbones"],
        "seeds": report["seeds"],
        "test_was_opened": report["test_was_opened"],
        "confusion_matrix_aggregation": (
            "row-normalize each seed's best-validation confusion matrix, then mean"
        ),
        "figures": [path.name for path in paths],
    }
    summary_path = output_dir / "visualization_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in paths:
        print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
