"""將第一階段 pipeline 產物彙整成可追溯的 ``report.md`` 與圖表。

輸出目錄只使用秒級 run_id，不把 ticker 放進目錄名稱：

    data/outputs/reports/{YYYYMMDD_HHMMSS}/
        report.json
        report.md
        sources/*.json
        figures/*.png

這支程式不重新訓練。它只接受同一次 pipeline 產生、且 temporal split、run_id、
checkpoint SHA256 彼此一致的結果；不一致時停止，避免把不同次執行的數字拼成假報告。

用法：
    python -m module_c_fusion.reporting.pipeline_report \
        --ticker AAPL --run_id 20261004_153000
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

_MPL_CACHE = Path(tempfile.gettempdir()) / "multifinrl-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from shared import paths
from shared.utils import read_json, write_json


LABELS = ("BEARISH", "NEUTRAL", "BULLISH")
RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")
COLORS = {
    "train": "#0072B2",
    "validation": "#E69F00",
    "test": "#009E73",
    "gap": "#A0A0A0",
    "loss_train": "#0072B2",
    "loss_validation": "#D55E00",
    "score": "#009E73",
}


class ReportConsistencyError(RuntimeError):
    """來源檔不是同一次 pipeline 時拒絕產生正式報告。"""


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


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id 必須是 YYYYMMDD_HHMMSS")
    try:
        dt.datetime.strptime(run_id, "%Y%m%d_%H%M%S")
    except ValueError as error:
        raise ValueError("run_id 不是有效日期時間") from error


def _required_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {label}：{path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 必須是 JSON object：{path}")
    return payload


def _same_path(left: str | None, right: Path) -> bool:
    if not left:
        return False
    return Path(left).expanduser().resolve() == right.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(paths.ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sample_vector_index(ticker: str, manifest: dict) -> Path:
    ordered_dates = []
    for split_name in ("train", "validation", "test"):
        ordered_dates.extend(manifest.get("dates", {}).get(split_name, []))
    for date in ordered_dates:
        candidate = paths.vector_dir(ticker, date) / "index.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到 {ticker} 的向量 index.json")


def _collect_sources(ticker: str, run_id: str) -> tuple[dict, dict[str, Path]]:
    source_paths = {
        "temporal_split": paths.temporal_split_path(ticker),
        "fusion_train": paths.OUTPUTS / "metrics" / f"fusion_train_report_{ticker}.json",
        "classification": paths.classification_report_path(ticker),
        "z_fused_meta": paths.OUTPUTS / "z_fused" / f"{ticker}_index.meta.json",
    }
    manifest = _required_json(source_paths["temporal_split"], "temporal split")
    fusion = _required_json(source_paths["fusion_train"], "Fusion 訓練報告")
    classification = _required_json(source_paths["classification"], "分類報告")
    z_meta = _required_json(source_paths["z_fused_meta"], "Z_fused metadata")
    source_paths["vector_index"] = _sample_vector_index(ticker, manifest)
    vector_index = _required_json(source_paths["vector_index"], "向量 index")

    errors = []
    for label, payload in (
        ("temporal split", manifest),
        ("Fusion report", fusion),
        ("classification report", classification),
        ("Z_fused metadata", z_meta),
        ("vector index", vector_index),
    ):
        payload_ticker = payload.get("ticker")
        if payload_ticker != ticker:
            errors.append(f"{label} ticker={payload_ticker!r}，預期 {ticker!r}")

    for label, payload in (
        ("Fusion report", fusion),
        ("classification report", classification),
        ("Z_fused metadata", z_meta),
    ):
        if payload.get("run_id") != run_id:
            errors.append(
                f"{label} run_id={payload.get('run_id')!r}，預期 {run_id!r}"
            )

    protocol = manifest.get("protocol")
    for label, payload in (("Fusion report", fusion), ("classification report", classification)):
        if payload.get("protocol") != protocol:
            errors.append(
                f"{label} protocol={payload.get('protocol')!r}，"
                f"但 temporal split={protocol!r}"
            )

    expected_counts = manifest.get("counts", {})
    fusion_counts = fusion.get("split_counts", {})
    z_counts = z_meta.get("split_counts", {})
    classifier_counts = {
        "train": classification.get("n_train"),
        "validation": classification.get("n_val"),
        "test": classification.get("n_test"),
    }
    for split_name in ("train", "validation", "test"):
        expected = expected_counts.get(split_name)
        for label, actual in (
            ("Fusion report", fusion_counts.get(split_name)),
            ("classification report", classifier_counts.get(split_name)),
            ("Z_fused metadata", z_counts.get(split_name)),
        ):
            if actual != expected:
                errors.append(
                    f"{label} 的 {split_name}={actual}，temporal split={expected}"
                )

    for label, raw_path in (
        ("Fusion report", fusion.get("split_manifest")),
        ("classification report", classification.get("split_manifest")),
        ("Z_fused metadata", z_meta.get("split_manifest")),
    ):
        if not _same_path(raw_path, source_paths["temporal_split"]):
            errors.append(f"{label} 沒有指向同一份 temporal split manifest")

    fusion_checkpoint = fusion.get("checkpoint")
    z_checkpoint = z_meta.get("fusion_checkpoint") or {}
    classifier_checkpoint = classification.get("fusion_checkpoint") or {}
    if not fusion_checkpoint:
        errors.append("Fusion report 缺少 checkpoint")
    else:
        checkpoint_path = Path(fusion_checkpoint)
        if not checkpoint_path.exists():
            errors.append(f"Fusion checkpoint 不存在：{checkpoint_path}")
        else:
            actual_hash = _sha256(checkpoint_path)
            for label, checkpoint in (
                ("Z_fused metadata", z_checkpoint),
                ("classification report", classifier_checkpoint),
            ):
                if not _same_path(checkpoint.get("path"), checkpoint_path):
                    errors.append(f"{label} 指向不同的 Fusion checkpoint")
                if checkpoint.get("sha256") != actual_hash:
                    errors.append(f"{label} 的 Fusion checkpoint SHA256 不一致")

    event = None
    event_path = paths.OUTPUTS / "metrics" / "event_extraction_report.json"
    if event_path.exists():
        candidate = _required_json(event_path, "事件抽取報告")
        if candidate.get("ticker") == ticker:
            source_paths["event_extraction"] = event_path
            event = candidate
            if event.get("run_id") != run_id:
                errors.append(
                    f"event extraction report run_id={event.get('run_id')!r}，"
                    f"預期 {run_id!r}"
                )

    if errors:
        details = "\n- ".join(errors)
        raise ReportConsistencyError(
            "報告來源不是同一次 pipeline，已停止產生 report.md：\n- " + details
        )

    daily_path = paths.daily_json(ticker, manifest["dates"]["train"][0])
    daily = _required_json(daily_path, "每日 dataset 樣本")
    source_paths["dataset_sample"] = daily_path
    payloads = {
        "manifest": manifest,
        "fusion": fusion,
        "classification": classification,
        "z_meta": z_meta,
        "vector_index": vector_index,
        "daily": daily,
        "event": event,
    }
    return payloads, source_paths


def _summary_report(ticker: str, run_id: str, payloads: dict) -> dict:
    manifest = payloads["manifest"]
    vectors = payloads["vector_index"].get("vectors", {})
    event = payloads["event"]
    event_summary = None
    if event is not None:
        event_summary = {
            key: value for key, value in event.items() if key != "events_per_day"
        }
        event_summary["event_days"] = len(event.get("events_per_day", {}))
        event_summary["detection_status"] = "complete"
        evaluation = event.get("evaluation")
        event_summary["evaluation_status"] = (
            "complete" if isinstance(evaluation, dict) else "pending_ground_truth"
        )

    return {
        "schema_version": 1,
        "run_id": run_id,
        "ticker": ticker,
        "generated_at": dt.datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
        "scope": "phase_1_fusion_and_classification",
        "phase_2": "pending_rl_and_backtest",
        "mode": "macro" if "macro" in payloads["daily"] else "equity",
        "inputs": {
            "vision": vectors.get("H_v", {}),
            "text": vectors.get("H_t", {}),
            "retrieval": vectors.get("H_r", {}),
        },
        "temporal_split": {
            "protocol": manifest.get("protocol"),
            "counts": manifest.get("counts", {}),
            "periods": manifest.get("periods", {}),
            "gap_trading_days": manifest.get("gap_trading_days"),
            "label": manifest.get("metadata", {}).get("label_quantile_thresholds", {}),
        },
        "fusion": payloads["fusion"],
        "classification": payloads["classification"],
        "event_extraction": event_summary,
        "z_fused": payloads["z_meta"],
    }


def _save_figure(fig: plt.Figure, directory: Path, name: str, dpi: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_temporal_split(report: dict, directory: Path, dpi: int) -> Path:
    configure_style()
    split = report["temporal_split"]
    order = (
        "train", "gap_train_validation", "validation",
        "gap_validation_test", "test",
    )
    labels = {
        "train": "Train",
        "gap_train_validation": "Gap",
        "validation": "Validation",
        "gap_validation_test": "Gap",
        "test": "Test",
    }
    colors = {
        "train": COLORS["train"],
        "gap_train_validation": COLORS["gap"],
        "validation": COLORS["validation"],
        "gap_validation_test": COLORS["gap"],
        "test": COLORS["test"],
    }
    fig, (timeline_ax, count_ax) = plt.subplots(
        2, 1, figsize=(11, 5.4), gridspec_kw={"height_ratios": [1, 1.3]}
    )
    for name in order:
        period = split["periods"].get(name, [])
        if len(period) != 2:
            continue
        start = mdates.date2num(dt.date.fromisoformat(period[0]))
        end = mdates.date2num(dt.date.fromisoformat(period[1]) + dt.timedelta(days=1))
        timeline_ax.broken_barh([(start, end - start)], (0, 8), facecolors=colors[name])
        timeline_ax.text(
            (start + end) / 2, 4, labels[name], ha="center", va="center",
            color="white" if "gap" not in name else "black", fontsize=8,
        )
    timeline_ax.set_ylim(0, 8)
    timeline_ax.set_yticks([])
    timeline_ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    timeline_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    timeline_ax.set_xlabel("Date")
    timeline_ax.set_title("A  Strict temporal split", loc="left")

    counts = [split["counts"].get(name, 0) for name in order]
    x = np.arange(len(order))
    bars = count_ax.bar(x, counts, color=[colors[name] for name in order])
    count_ax.set_xticks(x, [labels[name] for name in order])
    count_ax.set_ylabel("Trading days")
    count_ax.set_title("B  Samples by split", loc="left")
    count_ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for bar, value in zip(bars, counts):
        count_ax.text(
            bar.get_x() + bar.get_width() / 2, value, str(value),
            ha="center", va="bottom",
        )
    fig.suptitle(f"Temporal protocol — {report['ticker']}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.8)
    return _save_figure(fig, directory, "temporal_split.png", dpi)


def _plot_fusion_history(report: dict, directory: Path, dpi: int) -> Path:
    configure_style()
    fusion = report["fusion"]
    history = fusion.get("history", [])
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        epochs, [row["train_loss"] for row in history], marker="o",
        color=COLORS["loss_train"], label="Train loss",
    )
    axes[0].plot(
        epochs, [row["validation_loss"] for row in history], marker="s",
        color=COLORS["loss_validation"], label="Validation loss",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("A  Fusion loss", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(color="#DDDDDD", linewidth=0.6)

    scores = [row["validation_macro_f1"] for row in history]
    axes[1].plot(epochs, scores, marker="o", color=COLORS["score"])
    best_epoch = fusion.get("best_epoch")
    if best_epoch in epochs:
        index = epochs.index(best_epoch)
        axes[1].scatter(
            [best_epoch], [scores[index]], marker="*", s=120,
            color="#CC79A7", edgecolor="black", linewidth=0.5, zorder=4,
            label=f"Selected epoch {best_epoch}",
        )
        axes[1].legend(frameon=False)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation macro F1")
    axes[1].set_title("B  Checkpoint selection metric", loc="left")
    axes[1].grid(color="#DDDDDD", linewidth=0.6)
    fig.suptitle(f"Fusion training — {report['ticker']}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2)
    return _save_figure(fig, directory, "fusion_training_curves.png", dpi)


def _plot_classifier_selection(report: dict, directory: Path, dpi: int) -> Path | None:
    configure_style()
    classification = report["classification"]
    candidates = classification.get("validation_candidates", [])
    if not candidates:
        return None
    values = [row["C"] for row in candidates]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(
        values, [row["macro_f1"] for row in candidates], marker="o",
        color=COLORS["score"], label="Validation macro F1",
    )
    ax.plot(
        values, [row["accuracy"] for row in candidates], marker="s",
        color=COLORS["validation"], label="Validation accuracy",
    )
    selected = classification.get("selected_C")
    if selected in values:
        index = values.index(selected)
        ax.scatter(
            [selected], [candidates[index]["macro_f1"]], marker="*", s=130,
            color="#CC79A7", edgecolor="black", linewidth=0.5, zorder=4,
            label=f"Selected C={selected:g}",
        )
    ax.set_xscale("log")
    ax.set_xticks(values, [f"{value:g}" for value in values])
    ax.set_xlabel("Logistic Regression C (log scale)")
    ax.set_ylabel("Validation score")
    ax.set_title(f"Validation-only classifier selection — {report['ticker']}")
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, directory, "classifier_selection.png", dpi)


def _plot_classification_metrics(report: dict, directory: Path, dpi: int) -> Path:
    configure_style()
    classification = report["classification"]
    detail = classification["detail"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    overview_names = ["Accuracy", "Macro F1"]
    overview_values = [classification["accuracy"], classification["macro_f1"]]
    bars = axes[0].bar(
        overview_names, overview_values,
        color=[COLORS["validation"], COLORS["score"]],
    )
    axes[0].axhline(1 / 3, color="#777777", linestyle="--", linewidth=1,
                    label="1/3 reference")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("A  Held-out Test overview", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for bar, value in zip(bars, overview_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
            ha="center", va="bottom",
        )

    metric_names = ("precision", "recall", "f1-score")
    class_x = np.arange(len(LABELS))
    width = 0.24
    metric_colors = ("#0072B2", "#E69F00", "#009E73")
    for index, metric in enumerate(metric_names):
        axes[1].bar(
            class_x + (index - 1) * width,
            [detail[label][metric] for label in LABELS],
            width, color=metric_colors[index], label=metric.replace("-score", ""),
        )
    axes[1].set_xticks(class_x, [label.title() for label in LABELS])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Score")
    axes[1].set_title("B  Held-out Test metrics by class", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.suptitle(f"Classification performance — {report['ticker']}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2)
    return _save_figure(fig, directory, "classification_metrics.png", dpi)


def _plot_confusion_matrix(report: dict, directory: Path, dpi: int) -> Path:
    configure_style()
    counts = np.asarray(report["classification"]["confusion_matrix"], dtype=int)
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0,
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="cividis")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            value = normalized[row, column]
            ax.text(
                column, row, f"{counts[row, column]}\n{value:.0%}",
                ha="center", va="center",
                color="white" if value > 0.48 else "black", fontweight="bold",
            )
    ax.set_xticks(range(len(LABELS)), [label.title() for label in LABELS])
    ax.set_yticks(range(len(LABELS)), [label.title() for label in LABELS])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Held-out Test confusion matrix — {report['ticker']}")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized share")
    fig.tight_layout()
    return _save_figure(fig, directory, "confusion_matrix.png", dpi)


def _plot_event_metrics(report: dict, directory: Path, dpi: int) -> Path | None:
    event = report.get("event_extraction")
    evaluation = event.get("evaluation") if event else None
    if not isinstance(evaluation, dict):
        return None
    configure_style()
    names = ("Precision", "Recall", "F1")
    values = [evaluation["precision"], evaluation["recall"], evaluation["f1"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(names, values, color=("#0072B2", "#E69F00", "#009E73"))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Raw-text event extraction — {report['ticker']} "
        f"(ground truth n={evaluation.get('n_days', '—')})"
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
            ha="center", va="bottom",
        )
    fig.tight_layout()
    return _save_figure(fig, directory, "event_extraction_metrics.png", dpi)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return lines


def _figure(lines: list[str], title: str, filename: str) -> None:
    lines.extend([f"### {title}", "", f"![{title}](figures/{filename})", ""])


def _write_markdown(report: dict, output: Path, source_names: dict[str, str]) -> None:
    split = report["temporal_split"]
    fusion = report["fusion"]
    classification = report["classification"]
    detail = classification["detail"]
    inputs = report["inputs"]
    lines = [
        f"# Pipeline Report — {report['ticker']}",
        "",
        "> 本報告由保存的 JSON 結果自動產生；`report.json` 與 `sources/` 是數值追溯依據。",
        "",
        f"- Run ID：`{report['run_id']}`",
        f"- 產生時間：{report['generated_at']}",
        f"- 資料模式：`{report['mode']}`",
        f"- 時間協定：`{split['protocol']}`",
        "- 本次範圍：第一階段（資料、向量、Fusion、分類驗證）",
        "- **[第二階段 - Pending] RL 與 Backtest 未執行，因此沒有 Sharpe、MDD 或累積報酬。**",
        "",
        "## 結果摘要",
        "",
    ]
    lines.extend(_table(
        ["Ticker", "Best Fusion epoch", "Selected C", "Test accuracy", "Test macro F1"],
        [[
            report["ticker"], fusion.get("best_epoch"), classification.get("selected_C"),
            classification.get("accuracy"), classification.get("macro_f1"),
        ]],
    ))
    lines.extend(["", "## 模型輸入", ""])
    lines.extend(_table(
        ["模態", "Encoder", "Shape／設定"],
        [
            ["H_v", inputs.get("vision", {}).get("encoder"),
             f"shape={inputs.get('vision', {}).get('shape')}；inputs={inputs.get('vision', {}).get('input_types')}"],
            ["H_t", inputs.get("text", {}).get("encoder"),
             f"shape={inputs.get('text', {}).get('shape')}"],
            ["H_r", inputs.get("retrieval", {}).get("encoder"),
             f"shape={inputs.get('retrieval', {}).get('shape')}；top_k={inputs.get('retrieval', {}).get('top_k')}"],
        ],
    ))

    lines.extend(["", "## 時間切分與標籤", ""])
    split_rows = []
    for name in (
        "train", "gap_train_validation", "validation", "gap_validation_test", "test",
    ):
        period = split["periods"].get(name, [])
        split_rows.append([
            name, split["counts"].get(name),
            period[0] if len(period) == 2 else "—",
            period[1] if len(period) == 2 else "—",
        ])
    lines.extend(_table(["Split", "天數", "開始", "結束"], split_rows))
    thresholds = split.get("label", {})
    if thresholds:
        lines.extend([
            "",
            f"標籤門檻只由 `{thresholds.get('source_split', 'train')}` 計算："
            f"BEARISH < {_fmt(thresholds.get('bearish_below'))}；"
            f"BULLISH > {_fmt(thresholds.get('bullish_above'))}。",
            "",
        ])
    _figure(lines, "時間切分", "temporal_split.png")

    lines.extend(["## Fusion 訓練", ""])
    history_rows = [[
        row["epoch"], row["train_loss"], row["validation_loss"],
        row["validation_macro_f1"], "★" if row["epoch"] == fusion.get("best_epoch") else "",
    ] for row in fusion.get("history", [])]
    lines.extend(_table(
        ["Epoch", "Train loss", "Validation loss", "Validation macro F1", "採用"],
        history_rows,
    ))
    lines.append("")
    _figure(lines, "Fusion 訓練過程", "fusion_training_curves.png")

    lines.extend(["## 分類器選模與 Test", ""])
    lines.append(
        "Logistic Regression 的 C 只由 Validation 選擇；選定後以 Train＋Validation "
        "重新擬合，Test 只做最後評估。"
    )
    lines.append("")
    lines.extend(_table(
        ["C", "Validation accuracy", "Validation macro F1", "採用"],
        [[
            row["C"], row["accuracy"], row["macro_f1"],
            "★" if row["C"] == classification.get("selected_C") else "",
        ] for row in classification.get("validation_candidates", [])],
    ))
    lines.append("")
    _figure(lines, "分類器 Validation 選模", "classifier_selection.png")

    lines.extend(["### Held-out Test 類別結果", ""])
    lines.extend(_table(
        ["類別", "Precision", "Recall", "F1", "Support"],
        [[
            label, detail[label]["precision"], detail[label]["recall"],
            detail[label]["f1-score"], int(detail[label]["support"]),
        ] for label in LABELS] + [[
            "Macro avg", detail["macro avg"]["precision"], detail["macro avg"]["recall"],
            detail["macro avg"]["f1-score"], int(detail["macro avg"]["support"]),
        ]],
    ))
    lines.append("")
    _figure(lines, "分類指標", "classification_metrics.png")
    _figure(lines, "混淆矩陣（數量＋列百分比）", "confusion_matrix.png")

    event = report.get("event_extraction")
    if event:
        lines.extend(["## 原始文字事件抽取（輔助分析）", ""])
        lines.append("此結果直接分析新聞／文件，不是 Z_fused 的分類成績。")
        lines.append("")
        evaluation = event.get("evaluation")
        if isinstance(evaluation, dict):
            lines.extend(_table(
                ["Method", "資料天數", "Ground truth 天數", "Precision", "Recall", "F1"],
                [[
                    event.get("method"), event.get("days"), evaluation.get("n_days"),
                    evaluation.get("precision"), evaluation.get("recall"), evaluation.get("f1"),
                ]],
            ))
            lines.append("")
            _figure(lines, "事件抽取指標", "event_extraction_metrics.png")
        else:
            lines.extend([
                f"- 事件偵測狀態：已完成（method=`{event.get('method', '—')}`）",
                f"- 已處理資料：{event.get('days', '—')} 天",
                f"- 偵測到至少一個事件：{event.get('event_days', 0)} 天",
                f"- 評分狀態：尚未建立 `{report['ticker']}` ground truth，"
                "因此暫不計算 Precision／Recall／F1。",
                f"- 預期檔案：`data/labels/event_ground_truth/{report['ticker']}.json`",
                "- 影響範圍：只影響事件抽取評分，不影響 Fusion 訓練或漲跌分類結果。",
                "",
            ])
    else:
        lines.extend([
            "## 原始文字事件抽取（輔助分析）", "",
            "本次沒有同 run_id、同 ticker 的事件抽取報告；macro 模式會略過此步驟。", "",
        ])

    lines.extend(["## Provenance", ""])
    lines.extend(_table(
        ["來源", "快照"],
        [[name, f"`sources/{filename}`"] for name, filename in source_names.items()],
    ))
    checkpoint = report["z_fused"].get("fusion_checkpoint") or {}
    lines.extend([
        "",
        f"- Fusion checkpoint：`{checkpoint.get('path', '—')}`",
        f"- Checkpoint SHA256：`{checkpoint.get('sha256', '—')}`",
        "",
        "## 限制", "",
        "- 本報告的 Test 指標只適合最終評估，不應拿來反覆調參。",
        "- Event extraction 是原始文字的輔助分析，不能解讀成 Z_fused 的事件辨識能力。",
        "- [第二階段 - Pending] 尚未執行多資產 RL 與回測。",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def _update_index() -> Path:
    paths.PIPELINE_REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for report_path in sorted(paths.PIPELINE_REPORTS.glob("*/report.json"), reverse=True):
        try:
            report = read_json(report_path)
        except (OSError, ValueError):
            continue
        relative = report_path.parent.relative_to(paths.PIPELINE_REPORTS)
        rows.append([
            report.get("run_id"), report.get("ticker"), report.get("mode"),
            report.get("generated_at"), f"[{relative}/report.md]({relative.as_posix()}/report.md)",
        ])
    lines = ["# Pipeline Reports", "", "第一階段 pipeline 自動報告索引。", ""]
    lines.extend(_table(["Run ID", "Ticker", "Mode", "Generated at", "Report"], rows))
    index = paths.PIPELINE_REPORTS / "index.md"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def render_pipeline_report(ticker: str, run_id: str, dpi: int = 220) -> dict:
    _validate_run_id(run_id)
    payloads, source_paths = _collect_sources(ticker, run_id)
    report = _summary_report(ticker, run_id, payloads)
    output_dir = paths.pipeline_report_dir(run_id)
    if output_dir.exists():
        raise FileExistsError(f"報告目錄已存在，拒絕覆寫：{output_dir}")

    paths.PIPELINE_REPORTS.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}_", dir=paths.PIPELINE_REPORTS))
    try:
        figures_dir = temporary_dir / "figures"
        source_dir = temporary_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_names = {}
        for name, source in source_paths.items():
            filename = {
                "temporal_split": "temporal_split.json",
                "fusion_train": "fusion_train_report.json",
                "classification": "classification_report.json",
                "z_fused_meta": "z_fused_index.meta.json",
                "vector_index": "vector_index.json",
                "dataset_sample": "dataset_sample.json",
                "event_extraction": "event_extraction_report.json",
            }[name]
            shutil.copy2(source, source_dir / filename)
            source_names[name] = filename

        report["provenance"] = {
            "sources": {
                name: {
                    "original": _project_path(source_paths[name]),
                    "snapshot": f"sources/{filename}",
                }
                for name, filename in source_names.items()
            }
        }
        write_json(report, temporary_dir / "report.json")

        figure_paths = [
            _plot_temporal_split(report, figures_dir, dpi),
            _plot_fusion_history(report, figures_dir, dpi),
            _plot_classifier_selection(report, figures_dir, dpi),
            _plot_classification_metrics(report, figures_dir, dpi),
            _plot_confusion_matrix(report, figures_dir, dpi),
            _plot_event_metrics(report, figures_dir, dpi),
        ]
        figure_paths = [path for path in figure_paths if path is not None]
        _write_markdown(report, temporary_dir / "report.md", source_names)
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    index_path = _update_index()
    return {
        "run_id": run_id,
        "ticker": ticker,
        "directory": output_dir,
        "markdown": output_dir / "report.md",
        "json": output_dir / "report.json",
        "figures": [output_dir / "figures" / path.name for path in figure_paths],
        "index": index_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="產生第一階段 pipeline Markdown 與圖表報告")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--run_id", required=True, help="YYYYMMDD_HHMMSS；不包含 ticker")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi < 100:
        parser.error("--dpi 至少 100")
    try:
        result = render_pipeline_report(args.ticker, args.run_id, args.dpi)
    except ReportConsistencyError as error:
        raise SystemExit(str(error)) from error
    print(f"[pipeline_report] markdown -> {result['markdown']}")
    print(f"[pipeline_report] figures -> {result['directory'] / 'figures'}")
    print(f"[pipeline_report] index -> {result['index']}")


if __name__ == "__main__":
    main()
