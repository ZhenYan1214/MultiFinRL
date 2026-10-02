"""跨股票二分類實驗的圖表與 Markdown，供共用 renderer 路由。"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.vision.visualize_comparison_report import COLORS, configure_style, save_figure


def generate_figures(report: dict, output_dir: Path, dpi: int) -> list[Path]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    groups = ["validation_seen", "validation_unseen"]
    if report["test_was_opened"]:
        groups += ["test_seen", "test_unseen"]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x = np.arange(len(groups))
    for index, strategy in enumerate(report["strategies"]):
        available = report["aggregate"].get(strategy, {})
        values = [available.get(group, {}).get("mean_ticker_macro_f1", {}).get("mean", np.nan) for group in groups]
        errors = [available.get(group, {}).get("mean_ticker_macro_f1", {}).get("std", 0) for group in groups]
        ax.bar(x + (index - 0.5) * 0.32, values, 0.32, yerr=errors, capsize=4,
               color=COLORS[index], label=strategy)
    for name, baseline in report.get("baselines", {}).items():
        ax.plot(x, [baseline.get(group, {}).get("mean_ticker_macro_f1", np.nan) for group in groups],
                marker="x", linestyle="--", label=name)
    ax.set_xticks(x, [group.replace("_", " ") for group in groups], rotation=15)
    ax.set_ylabel("Mean-ticker macro F1 (mean ± seed SD)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Cross-stock shared-head ViT — " + ("SMOKE / not efficacy evidence" if report["smoke"] else
                 "Test " + ("OPENED" if report["test_was_opened"] else "LOCKED")))
    paths.append(save_figure(fig, output_dir, "overview", dpi))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    histories = 0
    for seed, results in report["results"].items():
        for index, strategy in enumerate(report["strategies"]):
            if strategy not in results:
                continue
            history = results[strategy]["history"]
            epochs = [entry["epoch"] for entry in history]
            axes[0].plot(epochs, [entry["train_loss"] for entry in history], color=COLORS[index],
                         alpha=0.75, label=f"{strategy} seed{seed}")
            for axis, group in zip(axes[1:], ("validation_seen", "validation_unseen")):
                axis.plot(epochs, [entry["validation"][group]["mean_ticker_macro_f1"] for entry in history],
                          color=COLORS[index], alpha=0.75, label=f"{strategy} seed{seed}")
            histories += 1
    for axis, title in zip(axes, ("Training loss", "Seen validation F1", "Unseen validation F1 (selection)")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    if histories:
        axes[0].legend(frameon=False, fontsize=7)
        paths.append(save_figure(fig, output_dir, "training_curves", dpi))
    else:
        plt.close(fig)

    panels = []
    for seed, results in report["results"].items():
        for strategy, result in results.items():
            for group in groups:
                key = "test" if group.startswith("test") else "best_validation"
                if group in result.get(key, {}):
                    panels.append((f"{strategy} seed{seed}\n{group}", result[key][group]["confusion_matrix"]))
    if panels:
        columns = min(4, len(panels))
        fig, axes = plt.subplots(math.ceil(len(panels) / columns), columns,
                                 figsize=(3.6 * columns, 3.4 * math.ceil(len(panels) / columns)),
                                 squeeze=False, constrained_layout=True)
        for axis, (title, counts) in zip(axes.flat, panels):
            counts = np.asarray(counts)
            totals = counts.sum(axis=1, keepdims=True)
            normalized = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
            axis.imshow(normalized, cmap="cividis", vmin=0, vmax=1)
            for row in range(2):
                for column in range(2):
                    axis.text(column, row, f"{counts[row, column]}\n{normalized[row, column]:.0%}",
                              ha="center", va="center", color="white" if normalized[row, column] < 0.45 else "black")
            axis.set_xticks([0, 1], report["label_names"])
            axis.set_yticks([0, 1], report["label_names"])
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            axis.set_title(title, fontsize=9)
        for axis in axes.flat[len(panels):]:
            axis.axis("off")
        fig.suptitle(("SMOKE — " if report["smoke"] else "") +
                     "Confusion matrices: counts + row-normalized percentages", fontsize=12)
        paths.append(save_figure(fig, output_dir, "confusion_matrices", dpi))

    available = [(seed, strategy, result) for seed, results in report["results"].items()
                 for strategy, result in results.items()]
    if available:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for index, strategy in enumerate(report["strategies"]):
            runs = [result for _seed, name, result in available if name == strategy]
            if not runs:
                continue
            for axis, key in zip(axes, ("by_sector", "by_year")):
                names = sorted({name for run in runs for name in run["best_validation"]["validation_unseen"][key]})
                values = [np.mean([run["best_validation"]["validation_unseen"][key][name]["macro_f1"]
                                   for run in runs if name in run["best_validation"]["validation_unseen"][key]]) for name in names]
                axis.plot(names, values, marker="o", color=COLORS[index], label=strategy)
        for axis, title in zip(axes, ("Unseen validation by sector", "Unseen validation by year")):
            axis.set_title(title)
            axis.set_ylabel("Pooled-group macro F1 (seed mean)")
            axis.tick_params(axis="x", labelrotation=55)
            axis.set_ylim(0, 1.05)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
        paths.append(save_figure(fig, output_dir, "group_breakdown", dpi))
    return paths


def append_markdown(report: dict, lines: list[str]) -> None:
    from experiments.vision.render_experiment_report import markdown_table
    config = report["config"]
    lines.extend(["## 實驗設定", "",
                  f"- 狀態：{'完整' if report.get('complete') else '部分策略已完成'}；{'SMOKE，不能判斷效果' if report['smoke'] else '正式資料'}",
                  "- 架構：共用 ViT、雙圖 CLS 串接、共用二分類頭；未見股票不再訓練新頭。",
                  f"- 標籤：{report['label_rule']}；0=DOWN、1=UP；精確零報酬排除。",
                  "- 報酬基於調整後收盤價，不是原始未調整 close。",
                  f"- 日期（右邊界不包含）：train [{config['train_start']}, {config['validation_start']})；"
                  f"val [{config['validation_start']}, {config['test_start']})；test [{config['test_start']}, {config['test_end']})。",
                  f"- 圖形窗口／預測跨度：{config['window_days']}／{config['horizon_days']} 交易日。",
                  f"- Seeds：{report['seeds']}；主指標：`{report['selection_metric']}`。",
                  f"- 選定策略：{report.get('selected') or '尚未全部完成'}。",
                  f"- 資料 manifest：`{report['dataset_manifest']}`。", ""])
    for name, items in config["groups"].items():
        lines.append(f"- {name} 股票（{len(items)}）：" + ", ".join(item["ticker"] for item in items))
    lines.extend(["", "## 結果：優先看 validation_unseen", "",
                  "seen 是訓練過的股票於後續日期；unseen 是完全未參與梯度更新的股票。SD 為 seed 間波動，不是市場泛化信賴區間。", ""])
    rows = []
    for strategy, aggregate in report["aggregate"].items():
        for group, values in aggregate.items():
            score = values["mean_ticker_macro_f1"]
            rows.append([strategy, group, f"{score['mean']:.4f} ± {score['std']:.4f}",
                         values["accuracy"]["mean"], values["balanced_accuracy"]["mean"]])
    for name, groups in report.get("baselines", {}).items():
        for group, values in groups.items():
            rows.append([name, group, values["mean_ticker_macro_f1"], values["accuracy"], values["balanced_accuracy"]])
    lines.extend(markdown_table(["方法", "資料組", "平均每股 Macro-F1", "Pooled accuracy", "Pooled balanced acc."], rows))
    for group in (["validation_unseen", "test_unseen"] if report["test_was_opened"] else ["validation_unseen"]):
        lines.extend(["", f"## 未見股票明細：{group}", ""])
        rows = []
        for strategy in report["strategies"]:
            key = "test" if group.startswith("test") else "best_validation"
            runs = [results[strategy][key][group] for results in report["results"].values()
                    if group in results.get(strategy, {}).get(key, {})]
            if not runs:
                continue
            for ticker in sorted(runs[0]["by_ticker"]):
                entries = [run["by_ticker"][ticker] for run in runs]
                rows.append([strategy, ticker, entries[0]["n_samples"],
                             float(np.mean([entry["macro_f1"] for entry in entries])),
                             float(np.mean([entry["accuracy"] for entry in entries]))])
        lines.extend(markdown_table(["策略", "股票", "每 seed 樣本數", "Macro-F1 seed mean", "Accuracy seed mean"], rows))
    lines.extend(["", "## LoRA − Frozen（同 seed 配對）", ""])
    lines.extend(markdown_table(["資料組", "差值 mean ± SD", "LoRA 勝出 seeds"], [
        [group, f"{values['mean']:+.4f} ± {values['std']:.4f}", f"{values['wins']}/{values['n_pairs']}"]
        for group, values in report["comparisons"].items()
    ]))
    lines.extend(["", "## 每個 run", ""])
    lines.extend(markdown_table(["Seed", "策略", "Best epoch", "秒數", "可訓練參數", "Checkpoint"], [
        [seed, name, run["best_epoch"], run["seconds"], run["parameter_counts"]["trainable"], f"`{run['checkpoint']}`"]
        for seed, runs in report["results"].items() for name, run in runs.items()
    ]))
    lines.extend(["", "各 seed／strategy 目錄內有 history.json 與每筆 *_predictions.jsonl，可對應回股票、日期與 label_end_date。", ""])
