"""依序執行 ViT Frozen vs LoRA r4 多 seed validation，並彙整 mean ± std。

這支腳本刻意不傳 ``--evaluate-test``，所以不會再次開啟 JPM future test。每個 seed
使用獨立輸出目錄；已完整產生 report 的 seed 會自動跳過，方便中斷後續跑。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "vision" / "experiment_vit_adaptation.py"
OUTPUT_ROOT = (
    ROOT / "data" / "outputs" / "experiments" / "vit_adaptation"
    / "AAPL_NVDA_MSFT_holdout_JPM"
)
STRATEGIES = ("frozen", "lora_r4")


def report_path(seed: int, smoke: bool) -> Path:
    prefix = "multiseed_smoke" if smoke else "multiseed"
    return OUTPUT_ROOT / f"{prefix}_seed{seed}" / "report.json"


def validate_existing_report(path: Path, seed: int) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("seed") != seed:
        raise RuntimeError(f"{path} 的 seed 不是 {seed}")
    if tuple(report.get("strategies", ())) != STRATEGIES:
        raise RuntimeError(f"{path} 的 strategies 不是 {STRATEGIES}")
    if report.get("test_was_opened"):
        raise RuntimeError(f"{path} 不應包含 test 評估")
    return report


def run_seed(args, seed: int) -> dict:
    smoke = args.limit is not None
    path = report_path(seed, smoke)
    if path.exists() and not args.force:
        print(f"[vit_multiseed] seed={seed} 已完成，跳過：{path}", flush=True)
        return validate_existing_report(path, seed)

    run_name = path.parent.name
    command = [
        sys.executable, "-u", str(EXPERIMENT),
        "--train-tickers", "AAPL", "NVDA", "MSFT",
        "--held-out-ticker", "JPM",
        "--strategies", *STRATEGIES,
        "--vit-lr", "1e-4",
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch", str(args.batch),
        "--seed", str(seed),
        "--run-name", run_name,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    print(f"[vit_multiseed] 開始 seed={seed}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not path.exists():
        raise RuntimeError(f"seed={seed} 完成後找不到報告：{path}")
    return validate_existing_report(path, seed)


def metric_value(report: dict, strategy: str, ticker: str | None = None) -> float:
    validation = report["results"][strategy]["best_validation"]
    if ticker is None:
        return float(validation["mean_ticker_macro_f1"])
    return float(validation["by_ticker"][ticker]["macro_f1"])


def summary_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_summary(reports: dict[int, dict]) -> dict:
    tickers = ("AAPL", "NVDA", "MSFT")
    per_seed = {}
    for seed, report in reports.items():
        frozen = metric_value(report, "frozen")
        lora = metric_value(report, "lora_r4")
        per_seed[str(seed)] = {
            "selected": report["selected"],
            "frozen_mean_ticker_macro_f1": frozen,
            "lora_r4_mean_ticker_macro_f1": lora,
            "delta_lora_r4_minus_frozen": lora - frozen,
            "by_ticker": {
                ticker: {
                    strategy: metric_value(report, strategy, ticker)
                    for strategy in STRATEGIES
                }
                for ticker in tickers
            },
        }

    aggregate = {}
    for strategy in STRATEGIES:
        aggregate[strategy] = {
            "mean_ticker_macro_f1": summary_stats([
                metric_value(report, strategy) for report in reports.values()
            ]),
            "by_ticker_macro_f1": {
                ticker: summary_stats([
                    metric_value(report, strategy, ticker)
                    for report in reports.values()
                ])
                for ticker in tickers
            },
        }
    deltas = [
        metric_value(report, "lora_r4") - metric_value(report, "frozen")
        for report in reports.values()
    ]
    aggregate["delta_lora_r4_minus_frozen"] = summary_stats(deltas)
    aggregate["lora_r4_win_count"] = sum(delta > 0 for delta in deltas)

    return {
        "protocol": "validation_only_no_test",
        "seeds": sorted(reports),
        "strategies": list(STRATEGIES),
        "selection_metric": "validation.mean_ticker_macro_f1",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "test_was_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="只供 smoke test 使用")
    parser.add_argument("--force", action="store_true", help="重跑已完成的 seed")
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="不啟動訓練，只彙整所有指定 seed 的既有報告",
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不可重複")

    reports = {}
    for seed in args.seeds:
        path = report_path(seed, args.limit is not None)
        if args.summarize_only:
            if not path.exists():
                raise SystemExit(f"找不到 seed={seed} 報告：{path}")
            reports[seed] = validate_existing_report(path, seed)
        else:
            reports[seed] = run_seed(args, seed)

    summary = build_summary(reports)
    suffix = "multiseed_smoke_summary" if args.limit else "multiseed_summary"
    output = OUTPUT_ROOT / suffix / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    delta = summary["aggregate"]["delta_lora_r4_minus_frozen"]
    wins = summary["aggregate"]["lora_r4_win_count"]
    print(
        f"[vit_multiseed] r4-frozen={delta['mean']:.4f} ± {delta['std']:.4f}, "
        f"wins={wins}/{len(reports)}",
        flush=True,
    )
    print(f"[vit_multiseed] summary -> {output}", flush=True)


if __name__ == "__main__":
    main()
