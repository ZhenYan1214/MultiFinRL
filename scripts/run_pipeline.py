"""完整 pipeline 入口：A -> B -> C 依序執行（整合階段用）。

三人各自開發時直接跑自己模組的主程式即可，不需要這支。

用法：
    python scripts/run_pipeline.py --fake            # 全 fake：驗證三模組銜接格式
    python scripts/run_pipeline.py --ticker AAPL --limit 50
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n=== {' '.join(cmd)} ===")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--fake", action="store_true", help="B/C 用假資料，驗證銜接格式")
    ap.add_argument("--limit", type=int, default=50, help="真實模式各階段筆數上限")
    args = ap.parse_args()

    if args.fake:
        run("module_b_encoder.generate_vectors", "--fake", "--n", "10")
        run("module_c_fusion.fusion.train", "--fake", "--n", "16", "--epochs", "1")
        print("\n[pipeline] fake 流程通過：B 產出格式 -> C 讀入訓練皆正常")
        return

    # A
    run("module_a_data.crawler.fetch_ohlcv", "--ticker", args.ticker)
    run("module_a_data.preprocess.chart_generator", "--ticker", args.ticker,
        "--limit", str(args.limit))
    run("module_a_data.build_dataset", "--ticker", args.ticker, "--limit", str(args.limit))
    # B
    run("module_b_encoder.generate_vectors", "--ticker", args.ticker,
        "--limit", str(args.limit))
    # C
    run("module_c_fusion.fusion.train", "--ticker", args.ticker)
    run("module_c_fusion.validation.classifier", "--ticker", args.ticker)
    run("module_c_fusion.backtest.backtest", "--ticker", args.ticker,
        "--strategy", "buy_and_hold")
    print("\n[pipeline] A -> B -> C 完成")


if __name__ == "__main__":
    main()
