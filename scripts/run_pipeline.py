"""完整 pipeline 入口：A -> B -> C 依序執行到底（第一階段成果展示用）。

平常各自開發時，直接跑自己模組的主程式即可，不需要這支；
這支是給「要一次跑出一支股票從頭到尾的完整結果」時用的，涵蓋 A 的全部資料來源
（股價、K 線圖、財報、新聞、法說會）、B 的向量與事件抽取、C 的訓練、驗證、回測。

用法：
    python scripts/run_pipeline.py --fake                     # 全 fake：驗證 B/C 銜接格式
    python scripts/run_pipeline.py --ticker AAPL               # 真跑一次完整流程
    python scripts/run_pipeline.py --ticker AAPL --limit 50    # 只處理前 50 天（測試用）
    python scripts/run_pipeline.py --ticker AAPL --skip_transcripts
        # 跳過法說會逐字稿抓取（foolcalls 是不穩定的第三方套件，抓失敗會中斷整條 pipeline，
        # 需要快速跑通其他部分時可以跳過；transcript_chunks 留空不影響其他步驟）
"""
import argparse
import subprocess
import sys
from pathlib import Path

from shared.utils import load_config

ROOT = Path(__file__).resolve().parent.parent


def run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n=== {' '.join(cmd)} ===")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--fake", action="store_true", help="B/C 用假資料，驗證銜接格式")
    ap.add_argument("--limit", type=int, default=None,
                    help="真實模式：K 線圖/dataset/向量各階段筆數上限（測試用，拿掉就是全量）")
    ap.add_argument("--skip_transcripts", action="store_true",
                    help="跳過法說會逐字稿抓取（foolcalls 套件不穩定時用）")
    ap.add_argument("--no_weighted", action="store_true",
                    help="C 訓練/驗證預設帶 --weighted（decisions.md #28 定案），"
                         "加這個旗標可以跑不加權的對照組")
    args = ap.parse_args()

    if args.fake:
        run("module_b_encoder.generate_vectors", "--fake", "--n", "10")
        run("module_c_fusion.fusion.train", "--fake", "--n", "16", "--epochs", "1")
        print("\n[pipeline] fake 流程通過：B 產出格式 -> C 讀入訓練皆正常")
        return

    limit_args = ["--limit", str(args.limit)] if args.limit else []
    weighted_args = [] if args.no_weighted else ["--weighted"]

    # A：資料工程
    run("module_a_data.crawler.fetch_ohlcv", "--ticker", args.ticker)
    run("module_a_data.preprocess.chart_generator", "--ticker", args.ticker, *limit_args)
    run("module_a_data.crawler.fetch_filings", "--ticker", args.ticker)
    run("module_a_data.crawler.fetch_news_alpaca", "--ticker", args.ticker)
    if not args.skip_transcripts:
        run("module_a_data.crawler.fetch_transcripts", "--ticker", args.ticker)
    run("module_a_data.build_dataset", "--ticker", args.ticker, *limit_args)

    # B：向量 + 事件抽取
    run("module_b_encoder.generate_vectors", "--ticker", args.ticker, *limit_args)
    run("module_b_encoder.event_extraction", "--ticker", args.ticker)

    # C：融合訓練 + 分類驗證 + RL + 回測
    run("module_c_fusion.fusion.train", "--ticker", args.ticker, *weighted_args)
    run("module_c_fusion.validation.classifier", "--ticker", args.ticker, *weighted_args)
    run("module_c_fusion.rl.train_ppo", "--ticker", args.ticker)
    for strategy in ("buy_and_hold", "rule_based", "ppo"):
        run("module_c_fusion.backtest.backtest", "--ticker", args.ticker, "--strategy", strategy)

    print("\n[pipeline] A -> B -> C 完成")


if __name__ == "__main__":
    main()
