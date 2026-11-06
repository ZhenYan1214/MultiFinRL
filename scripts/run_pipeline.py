"""第一階段 pipeline 入口：A -> B -> C（融合層 L_align＋L_ground／分類驗證）依序執行。

B 的步驟（docs/architecture/architecture.html）：K 線文字敘述（Claude Haiku，存檔重用）-> 向量（H_v、
H_t = 敘述）-> RAG-1 定義庫 -> α* -> RAG-2 候選。Decoder 聯合訓練（decoder/train.py）不在這支裡，
要另外跑（主線需要 GPU）。

平常各自開發時，直接跑自己模組的主程式即可，不需要這支；
這支是給「要一次跑出一支股票從頭到尾的完整結果」時用的，涵蓋 A 的全部資料來源
（股價、K 線圖、財報、新聞、法說會，或 ETF／指數的總經替代資料）、
B 的向量與事件抽取，以及 C 的 Fusion 訓練與分類驗證。

[第二階段 - Pending] RL 與 Backtest 模組目前保留，但不由本入口自動執行；待多資產
state／action／reward 與回測協定確認後再接回 pipeline。

用法：
    python scripts/run_pipeline.py --fake                     # 全 fake：驗證 B/C 銜接格式
    python scripts/run_pipeline.py --ticker AAPL               # 真跑第一階段完整流程
    python scripts/run_pipeline.py --ticker AAPL --limit 50    # 只處理前 50 天（測試用）
    python scripts/run_pipeline.py --ticker AAPL --skip_transcripts
        # 跳過法說會逐字稿抓取（fetch_transcripts.py 現在改用 Alpha Vantage API，需要
        # ALPHA_VANTAGE_API_KEY，且免費方案有速率限制；還沒設定金鑰或想快速跑通其他部分時
        # 可以跳過；transcript_chunks 留空不影響其他步驟）
    python scripts/run_pipeline.py --ticker SPY --macro
        # ETF／指數：以 point-in-time 總經指標與 FOMC 新聞取代公司文件；
        # ViT 圖片由 config.yaml 的 macro.vision_inputs 決定
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.utils import load_config


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
                    help="跳過法說會逐字稿抓取（沒有 ALPHA_VANTAGE_API_KEY 或想快速跑通其他部分時用）")
    ap.add_argument("--macro", action="store_true",
                    help="ETF／指數總經模式：抓 FRED/FOMC、跳過公司文件；ViT 圖片讀 macro.vision_inputs")
    ap.add_argument("--no_weighted", action="store_true",
                    help="分類驗證預設帶 --weighted（decisions.md #28 定案），加這個旗標跑不加權的對照組")
    args = ap.parse_args()

    if args.fake:
        run("module_b_encoder.generate_vectors", "--fake", "--n", "10")
        run("module_c_fusion.fusion.train", "--fake", "--n", "32", "--epochs", "1")
        print("\n[pipeline] fake 流程通過：B 產出格式 -> C 讀入訓練皆正常")
        return

    limit_args = ["--limit", str(args.limit)] if args.limit else []
    weighted_args = [] if args.no_weighted else ["--weighted"]
    run_id = dt.datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")

    # A：資料工程
    run("module_a_data.crawler.fetch_ohlcv", "--ticker", args.ticker)
    # macro 視覺輸入必須由 macro.vision_inputs 決定，不能因為開啟 --macro 就固定關閉
    # Volume。預設採 K 線＋Volume；若日後把第二張切回 macro_dot_plot，才關閉 Volume。
    macro_vision_inputs = cfg.get("macro", {}).get(
        "vision_inputs", ["candlestick", "volume"]
    )
    chart_args = []
    if args.macro:
        chart_args.append("--volume" if "volume" in macro_vision_inputs else "--no-volume")
    # macro record 還需完成 point-in-time as-of join；此模式先產完整 K 線／Volume，避免
    # --limit N 只畫 OHLCV 最前 N 天，卻因總經 context 不完整而得不到 N 筆有效 record。
    chart_limit_args = [] if args.macro else limit_args
    run("module_a_data.preprocess.chart_generator", "--ticker", args.ticker,
        *chart_args, *chart_limit_args)
    if args.macro:
        # 多抓一年前置期，讓資料起始日可以沿用最近一次 SEP 點陣圖及上一期月頻指標；
        # build_dataset 仍只會輸出 OHLCV 範圍內的交易日。
        macro_start = (
            dt.date.fromisoformat(cfg["date_range"]["start"]) - dt.timedelta(days=366)
        ).isoformat()
        run("module_a_data.crawler.fetch_macro", "--kind", "all",
            "--start", macro_start, "--end", cfg["date_range"]["end"])
    else:
        run("module_a_data.crawler.fetch_filings", "--ticker", args.ticker)
    run("module_a_data.crawler.fetch_news_alpaca", "--ticker", args.ticker)
    if not args.macro and not args.skip_transcripts:
        run("module_a_data.crawler.fetch_transcripts", "--ticker", args.ticker)
    macro_args = ["--macro"] if args.macro else []
    run("module_a_data.build_dataset", "--ticker", args.ticker, *macro_args, *limit_args)
    # K 線文字敘述（Claude Haiku 讀圖，會呼叫 API；已做過的日期會跳過）
    run("module_a_data.chart_caption", "--ticker", args.ticker, "--workers", "8",
        *(["--limit", str(args.limit)] if args.limit else []))

    # B：RAG-1 定義 -> 向量 -> RAG-2（事件抽取與事件驗證頭已移出 pipeline，#130）
    run("module_b_encoder.rag.pattern_kb")
    run("module_b_encoder.generate_vectors", "--ticker", args.ticker, *limit_args)
    # α* 預設用敘述的型態標籤（#131），不需要 alpha_star_rag1 的 cosine 答案卷
    run("module_b_encoder.rag.rag2", "--ticker", args.ticker, *limit_args)

    # C（第一階段）：融合層 L_align＋L_ground -> Z_fused
    run("module_c_fusion.fusion.train", "--ticker", args.ticker, "--run_id", run_id)
    # 型態 probe 只看遮掉 H_t 的融合層（敘述直接寫了型態，沒遮會偷看答案，#125）
    run("module_c_fusion.fusion.train", "--ticker", args.ticker, "--mask_ht", "--tag", "mask_ht",
        "--run_id", run_id)
    run("module_c_fusion.validation.pattern_probe", "--ticker", args.ticker, "--run_id", run_id)
    # 3 類情緒分類器：附屬參考，不當主要結果（#125）
    run("module_c_fusion.validation.classifier", "--ticker", args.ticker,
        "--run_id", run_id, *weighted_args)

    # 將這次 pipeline 的來源 JSON 凍結成一份可追溯的 Markdown＋圖表報告。
    try:
        run("module_c_fusion.reporting.pipeline_report", "--ticker", args.ticker,
            "--run_id", run_id)
    except subprocess.CalledProcessError as error:
        print(f"\n[pipeline] WARNING: 模型產物已完成，但報告產生失敗：{error}")

    # [第二階段 - Pending] 保留原始 RL／Backtest pipeline，現階段先不執行。
    # 待多資產 state／action／reward 與回測協定確認後，再解除下列註解：
    # run("module_c_fusion.rl.train_ppo", "--ticker", args.ticker)
    # for strategy in ("buy_and_hold", "rule_based", "ppo"):
    #     strategy_args = weighted_args if strategy == "rule_based" else []
    #     run("module_c_fusion.backtest.backtest", "--ticker", args.ticker,
    #         "--strategy", strategy, *strategy_args)

    print("\n[pipeline] 第一階段 A -> B -> C（融合層 + 分類驗證）完成")
    print("[pipeline] 下一步：python -m module_c_fusion.decoder.train --ticker", args.ticker,
          "（主線加 RAG-2，需要 GPU；Mac 可先跑 --no_rag2）")
    print("[pipeline] [第二階段 - Pending] RL 與 Backtest 未執行")


if __name__ == "__main__":
    main()
