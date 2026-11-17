"""彙整 OHLCV / 雙視覺輸入 / 新聞 / 文件或總經 context / 標籤，輸出每日 JSON。

輸出：data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json（一天一筆）
寫出前必須通過 shared.schemas.validate_daily_record()。

前置條件（先跑過）：
    python -m module_a_data.crawler.fetch_ohlcv
    python -m module_a_data.preprocess.chart_generator --ticker AAPL
    python -m module_a_data.crawler.fetch_news --ticker AAPL       # 可先跳過，新聞為空陣列
    python -m module_a_data.crawler.fetch_filings --ticker AAPL    # 可先跳過，chunk 為空陣列
    python -m module_a_data.crawler.fetch_macro --kind all         # ETF／指數 --macro 模式

用法：
    python -m module_a_data.build_dataset --ticker AAPL --limit 100   # 先交付 50-100 筆樣本
    python -m module_a_data.build_dataset --ticker AAPL               # 全量
    python -m module_a_data.build_dataset --ticker SPY --macro        # 總經模式，圖片讀 macro.vision_inputs
"""
import argparse
import datetime as dt
from bisect import bisect_left
from functools import lru_cache
from zoneinfo import ZoneInfo

from shared import paths, schemas
from shared.temporal_split import build_temporal_manifest, save_temporal_manifest
from shared.utils import load_config, read_json, write_json
from module_a_data.crawler.fetch_ohlcv import load_ohlcv
from module_a_data.labeling import build_price_and_label, compute_quantile_thresholds
from module_a_data.macro_context import MacroDataStore
from module_a_data.preprocess.chunker import make_chunks
from module_a_data.preprocess.text_cleaner import clean

NEWS_BACKFILL_DAYS = 7  # 當日無新聞時，最多往前找幾天
ET = ZoneInfo("America/New_York")
MARKET_CLOSE = dt.time(16, 0)


# ---------------------------------------------------------------------------
# 文件的「可得交易日」（available_date）
#
# 分析日 t 以收盤為準（標籤 = 第 5 個交易日收盤 / t 收盤），所以 t 收盤後才公開的文件只能從
# 下一個交易日開始用，跟新聞的盤後規則一致（fetch_news.py、decisions.md #23）。
# 財報 8-K 和法說會都在美東 16:30 之後才公開（AAPL 全部是 post-market），舊版用
# filing_date／event_date <= t 判斷，財報日當天就看得到盤後財報，是 look-ahead。
# 下一個交易日取 OHLCV 的交易日序列，會跳過假日。
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _trading_days(ticker: str) -> tuple[str, ...]:
    return tuple(ts.strftime("%Y-%m-%d") for ts in load_ohlcv(ticker).index)


def available_trading_day(ticker: str, date: str, after_close: bool) -> str:
    """在美東日期 date 公開的文件，第一個能用它的分析日。

    盤前／盤中公開且 date 是交易日 → date；盤後公開或 date 不是交易日 → 下一個交易日。
    超出 OHLCV 範圍時回傳 date 的隔天（反正沒有分析日會用到）。
    """
    days = _trading_days(ticker)
    i = bisect_left(days, date)
    if after_close and i < len(days) and days[i] == date:
        i += 1
    if i < len(days):
        return days[i]
    return (dt.date.fromisoformat(date) + dt.timedelta(days=1)).isoformat()


def filing_available_date(ticker: str, filing: dict) -> str:
    """EDGAR 收件時間（accepted_at，UTC）換成美東時間判斷；舊 index 沒有收件時間時保守當成盤後。"""
    accepted = filing.get("accepted_at")
    if not accepted:
        return available_trading_day(ticker, filing["filing_date"], after_close=True)
    t = dt.datetime.fromisoformat(accepted.replace("Z", "+00:00")).astimezone(ET)
    return available_trading_day(ticker, t.date().isoformat(), t.time() >= MARKET_CLOSE)


def transcript_available_date(ticker: str, transcript: dict) -> str:
    """法說會依 report_time：只有明確 pre-market 才算當天可得，其他（post-market、未知）都延到下一個交易日。"""
    return available_trading_day(ticker, transcript["event_date"],
                                 after_close=transcript.get("report_time") != "pre-market")


def _with_available(chunks: list[dict], available_date: str) -> list[dict]:
    return [{**c, "available_date": available_date} for c in chunks]


def collect_news(ticker: str, date: str) -> list[dict]:
    """當日新聞；沒有就往前回補並標 days_ago（docs/decisions.md #6）。"""
    d0 = dt.date.fromisoformat(date)
    for days_ago in range(NEWS_BACKFILL_DAYS + 1):
        day = (d0 - dt.timedelta(days=days_ago)).isoformat()
        f = paths.RAW_NEWS / ticker / f"{day}.json"
        if f.exists():
            items = read_json(f)["news"]
            return [
                {"headline": n["headline"], "content": n["content"], "source": n["source"],
                 "published_at": n["published_at"], "days_ago": days_ago}
                for n in items
            ]
    return []  # 回補範圍內都沒新聞：空陣列（格式允許）


def collect_filing_chunks(ticker: str, date: str, max_tokens: int) -> list[dict]:
    """10-K/10-Q 當背景、8-K 當補充事件，兩者並存，不互相覆蓋（decisions.md #41/#44/#45）。

    一律用 available_date（可得交易日，見上方說明）判斷 t 能不能用，不用 filing_date。
    10-K/10-Q：取 available_date <= t 的最新一份，沿用到下一份可得為止。
    8-K：只列「比目前背景這份 10-K/10-Q 還新」的部分，每則標上「N 天前」前綴（N 從
    available_date 算，比照新聞 days_ago，見 #21），讓模型自己從文字學相關性強度衰減。
    例外：今天剛變成可得的 8-K 一定列入。財報 8-K（t 盤後）和 10-Q（t 晚上收件）常常同一個
    交易日變成可得，只用「比背景新」的話財報 8-K 會在可得的那天就被新背景擋掉、永遠看不到；
    之後幾天就跟原本一樣，由 10-Q 當背景、不再重複列 8-K。
    chunk 保留原本的 filing_date，另外加 available_date。
    """
    index_path = paths.RAW_FILINGS / ticker / "index.json"
    if not index_path.exists():
        return []
    filings = [{**f, "available_date": filing_available_date(ticker, f)}
               for f in read_json(index_path)["filings"]]

    periodic = [f for f in filings if f["form"] in ("10-K", "10-Q") and f["available_date"] <= date]
    chunks: list[dict] = []
    background_since = None
    if periodic:
        latest = max(periodic, key=lambda f: (f["available_date"], f["filing_date"]))
        background_since = latest["available_date"]
        html = (paths.RAW_FILINGS / ticker / latest["file"]).read_text(encoding="utf-8", errors="ignore")
        chunks += _with_available(
            make_chunks(ticker, latest["form"], latest["filing_date"], clean(html),
                        max_tokens, date_key="filing_date"),
            latest["available_date"])

    eight_k = [f for f in filings if f["form"] == "8-K" and f["available_date"] <= date
              and (background_since is None or f["available_date"] > background_since
                   or f["available_date"] == date)]
    d0 = dt.date.fromisoformat(date)
    for f in sorted(eight_k, key=lambda f: (f["available_date"], f["filing_date"])):
        days_ago = (d0 - dt.date.fromisoformat(f["available_date"])).days
        html = (paths.RAW_FILINGS / ticker / f["file"]).read_text(encoding="utf-8", errors="ignore")
        text = f"[{days_ago} 天前的重大訊息揭露 8-K] {clean(html)}"
        chunks += _with_available(
            make_chunks(ticker, f["form"], f["filing_date"], text, max_tokens, date_key="filing_date"),
            f["available_date"])
    return chunks


def collect_transcript_chunks(ticker: str, date: str, max_tokens: int) -> list[dict]:
    """取 available_date <= date 的最新一場法說會逐字稿（盤後開的法說會從下一個交易日才可得）。"""
    index_path = paths.RAW_TRANSCRIPTS / ticker / "index.json"
    if not index_path.exists():
        return []
    transcripts = [{**t, "available_date": transcript_available_date(ticker, t)}
                   for t in read_json(index_path)["transcripts"]]
    valid = [t for t in transcripts if t["available_date"] <= date]
    if not valid:
        return []
    latest = max(valid, key=lambda t: t["event_date"])
    text = (paths.RAW_TRANSCRIPTS / ticker / latest["file"]).read_text(encoding="utf-8", errors="ignore")
    return _with_available(
        make_chunks(ticker, "earnings_call", latest["event_date"], text,
                    max_tokens, date_key="event_date"),
        latest["available_date"])


def _vision_input_types(cfg: dict, macro_store: MacroDataStore | None) -> list[str]:
    if macro_store is not None:
        return cfg.get("macro", {}).get("vision_inputs", ["candlestick", "volume"])
    return cfg["chart"].get("vision_inputs", ["candlestick", "volume"])


def _chart_input_path(
    ticker: str,
    date: str,
    input_type: str,
    macro_store: MacroDataStore | None = None,
):
    """把 config 的視覺輸入名稱解析成圖檔路徑；technical/<set> 可供未來擴充。"""
    if input_type == "candlestick":
        return paths.RAW_CHARTS / ticker / f"{date}.png"
    if input_type == "volume":
        return paths.RAW_CHARTS / ticker / "volume" / f"{date}.png"
    if input_type.startswith("technical/"):
        indicator_set = input_type.removeprefix("technical/")
        return paths.RAW_CHARTS / ticker / "technical" / indicator_set / f"{date}.png"
    if input_type == "macro_dot_plot":
        if macro_store is None:
            raise ValueError("macro_dot_plot 只能在 --macro 模式使用")
        dot_plot = macro_store.dot_plot_as_of(date)
        return macro_store.root / dot_plot["path"] if dot_plot else None
    raise ValueError(f"不支援的 chart.vision_inputs 類型: {input_type}")


def build_daily_record(ticker: str, date: str, df, cfg,
                       bullish_threshold: float, bearish_threshold: float,
                       macro_store: MacroDataStore | None = None) -> dict | None:
    """組一筆每日 JSON；K 線圖不存在或標籤產不出來（資料頭尾）回傳 None。

    bullish_threshold/bearish_threshold：只用 train 日期算一次（見 main()），
    再固定套用到 train/validation/test，不是每天或每個區間重算。
    """
    input_types = _vision_input_types(cfg, macro_store)
    if len(input_types) != 2:
        raise ValueError(f"雙圖 ViT 目前要求 chart.vision_inputs 恰好 2 項，實際為 {input_types}")
    chart_paths = [
        _chart_input_path(ticker, date, input_type, macro_store) for input_type in input_types
    ]
    if any(path is None or not path.exists() for path in chart_paths):
        return None
    macro = macro_store.build_daily_context(date) if macro_store else None
    if macro_store and macro is None:
        return None
    pl = build_price_and_label(
        df, date,
        horizon=cfg["label"]["horizon_trading_days"],
        bullish_threshold=bullish_threshold,
        bearish_threshold=bearish_threshold,
    )
    if pl is None:
        return None
    max_tokens = cfg["text"]["max_chunk_tokens"]
    record = {
        "ticker": ticker,
        "date": date,
        "chart": {
            # path 保留為第一張圖的向下相容欄位；新流程一律使用 inputs。
            "path": str(chart_paths[0].relative_to(paths.ROOT)).replace("\\", "/"),
            "inputs": [
                {
                    "type": input_type,
                    "path": str(path.relative_to(paths.ROOT)).replace("\\", "/"),
                }
                for input_type, path in zip(input_types, chart_paths)
            ],
            "window_days": cfg["chart"]["window_days"],
            "size": cfg["chart"]["image_size"],
            "channels": cfg["chart"]["channels"],
        },
        "news": collect_news(ticker, date),
        # ETF／指數沒有公司財報與法說會；--macro 模式用下方 macro context 取代。
        "filing_chunks": [] if macro_store else collect_filing_chunks(ticker, date, max_tokens),
        "transcript_chunks": [] if macro_store else collect_transcript_chunks(ticker, date, max_tokens),
        **pl,  # prices + label
    }
    if macro is not None:
        record["macro"] = macro
    return record


def eligible_dates(
    ticker: str,
    df,
    cfg,
    macro_store: MacroDataStore | None = None,
) -> list[str]:
    """列出能產生完整雙圖與未來報酬標籤的日期，不讀取未來文字內容。"""
    horizon = cfg["label"]["horizon_trading_days"]
    input_types = _vision_input_types(cfg, macro_store)
    dates = []
    for index, timestamp in enumerate(df.index):
        if index + horizon >= len(df):
            continue
        date = timestamp.strftime("%Y-%m-%d")
        chart_paths = [
            _chart_input_path(ticker, date, input_type, macro_store) for input_type in input_types
        ]
        if all(path is not None and path.exists() for path in chart_paths):
            dates.append(date)
    return dates


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--limit", type=int, default=None, help="只產前 N 筆（樣本交付用）")
    ap.add_argument(
        "--macro", action="store_true",
        help="ETF／指數總經模式：圖片讀 macro.vision_inputs，加入 point-in-time 指標與新聞",
    )
    args = ap.parse_args()

    df = load_ohlcv(args.ticker)
    macro_cfg = cfg.get("macro", {})
    macro_store = MacroDataStore(
        news_backfill_days=macro_cfg.get("news_backfill_days", 7),
    ) if args.macro else None
    horizon = cfg["label"]["horizon_trading_days"]
    dates = eligible_dates(args.ticker, df, cfg, macro_store)
    if args.limit:
        dates = dates[:args.limit]
    split_cfg = cfg.get("temporal_split", {})
    manifest = build_temporal_manifest(
        args.ticker,
        dates,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        validation_ratio=split_cfg.get("validation_ratio", 0.15),
        gap_trading_days=split_cfg.get("gap_trading_days", horizon),
        metadata={
            "label_horizon_trading_days": horizon,
            "macro_mode": args.macro,
            "vision_input_types": _vision_input_types(cfg, macro_store),
        },
    )
    bearish_threshold, bullish_threshold = compute_quantile_thresholds(
        df, horizon,
        low_q=cfg["label"].get("quantile_low", 1 / 3),
        high_q=cfg["label"].get("quantile_high", 2 / 3),
        dates=manifest["dates"]["train"],
    )
    manifest["metadata"]["label_quantile_thresholds"] = {
        "source_split": "train",
        "low_quantile": cfg["label"].get("quantile_low", 1 / 3),
        "high_quantile": cfg["label"].get("quantile_high", 2 / 3),
        "bearish_below": bearish_threshold,
        "bullish_above": bullish_threshold,
    }
    save_temporal_manifest(manifest)
    print(f"[build_dataset] 分位數門檻（只依 {args.ticker} Train 報酬分布算出）："
         f"bearish<{bearish_threshold:.4f}, bullish>{bullish_threshold:.4f}")
    print(f"[build_dataset] split train/val/test="
          f"{manifest['counts']['train']}/{manifest['counts']['validation']}/"
          f"{manifest['counts']['test']}，gap={manifest['gap_trading_days']} "
          f"-> {paths.temporal_split_path(args.ticker)}")

    n = 0
    for date in dates:
        record = build_daily_record(args.ticker, date, df, cfg,
                                    bullish_threshold, bearish_threshold, macro_store)
        if record is None:
            continue
        schemas.validate_daily_record(record)  # 不合法會 raise，直接中斷
        write_json(record, paths.daily_json(args.ticker, date))
        n += 1
    mode = "macro" if args.macro else "equity"
    print(f"[build_dataset] {args.ticker}: {n} records ({mode}) -> {paths.DATASET / args.ticker}")


if __name__ == "__main__":
    main()
