"""彙整 OHLCV / K 線圖 / 新聞 / 文件 Chunk / 標籤，輸出每日 JSON。

輸出：data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json（一天一筆）
寫出前必須通過 shared.schemas.validate_daily_record()。

前置條件（先跑過）：
    python -m module_a_data.crawler.fetch_ohlcv
    python -m module_a_data.preprocess.chart_generator --ticker AAPL
    python -m module_a_data.crawler.fetch_news --ticker AAPL       # 可先跳過，新聞為空陣列
    python -m module_a_data.crawler.fetch_filings --ticker AAPL    # 可先跳過，chunk 為空陣列

用法：
    python -m module_a_data.build_dataset --ticker AAPL --limit 100   # 先交付 50-100 筆樣本
    python -m module_a_data.build_dataset --ticker AAPL               # 全量
"""
import argparse
import datetime as dt

from shared import paths, schemas
from shared.utils import load_config, read_json, write_json
from module_a_data.crawler.fetch_ohlcv import load_ohlcv
from module_a_data.labeling import build_price_and_label
from module_a_data.preprocess.chunker import make_chunks
from module_a_data.preprocess.text_cleaner import clean

NEWS_BACKFILL_DAYS = 7  # 當日無新聞時，最多往前找幾天


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
    """「最新一份沿用到下一份發布為止」：取 filing_date <= date 的最新一份，切 Chunk。"""
    index_path = paths.RAW_FILINGS / ticker / "index.json"
    if not index_path.exists():
        return []
    valid = [f for f in read_json(index_path)["filings"] if f["filing_date"] <= date]
    if not valid:
        return []
    latest = max(valid, key=lambda f: f["filing_date"])
    html = (paths.RAW_FILINGS / ticker / latest["file"]).read_text(encoding="utf-8", errors="ignore")
    return make_chunks(ticker, latest["form"], latest["filing_date"], clean(html),
                       max_tokens, date_key="filing_date")


def collect_transcript_chunks(ticker: str, date: str, max_tokens: int) -> list[dict]:
    """同上，取 event_date <= date 的最新一份法說會逐字稿。"""
    index_path = paths.RAW_TRANSCRIPTS / ticker / "index.json"
    if not index_path.exists():
        return []
    valid = [t for t in read_json(index_path)["transcripts"] if t["event_date"] <= date]
    if not valid:
        return []
    latest = max(valid, key=lambda t: t["event_date"])
    text = (paths.RAW_TRANSCRIPTS / ticker / latest["file"]).read_text(encoding="utf-8", errors="ignore")
    return make_chunks(ticker, "earnings_call", latest["event_date"], text,
                       max_tokens, date_key="event_date")


def build_daily_record(ticker: str, date: str, df, cfg) -> dict | None:
    """組一筆每日 JSON；K 線圖不存在或標籤產不出來（資料頭尾）回傳 None。"""
    chart_path = paths.RAW_CHARTS / ticker / f"{date}.png"
    if not chart_path.exists():
        return None
    pl = build_price_and_label(
        df, date,
        horizon=cfg["label"]["horizon_trading_days"],
        bullish_threshold=cfg["label"]["bullish_threshold"],
        bearish_threshold=cfg["label"]["bearish_threshold"],
    )
    if pl is None:
        return None
    max_tokens = cfg["text"]["max_chunk_tokens"]
    return {
        "ticker": ticker,
        "date": date,
        "chart": {
            "path": str(chart_path.relative_to(paths.ROOT)).replace("\\", "/"),
            "window_days": cfg["chart"]["window_days"],
            "size": cfg["chart"]["image_size"],
            "channels": cfg["chart"]["channels"],
        },
        "news": collect_news(ticker, date),
        "filing_chunks": collect_filing_chunks(ticker, date, max_tokens),
        "transcript_chunks": collect_transcript_chunks(ticker, date, max_tokens),
        **pl,  # prices + label
    }


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--limit", type=int, default=None, help="只產前 N 筆（樣本交付用）")
    args = ap.parse_args()

    df = load_ohlcv(args.ticker)
    n = 0
    for d in df.index:
        date = d.strftime("%Y-%m-%d")
        record = build_daily_record(args.ticker, date, df, cfg)
        if record is None:
            continue
        schemas.validate_daily_record(record)  # 不合法會 raise，直接中斷
        write_json(record, paths.daily_json(args.ticker, date))
        n += 1
        if args.limit and n >= args.limit:
            break
    print(f"[build_dataset] {args.ticker}: {n} records -> {paths.DATASET / args.ticker}")


if __name__ == "__main__":
    main()
