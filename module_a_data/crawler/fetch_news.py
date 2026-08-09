"""爬取每日財經新聞，存到 data/raw/news/{TICKER}/{YYYY-MM-DD}.json。

必要欄位：headline, content, source, published_at（美東時區）、url。

盤後新聞歸屬規則（docs/decisions.md #23，老師確認為「前一天」）：
    收盤（16:00 美東）後才發布的新聞，視為隔一個交易日才「可得」，
    歸入下一個交易日的資料，而非發布當天——避免當天資料出現實際上
    收盤後才知道的資訊。這是目前對「前一天」這個回覆的解讀，
    若跟老師原意不同，之後再調整 _effective_date() 就好，其他程式不用改。
    註：目前只用「跳過六日」近似下一個交易日，未排除美股假日；
    真正精確需要交易日曆（例如 pandas_market_calendars），先用簡化版。

這支只抓近期新聞（yfinance），歷史新聞回補改用 Alpaca News API
（見 module_a_data/crawler/fetch_news_alpaca.py，decisions.md #27）。
FNSPID（爬蟲 + 現成 dataset）已驗證不可行，程式碼已移除，見 decisions.md #26。
換來源時只要維持本檔輸出格式不變，下游（build_dataset）就不用改。

用法：
    python -m module_a_data.crawler.fetch_news --ticker AAPL
"""
import argparse
import datetime as dt
from zoneinfo import ZoneInfo

import yfinance as yf

from shared import paths
from shared.utils import write_json
from module_a_data.preprocess.text_cleaner import clean

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = dt.time(16, 0)


def _effective_date(published: dt.datetime) -> dt.date:
    """收盤後發布 -> 歸下一個交易日；再跳過週六日（暫不處理美股假日）。"""
    d = published.date()
    if published.timetz().replace(tzinfo=None) >= MARKET_CLOSE:
        d = d + dt.timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d = d + dt.timedelta(days=1)
    return d


def fetch_recent_news(ticker: str) -> list[dict]:
    """用 yfinance 抓近期新聞，轉成統一格式（依美東日期分組前的平面列表）。"""
    items = []
    for n in yf.Ticker(ticker).news or []:
        content = n.get("content", n)  # yfinance 新版包在 content 底下
        ts = content.get("pubDate") or content.get("providerPublishTime")
        if isinstance(ts, (int, float)):
            published = dt.datetime.fromtimestamp(ts, tz=ET)
        elif isinstance(ts, str):
            published = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ET)
        else:
            continue
        items.append({
            "headline": clean(content.get("title", "")),
            "content": clean(content.get("summary", "") or content.get("description", "")),
            "source": (content.get("provider") or {}).get("displayName", "yahoo_finance")
                      if isinstance(content.get("provider"), dict) else "yahoo_finance",
            "published_at": published.isoformat(),
            "url": (content.get("canonicalUrl") or {}).get("url", "")
                   if isinstance(content.get("canonicalUrl"), dict) else "",
        })
    return items


def save_news_by_day(ticker: str, items: list[dict]) -> None:
    """依「有效交易日」分組，一天一個 JSON 檔（同日多則新聞放同一檔）。

    有效交易日 = 收盤後新聞歸下一個交易日之後的日期（見 _effective_date），
    不是直接取 published_at 的日期部分。"""
    by_day: dict[str, list[dict]] = {}
    for it in items:
        published = dt.datetime.fromisoformat(it["published_at"])
        day = _effective_date(published).isoformat()
        by_day.setdefault(day, []).append(it)
    for day, day_items in by_day.items():
        out = paths.RAW_NEWS / ticker / f"{day}.json"
        write_json({"ticker": ticker, "date": day, "news": day_items}, out)
    print(f"[fetch_news] {ticker}: {len(items)} items over {len(by_day)} days -> {paths.RAW_NEWS / ticker}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    args = ap.parse_args()
    save_news_by_day(args.ticker, fetch_recent_news(args.ticker))


if __name__ == "__main__":
    main()
