"""爬取每日財經新聞，存到 data/raw/news/{TICKER}/{YYYY-MM-DD}.json。

必要欄位：headline, content, source, published_at（美東時區）、url。
盤後新聞的歸屬規則見 docs/data_format.md 第 4 節（美東時間切日）。

基礎版先用 yfinance 的 news API（只有近期新聞，歷史新聞有限）。
歷史新聞需另找來源（負責人未指定，自行尋找），候選：
  - GDELT（免費、歷史完整，但需自行過濾相關性）
  - Finnhub / NewsAPI（有免費額度上限）
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
    """依美東日期分組，一天一個 JSON 檔（同日多則新聞放同一檔）。"""
    by_day: dict[str, list[dict]] = {}
    for it in items:
        day = it["published_at"][:10]
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
