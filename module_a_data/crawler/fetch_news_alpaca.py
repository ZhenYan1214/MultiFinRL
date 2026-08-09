"""用 Alpaca News API 抓歷史新聞（例如 2021 年至今），取代 FNSPID 那條走不通的路。

docs/decisions.md #26：FNSPID 爬蟲與現成 dataset 都驗證不可行（自動化偵測擋爬蟲、
dataset 全文欄位是空的），改用官方 API 這個方向。

需要的環境變數：
    ALPACA_API_KEY=你的 API key
    ALPACA_API_SECRET=你的 API secret
    （注意：這裡只需要 API key/secret，不需要帳號密碼或 recovery code，
    那兩個不要放進任何檔案、程式碼或環境變數裡，跟這支腳本無關。）

用法：
    python -m module_a_data.crawler.fetch_news_alpaca --ticker AAPL \\
        --start 2021-01-01 --end 2026-08-08

輸出格式、盤後新聞歸屬規則跟 fetch_news.py 完全一樣（同一支 _effective_date），
會跟既有的每日新聞 JSON 依 url 合併去重，不會覆蓋掉其他來源已經抓到的資料。
"""
import argparse
import datetime as dt
import os
import time
from zoneinfo import ZoneInfo

import requests

from shared import paths
from shared.utils import write_json, read_json
from module_a_data.preprocess.text_cleaner import clean
from module_a_data.crawler.fetch_news import _effective_date

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 沒裝 python-dotenv 也沒關係，退回用系統環境變數

ET = ZoneInfo("America/New_York")
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
PAGE_LIMIT = 50            # Alpaca News API 單頁上限
REQUEST_DELAY_SEC = 0.35   # 免費方案 200 requests/分鐘，留點餘裕不要卡到上限


def _headers() -> dict:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "缺少 ALPACA_API_KEY / ALPACA_API_SECRET 環境變數，"
            "請在專案根目錄建立 .env 檔案設定（不要寫進程式碼、不要進版控）。"
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _parse_created_at(raw: str) -> dt.datetime:
    """Alpaca 回傳的時間戳是 UTC ISO8601（例如 2021-12-31T17:26:14Z），轉成美東時間。"""
    return dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(ET)


def fetch_alpaca_news(ticker: str, start: str, end: str) -> list[dict]:
    """分頁抓完整個日期區間，回傳統一格式的新聞列表（未分日）。

    每抓完一頁會印進度（第幾頁、累積則數、耗時），方便直接看真實抓取速度，
    不用光靠估計。"""
    items = []
    page_token = None
    page_count = 0
    t0 = time.time()
    while True:
        params = {
            "symbols": ticker,
            "start": start,
            "end": end,
            "limit": PAGE_LIMIT,
            "sort": "asc",
            "include_content": "true",
            "exclude_contentless": "true",
        }
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(NEWS_URL, headers=_headers(), params=params, timeout=30)
        if resp.status_code == 429:
            print("[fetch_news_alpaca] 撞到速率限制，等 5 秒重試")
            time.sleep(5)
            continue
        resp.raise_for_status()
        data = resp.json()

        for a in data.get("news", []):
            published = _parse_created_at(a["created_at"])
            items.append({
                "headline": clean(a.get("headline", "")),
                "content": clean(a.get("content") or a.get("summary") or ""),
                "source": f"alpaca_{a.get('source', 'benzinga')}",
                "published_at": published.isoformat(),
                "url": a.get("url", ""),
            })

        page_count += 1
        elapsed = time.time() - t0
        print(f"[fetch_news_alpaca] 第 {page_count} 頁，累積 {len(items)} 則，耗時 {elapsed:.1f} 秒")

        page_token = data.get("next_page_token")
        if not page_token:
            break
        time.sleep(REQUEST_DELAY_SEC)

    return items


def save_news_by_day(ticker: str, items: list[dict]) -> None:
    """依有效交易日分組存檔，跟既有檔案合併去重（依 url 去重），
    避免覆蓋掉其他來源（例如 fetch_news.py 抓的近期新聞）已經寫進去的資料。"""
    by_day: dict[str, list[dict]] = {}
    for it in items:
        published = dt.datetime.fromisoformat(it["published_at"])
        day = _effective_date(published).isoformat()
        by_day.setdefault(day, []).append(it)

    for day, day_items in by_day.items():
        out = paths.RAW_NEWS / ticker / f"{day}.json"
        existing = []
        if out.exists():
            existing = read_json(out)["news"]
        seen_urls = {it.get("url") for it in existing if it.get("url")}
        merged = existing + [it for it in day_items if it.get("url") not in seen_urls]
        write_json({"ticker": ticker, "date": day, "news": merged}, out)

    print(f"[fetch_news_alpaca] {ticker}: {len(items)} 則 -> {len(by_day)} 天 -> {paths.RAW_NEWS / ticker}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    args = ap.parse_args()

    t0 = time.time()
    items = fetch_alpaca_news(args.ticker, args.start, args.end)
    save_news_by_day(args.ticker, items)
    print(f"[fetch_news_alpaca] 全部完成，總耗時 {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
