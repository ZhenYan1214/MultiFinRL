"""抓取 Alpaca 歷史新聞；只查尚未覆蓋的日期區間，可從上一頁續跑。

docs/decisions.md #26：FNSPID 爬蟲與現成 dataset 都驗證不可行（自動化偵測擋爬蟲、
dataset 全文欄位是空的），改用官方 API 這個方向。

需要的環境變數：
    ALPACA_API_KEY=你的 API key
    ALPACA_API_SECRET=你的 API secret

用法：
    python -m module_a_data.crawler.fetch_news_alpaca --ticker AAPL \
        --start 2021-01-01 --end 2026-08-08

不指定日期時，研究範圍為 2021-01-01 至今天；已完成的區間會跳過。
下載進度存於 data/raw/news/{TICKER}/meta.json。
"""

import argparse
import datetime as dt
import os
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests

from shared import paths
from shared.utils import read_json, write_json
from module_a_data.crawler.fetch_news import _effective_date
from module_a_data.preprocess.text_cleaner import clean

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 沒裝 python-dotenv 時使用系統環境變數


ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
PAGE_LIMIT = 50            # Alpaca News API 單頁上限
REQUEST_DELAY_SEC = 0.35   # 免費方案 200 requests/分鐘，留點餘裕不要卡到上限
DEFAULT_START = "2021-01-01"
_STATE_VERSION = 1
_QUERY_VERSION = 1  # 查詢條件有語意變更時遞增，避免誤用舊 coverage
_TRACKING_QUERY_KEYS = {"fbclid", "gclid"}


def _headers() -> dict:
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "缺少 ALPACA_API_KEY / ALPACA_API_SECRET 環境變數，"
            "請在專案根目錄建立 .env 檔案設定（不要寫進程式碼、不要進版控）。"
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _request_news_page(ticker: str, start: dt.date, end: dt.date,
                       page_token: str | None = None) -> dict:
    """抓取一頁原始新聞；429 時維持原有的 5 秒重試行為。"""
    start_at = dt.datetime.combine(start, dt.time.min, tzinfo=UTC)
    end_at = dt.datetime.combine(end, dt.time.max, tzinfo=UTC)
    params = {
        "symbols": ticker,
        "start": start_at.isoformat().replace("+00:00", "Z"),
        "end": end_at.isoformat().replace("+00:00", "Z"),
        "limit": PAGE_LIMIT,
        "sort": "asc",
        "include_content": "true",
        "exclude_contentless": "true",
    }
    if page_token:
        params["page_token"] = page_token

    while True:
        resp = requests.get(NEWS_URL, headers=_headers(), params=params, timeout=30)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        print("[fetch_news_alpaca] 撞到速率限制，等 5 秒重試")
        time.sleep(5)


def _normalize_article(article: dict) -> dict:
    """將 Alpaca 原始欄位轉成 raw/news 使用的格式。"""
    published = dt.datetime.fromisoformat(
        article["created_at"].replace("Z", "+00:00")
    ).astimezone(ET)
    source = article.get("source") or "benzinga"
    return {
        "article_id": article.get("id"),
        "headline": clean(article.get("headline", "")),
        "content": clean(article.get("content") or article.get("summary") or ""),
        "source": f"alpaca_{source}",
        "published_at": published.isoformat(),
        "updated_at": article.get("updated_at") or article["created_at"],
        "url": article.get("url", ""),
        "symbols": article.get("symbols") or [],
    }


def _canonical_url(raw: str) -> str:
    """忽略追蹤參數，讓舊資料的 URL 可當 article_id fallback。"""
    if not raw:
        return ""
    try:
        parts = urlsplit(raw.strip())
        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
            and key.lower() not in _TRACKING_QUERY_KEYS
        ]
        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/") or "/",
            urlencode(sorted(query)),
            "",
        ))
    except ValueError:
        return raw.strip()


def _is_older_revision(existing: dict, incoming: dict) -> bool:
    old_raw = existing.get("updated_at")
    new_raw = incoming.get("updated_at")
    if not old_raw or not new_raw:
        return False
    try:
        old = dt.datetime.fromisoformat(old_raw.replace("Z", "+00:00"))
        new = dt.datetime.fromisoformat(new_raw.replace("Z", "+00:00"))
        return new < old
    except ValueError:
        return False


def save_news_by_day(ticker: str, items: list[dict]) -> dict[str, int]:
    """依有效交易日 upsert；article_id 優先，URL 為舊資料 fallback。"""
    by_day: dict[str, list[dict]] = {}
    for item in items:
        published = dt.datetime.fromisoformat(item["published_at"])
        day = _effective_date(published).isoformat()
        by_day.setdefault(day, []).append(item)

    added = updated = unchanged = 0
    for day, day_items in by_day.items():
        out = paths.RAW_NEWS / ticker / f"{day}.json"
        existing = read_json(out).get("news", []) if out.exists() else []
        by_id = {
            str(item["article_id"]): index
            for index, item in enumerate(existing)
            if item.get("article_id") is not None
        }
        by_url = {
            canonical: index
            for index, item in enumerate(existing)
            if (canonical := _canonical_url(item.get("url", "")))
        }
        changed = False

        for item in day_items:
            article_id = item.get("article_id")
            canonical = _canonical_url(item.get("url", ""))
            index = by_id.get(str(article_id)) if article_id is not None else None
            if index is None and canonical in by_url:
                candidate_index = by_url[canonical]
                old_id = existing[candidate_index].get("article_id")
                if old_id is None or article_id is None or str(old_id) == str(article_id):
                    index = candidate_index

            if index is None:
                existing.append(item)
                index = len(existing) - 1
                added += 1
                changed = True
            else:
                old_item = existing[index]
                candidate = old_item if _is_older_revision(old_item, item) else {
                    **old_item,
                    **item,
                }
                if candidate == old_item:
                    unchanged += 1
                else:
                    existing[index] = candidate
                    updated += 1
                    changed = True

            if article_id is not None:
                by_id[str(article_id)] = index
            if canonical:
                by_url[canonical] = index

        if changed:
            write_json({"ticker": ticker, "date": day, "news": existing}, out)

    print(
        f"[fetch_news_alpaca] {ticker}: 本頁/批次 {len(items)} 則，"
        f"新增 {added}、更新 {updated}、未變 {unchanged}"
    )
    return {"added": added, "updated": updated, "unchanged": unchanged}


def _merge_ranges(ranges: list[tuple[dt.date, dt.date]]) -> list[tuple[dt.date, dt.date]]:
    merged: list[list[dt.date]] = []
    one_day = dt.timedelta(days=1)
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + one_day:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _missing_ranges(start: dt.date, end: dt.date,
                    completed: list[tuple[dt.date, dt.date]]) -> list[tuple[dt.date, dt.date]]:
    """從含首尾的要求區間扣除已完成區間。"""
    missing = []
    cursor = start
    one_day = dt.timedelta(days=1)
    for covered_start, covered_end in _merge_ranges(completed):
        if covered_end < cursor:
            continue
        if covered_start > end:
            break
        if covered_start > cursor:
            missing.append((cursor, min(end, covered_start - one_day)))
        cursor = max(cursor, covered_end + one_day)
        if cursor > end:
            break
    if cursor <= end:
        missing.append((cursor, end))
    return missing


class FetchCheckpoint:
    """集中處理 coverage 與 page token，讓抓取主流程只描述步驟。"""

    def __init__(self, ticker: str, state: dict):
        self.ticker = ticker
        self.state = state

    @property
    def path(self) -> Path:
        return paths.RAW_NEWS / self.ticker / "meta.json"

    @classmethod
    def load(cls, ticker: str) -> "FetchCheckpoint":
        path = paths.RAW_NEWS / ticker / "meta.json"
        if path.exists():
            state = read_json(path)
            if (state.get("version") != _STATE_VERSION
                    or state.get("query_version") != _QUERY_VERSION
                    or state.get("ticker") != ticker):
                raise RuntimeError(
                    f"{path} 的版本、ticker 或 query_version 與目前程式不相容；"
                    "請先人工確認 coverage，不要直接刪除狀態後重抓。"
                )
        else:
            state = {
                "version": _STATE_VERSION,
                "query_version": _QUERY_VERSION,
                "ticker": ticker,
                "completed_ranges": [],
                "in_progress": None,
                "last_successful_sync": None,
            }
        return cls(ticker, state)

    def _save(self) -> None:
        """先寫同目錄暫存檔再 replace，避免留下半份 checkpoint。"""
        tmp = self.path.with_suffix(".tmp")
        write_json(self.state, tmp)
        tmp.replace(self.path)

    def _completed_ranges(self) -> list[tuple[dt.date, dt.date]]:
        return [
            (dt.date.fromisoformat(item["start"]), dt.date.fromisoformat(item["end"]))
            for item in self.state.get("completed_ranges", [])
        ]

    def missing_ranges(self, start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
        return _missing_ranges(start, end, self._completed_ranges())

    def begin_range(self, start: dt.date, end: dt.date) -> str | None:
        """範圍與上次未完成的範圍一致才沿用 page token。"""
        pending = self.state.get("in_progress")
        token = None
        if pending and pending["start"] == start.isoformat() and pending["end"] == end.isoformat():
            token = pending.get("page_token")
            print(
                f"[fetch_news_alpaca] 繼續未完成區間 {start}~{end}，"
                f"page_token={'有' if token else '從頭'}"
            )
        else:
            print(f"[fetch_news_alpaca] 開始缺口 {start}~{end}")
        self.state["in_progress"] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "page_token": token,
            "updated_at": dt.datetime.now(UTC).isoformat(),
        }
        self._save()
        return token

    def advance_page(self, page_token: str | None) -> None:
        self.state["in_progress"]["page_token"] = page_token
        self.state["in_progress"]["updated_at"] = dt.datetime.now(UTC).isoformat()
        self._save()

    def finish_range(self, start: dt.date, end: dt.date) -> None:
        """今天尚未結束，不把今天或未來日期標成完整 coverage。"""
        yesterday_et = dt.datetime.now(ET).date() - dt.timedelta(days=1)
        completed_end = min(end, yesterday_et)
        if start <= completed_end:
            ranges = _merge_ranges(self._completed_ranges() + [(start, completed_end)])
            self.state["completed_ranges"] = [
                {"start": a.isoformat(), "end": b.isoformat()}
                for a, b in ranges
            ]
        self.state["in_progress"] = None
        self.state["last_successful_sync"] = dt.datetime.now(UTC).isoformat()
        self._save()


def _parse_range(start: str, end: str) -> tuple[dt.date, dt.date]:
    try:
        start_date = dt.date.fromisoformat(start)
        end_date = dt.date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("start/end 必須是 YYYY-MM-DD") from exc
    if start_date > end_date:
        raise ValueError(f"start 不可晚於 end：{start} > {end}")
    return start_date, end_date


def _fetch_range(ticker: str, start: dt.date, end: dt.date,
                 checkpoint: FetchCheckpoint) -> tuple[int, int]:
    """逐頁抓取單一缺口；每頁存妥後才推進 page token。"""
    token = checkpoint.begin_range(start, end)
    pages = items_count = 0
    while True:
        response = _request_news_page(ticker, start, end, token)
        items = [_normalize_article(article) for article in response.get("news", [])]
        save_news_by_day(ticker, items)

        pages += 1
        items_count += len(items)
        token = response.get("next_page_token")
        checkpoint.advance_page(token)
        print(f"[fetch_news_alpaca] 缺口 {start}~{end}：第 {pages} 頁，累積 {items_count} 則")
        if not token:
            break
        time.sleep(REQUEST_DELAY_SEC)

    checkpoint.finish_range(start, end)
    return pages, items_count


def fetch_alpaca_news(ticker: str, start: str, end: str) -> dict[str, int]:
    """抓取要求的日期範圍；已完成區間不打 API，中斷時從上一頁繼續。"""
    start_date, end_date = _parse_range(start, end)
    checkpoint = FetchCheckpoint.load(ticker)
    gaps = checkpoint.missing_ranges(start_date, end_date)
    if not gaps:
        print(f"[fetch_news_alpaca] {ticker} {start}~{end} 已完整覆蓋，不呼叫 API")
        return {"requested_ranges": 0, "pages": 0, "items": 0}

    t0 = time.time()
    total_pages = total_items = 0
    for gap_start, gap_end in gaps:
        pages, items = _fetch_range(ticker, gap_start, gap_end, checkpoint)
        total_pages += pages
        total_items += items

    print(
        f"[fetch_news_alpaca] 全部完成：{len(gaps)} 個缺口、{total_pages} 頁、"
        f"{total_items} 則，總耗時 {time.time() - t0:.1f} 秒"
    )
    return {"requested_ranges": len(gaps), "pages": total_pages, "items": total_items}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default=DEFAULT_START,
                    help=f"研究範圍起日 YYYY-MM-DD（預設 {DEFAULT_START}）")
    ap.add_argument("--end", default=dt.datetime.now(ET).date().isoformat(),
                    help="研究範圍迄日 YYYY-MM-DD（預設今天）")
    args = ap.parse_args()
    fetch_alpaca_news(args.ticker, args.start, args.end)


if __name__ == "__main__":
    main()
