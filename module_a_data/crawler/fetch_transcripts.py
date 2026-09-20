"""使用 Alpha Vantage API 下載法說會逐字稿，存到 data/raw/transcripts/{TICKER}/。

取代原本用 foolcalls（fool.com 爬蟲）的版本——foolcalls 不是正式上架的套件、原始碼本身
有壞掉的 import、且從未成功端到端跑出過真實資料（見 docs/decisions.md #25，`--start`/
`--max_pages` 那個版本）。改用 Alpha Vantage 官方 API，穩定、有官方文件保障。

API 端點：
  - EARNINGS_CALL_TRANSCRIPT: 抓取指定公司與季度的法說會逐字稿及發言人內容
  - EARNINGS（輔助）: 取得季度財報發布日（reportedDate），作為精準的 event_date

環境變數：
  ALPHA_VANTAGE_API_KEY=你的 API key（或 ALPHAVANTAGE_API_KEY）
  可在專案根目錄的 .env 檔案中設定，或透過命令列參數 --api_key 傳入。

輸出格式（與 fetch_filings 對齊，供 build_dataset 讀取）：
  data/raw/transcripts/{TICKER}/index.json              # 清單（event_date / quarter / 檔名）
  data/raw/transcripts/{TICKER}/EC_{YYYY-MM-DD}.txt     # 純文字逐字稿

用法：
  python -m module_a_data.crawler.fetch_transcripts --ticker AAPL
  python -m module_a_data.crawler.fetch_transcripts --ticker AAPL --start 2021-01-01 --end 2026-08-09

免費方案有速率限制（約 5 次/分鐘），--delay 預設 1.0 秒對免費方案太快，
遇到大量重試訊息時改成 --delay 12.0。
"""
import argparse
import datetime as dt
import os
import time

import requests

from shared import paths
from shared.utils import load_config, read_json, write_json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://www.alphavantage.co/query"


def get_api_key(cli_key: str | None = None) -> str:
    """取得 Alpha Vantage API Key（優先順序：CLI 參數 > ALPHA_VANTAGE_API_KEY > ALPHAVANTAGE_API_KEY）。"""
    key = cli_key or os.environ.get("ALPHA_VANTAGE_API_KEY") or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "缺少 Alpha Vantage API Key！請在專案根目錄的 .env 檔案中設定 "
            "ALPHA_VANTAGE_API_KEY=your_key，或透過 --api_key 傳入。"
        )
    return key


def generate_quarters(start: str, end: str) -> list[str]:
    """根據起訖日期（YYYY-MM-DD），產生所有涵蓋的季度清單（例如 ['2021Q1', '2021Q2', ...]）。"""
    d_start = dt.date.fromisoformat(start)
    d_end = dt.date.fromisoformat(end)
    start_q = (d_start.month - 1) // 3 + 1
    end_q = (d_end.month - 1) // 3 + 1
    quarters = []
    y, q = d_start.year, start_q
    while (y < d_end.year) or (y == d_end.year and q <= end_q):
        quarters.append(f"{y}Q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return quarters


def fetch_earnings_calendar(ticker: str, api_key: str) -> dict[str, str]:
    """呼叫 EARNINGS 端點取得歷史財報公布日（reportedDate），建立 quarter -> event_date 的對照表。"""
    params = {
        "function": "EARNINGS",
        "symbol": ticker,
        "apikey": api_key,
    }
    quarter_to_date: dict[str, str] = {}
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return quarter_to_date
        data = resp.json()
        quarterly = data.get("quarterlyEarnings", [])
        for item in quarterly:
            fiscal_end = item.get("fiscalDateEnding")
            reported_date = item.get("reportedDate")
            if fiscal_end and reported_date:
                f_dt = dt.date.fromisoformat(fiscal_end)
                q_num = (f_dt.month - 1) // 3 + 1
                q_key = f"{f_dt.year}Q{q_num}"
                quarter_to_date[q_key] = reported_date
    except Exception as e:
        print(f"[fetch_transcripts] 查詢 EARNINGS 日曆時發生錯誤（將使用預估日期）: {e}")
    return quarter_to_date


def default_quarter_event_date(quarter: str) -> str:
    """若無精確公布日，回傳預設的估算公布日（通常為該季結束次月下旬）。"""
    y = int(quarter[:4])
    q = int(quarter[-1])
    month_day_map = {
        1: (4, 25),   # Q1 財報約 4 月底公布
        2: (7, 25),   # Q2 財報約 7 月底公布
        3: (10, 25),  # Q3 財報約 10 月底公布
        4: (1, 25),   # Q4 財報約次年 1 月底公布
    }
    if q == 4:
        return f"{y + 1}-01-25"
    m, d = month_day_map[q]
    return f"{y}-{m:02d}-{d:02d}"


def transcript_to_text(data: dict) -> str:
    """將 Alpha Vantage 的逐字稿 JSON 轉換成純文字格式。"""
    symbol = data.get("symbol", "")
    quarter = data.get("quarter", "")
    parts = [f"{symbol} {quarter} Earnings Call Transcript\n"]

    raw_transcript = data.get("transcript")
    if isinstance(raw_transcript, list):
        for item in raw_transcript:
            if isinstance(item, dict):
                speaker = item.get("speaker") or item.get("name") or "Speaker"
                title = item.get("title") or item.get("role") or ""
                speaker_tag = f"{speaker} ({title})" if title else speaker
                content = item.get("content") or item.get("text") or item.get("statement") or ""
                parts.append(f"{speaker_tag}: {content}")
            else:
                parts.append(str(item))
    elif isinstance(raw_transcript, str):
        parts.append(raw_transcript)

    return "\n\n".join(p for p in parts if p)


def fetch_transcript_by_quarter(
    ticker: str,
    quarter: str,
    api_key: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict | None:
    """呼叫 EARNINGS_CALL_TRANSCRIPT 取得單季逐字稿。"""
    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": ticker,
        "quarter": quarter,
        "apikey": api_key,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 429:
                print(f"[fetch_transcripts] 遭遇速率限制 (429)，等待 {retry_delay} 秒後重試...")
                time.sleep(retry_delay)
                continue
            resp.raise_for_status()
            data = resp.json()

            # 檢查 Alpha Vantage 特殊提示訊息（如額度限制或頻率過高）
            if "Information" in data or "Note" in data:
                info_msg = data.get("Information") or data.get("Note")
                print(f"[fetch_transcripts] API 提示訊息: {info_msg}，等待 {retry_delay} 秒重試...")
                time.sleep(retry_delay)
                continue
            if "Error Message" in data:
                print(f"[fetch_transcripts] {ticker} {quarter} 無法取得逐字稿: {data['Error Message']}")
                return None
            if not data.get("transcript"):
                return None
            return data
        except requests.RequestException as e:
            print(f"[fetch_transcripts] {ticker} {quarter} 連線錯誤（第 {attempt + 1} 次）: {e}")
            time.sleep(retry_delay)
    return None


def _next_weekday(date: str) -> str:
    day = dt.date.fromisoformat(date) + dt.timedelta(days=1)
    while day.weekday() >= 5:
        day += dt.timedelta(days=1)
    return day.isoformat()


def save_transcript(ticker: str, event_date: str, quarter: str, text: str) -> None:
    """統一儲存格式：產出 EC_{event_date}.txt 並更新 index.json。"""
    out_dir = paths.RAW_TRANSCRIPTS / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"EC_{event_date}.txt"
    (out_dir / fname).write_text(text, encoding="utf-8")

    index_path = out_dir / "index.json"
    index = {"ticker": ticker, "transcripts": []}
    if index_path.exists():
        index = read_json(index_path)
    # 去重並依 event_date 排序
    entries = [t for t in index.get("transcripts", []) if t["event_date"] != event_date]
    # Alpha Vantage 只提供公布日期，沒有可靠的逐字稿可用時間；保守地從下一個平日使用。
    entries.append({"event_date": event_date, "available_date": _next_weekday(event_date),
                    "quarter": quarter, "file": fname})
    index["transcripts"] = sorted(entries, key=lambda t: t["event_date"])
    write_json(index, index_path)
    print(f"[fetch_transcripts] {ticker} {quarter} ({event_date}) -> {out_dir / fname}")


def fetch_all_transcripts(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    delay_sec: float = 1.0,
) -> int:
    """按季度依序下載指定時間範圍內的所有逐字稿（支援斷點續抓）。"""
    quarters = generate_quarters(start, end)
    print(f"[fetch_transcripts] 開始處理 {ticker}，範圍 {start} ~ {end}，共 {len(quarters)} 個季度: {quarters}")

    # 讀取現有索引以跳過已抓取的季度
    out_dir = paths.RAW_TRANSCRIPTS / ticker
    index_path = out_dir / "index.json"
    existing_quarters = set()
    if index_path.exists():
        existing_index = read_json(index_path)
        existing_quarters = {t.get("quarter") for t in existing_index.get("transcripts", []) if t.get("quarter")}

    # 取得歷史財報公布日期對照表
    quarter_calendar = fetch_earnings_calendar(ticker, api_key)

    saved_count = 0
    for q in quarters:
        if q in existing_quarters:
            print(f"[fetch_transcripts] {ticker} {q} 已存在，跳過")
            continue

        print(f"[fetch_transcripts] 正在抓取 {ticker} {q}...")
        data = fetch_transcript_by_quarter(ticker, q, api_key)
        if not data:
            print(f"[fetch_transcripts] {ticker} {q} 無資料或下載失敗")
            time.sleep(delay_sec)
            continue

        event_date = quarter_calendar.get(q) or default_quarter_event_date(q)

        # 防呆：這一季的財報公布日還沒到（不論是 EARNINGS 日曆查到的真實日期，還是沒查到
        # 時用的估算日期），代表這場法說會實際上還沒開。Alpha Vantage 對這種「還沒發生」的
        # 季度，觀察到會回傳看似正常、格式正確、但內容是合成／推測出來的逐字稿（不是空
        # 資料、也不是明確的錯誤訊息），必須主動擋掉，否則會把假資料當成真實資料存進
        # pipeline（實際發生過一次：AAPL 2026Q3，估算日期 2026-10-25，當時系統日期
        # 2026-09-02，尚未開完，但 API 仍回傳一份內容詳實、格式正確的「逐字稿」）。
        if event_date > dt.date.today().isoformat():
            print(f"[fetch_transcripts] {ticker} {q} 的公布日 {event_date} 尚未到（今天 "
                  f"{dt.date.today().isoformat()}），這場法說會實際上還沒開，Alpha Vantage 回傳的內容"
                  f"疑似合成／推測資料，已跳過不儲存")
            time.sleep(delay_sec)
            continue

        text = transcript_to_text(data)
        if text:
            save_transcript(ticker, event_date, q, text)
            saved_count += 1
        time.sleep(delay_sec)

    return saved_count


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="使用 Alpha Vantage 下載法說會逐字稿")
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--start", default=cfg["date_range"]["start"])
    ap.add_argument("--end", default=cfg["date_range"]["end"])
    ap.add_argument("--api_key", default=None, help="Alpha Vantage API Key（未提供則讀取環境變數）")
    ap.add_argument("--delay", type=float, default=1.0, help="每次 API 請求間隔秒數（免費方案建議設為 12.0 秒）")
    args = ap.parse_args()

    api_key = get_api_key(args.api_key)
    n = fetch_all_transcripts(args.ticker, args.start, args.end, api_key, args.delay)
    print(f"[fetch_transcripts] {args.ticker}: 本次共新增 {n} 篇逐字稿 -> {paths.RAW_TRANSCRIPTS / args.ticker}")


if __name__ == "__main__":
    main()
