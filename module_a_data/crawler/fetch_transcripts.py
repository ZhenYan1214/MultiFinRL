"""使用 Alpha Vantage API 下載法說會逐字稿，存到 data/raw/transcripts/{TICKER}/。

取代原本用 foolcalls（fool.com 爬蟲）的版本——foolcalls 不是正式上架的套件、原始碼本身
有壞掉的 import、且從未成功端到端跑出過真實資料（見 docs/decisions.md #25，`--start`/
`--max_pages` 那個版本）。改用 Alpha Vantage 官方 API，穩定、有官方文件保障。

API 端點：
  - EARNINGS_CALL_TRANSCRIPT: 抓取指定公司與季度的法說會逐字稿及發言人內容
  - EARNINGS（輔助）: 取得每季季末日（fiscalDateEnding）、財報發布日（reportedDate）與
    會計年度結束月，換算成逐字稿端點要的「會計季」，event_date = reportedDate

注意：逐字稿端點的 quarter 是公司會計年度的季別（AAPL 2021Q1 = 2020-12 季），不是日曆季。

環境變數：
  ALPHA_VANTAGE_API_KEY=你的 API key（或 ALPHAVANTAGE_API_KEY）
  可在專案根目錄的 .env 檔案中設定，或透過命令列參數 --api_key 傳入。

輸出格式（與 fetch_filings 對齊，供 build_dataset 讀取）：
  data/raw/transcripts/{TICKER}/index.json              # 清單（event_date / 會計季 quarter /
                                                         #       period_end / report_time / 檔名）
  data/raw/transcripts/{TICKER}/EC_{YYYY-MM-DD}.txt     # 純文字逐字稿

用法：
  python -m module_a_data.crawler.fetch_transcripts --ticker AAPL
  python -m module_a_data.crawler.fetch_transcripts --ticker AAPL --start 2021-01-01 --end 2026-08-09
  python -m module_a_data.crawler.fetch_transcripts --ticker AAPL --check   # 只檢查內文季度，不打 API

免費方案有速率限制（約 5 次/分鐘），--delay 預設 1.0 秒對免費方案太快，
遇到大量重試訊息時改成 --delay 12.0。
"""
import argparse
import datetime as dt
import os
import re
import time
from collections import Counter

import requests

from shared import paths
from shared.utils import load_config, read_json, write_json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_URL = "https://www.alphavantage.co/query"

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def get_api_key(cli_key: str | None = None) -> str:
    """取得 Alpha Vantage API Key（優先順序：CLI 參數 > ALPHA_VANTAGE_API_KEY > ALPHAVANTAGE_API_KEY）。"""
    key = cli_key or os.environ.get("ALPHA_VANTAGE_API_KEY") or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "缺少 Alpha Vantage API Key！請在專案根目錄的 .env 檔案中設定 "
            "ALPHA_VANTAGE_API_KEY=your_key，或透過 --api_key 傳入。"
        )
    return key


# ---------------------------------------------------------------------------
# 會計季對應
#
# EARNINGS_CALL_TRANSCRIPT 的 `quarter` 參數是「公司會計年度的季別」，不是日曆季：
#   AAPL（9 月結）2021Q1 = 2020-12 季，2021-01-27 開的法說會
#   MSFT（6 月結）2024Q1 = 2023-09 季（內文：Fiscal Year 2024 First Quarter）
#   NVDA（1 月結）2021Q1 = 2020-04 季（內文：first quarter of fiscal 2021）
# 舊版把 EARNINGS 的 fiscalDateEnding 換成日曆季再拿去要逐字稿，AAPL 每份晚一季、NVDA 晚三季。
# 現在改成：以 EARNINGS 每一筆（季末日、公布日）為準，依公司會計年度結束月換算會計季，
# 用會計季要逐字稿，event_date = 公布日（reportedDate）。
# ---------------------------------------------------------------------------

def _period_end_month(period_end: str) -> tuple[int, int]:
    """季末日 → (年, 月)。52/53 週制的季末可能落在下個月月初（例如 10-01），歸回上個月。"""
    d = dt.date.fromisoformat(period_end)
    if d.day <= 7:
        d = d.replace(day=1) - dt.timedelta(days=1)
    return d.year, d.month


def fiscal_quarter_label(period_end: str, fy_end_month: int) -> str | None:
    """季末日 + 會計年度結束月 → 逐字稿端點用的會計季（例如 AAPL 2020-12-31 → '2021Q1'）。

    會計年度以「結束那一年」命名（AAPL FY2024 於 2024-09 結束；NVDA FY2025 於 2025-01 結束）。
    季末月份跟會計年度結束月對不上 3 的倍數時回傳 None（資料異常，呼叫端跳過）。
    """
    year, month = _period_end_month(period_end)
    k = (month - fy_end_month) % 12
    if k % 3:
        return None
    q = 4 if k == 0 else k // 3
    fy = year if month <= fy_end_month else year + 1
    return f"{fy}Q{q}"


def fetch_earnings_calendar(ticker: str, api_key: str) -> tuple[int, list[dict]]:
    """呼叫 EARNINGS 端點，回傳 (會計年度結束月, 每季 [{period_end, reported_date, report_time}])。

    會計年度結束月取 annualEarnings 各年 fiscalDateEnding 最常見的月份。查不到就直接報錯，
    不再用估算日期（估算日期曾造成存下合成逐字稿，見 fetch_all_transcripts 的防呆說明）。
    """
    params = {"function": "EARNINGS", "symbol": ticker, "apikey": api_key}
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    annual = [a["fiscalDateEnding"] for a in data.get("annualEarnings", []) if a.get("fiscalDateEnding")]
    quarterly = data.get("quarterlyEarnings", [])
    if not annual or not quarterly:
        msg = data.get("Information") or data.get("Note") or data.get("Error Message") or "回應沒有財報資料"
        raise RuntimeError(f"[fetch_transcripts] {ticker} EARNINGS 查詢失敗：{msg}")
    fy_end_month = Counter(_period_end_month(a)[1] for a in annual).most_common(1)[0][0]
    entries = [
        {"period_end": q["fiscalDateEnding"], "reported_date": q["reportedDate"],
         "report_time": q.get("reportTime")}
        for q in quarterly if q.get("fiscalDateEnding") and q.get("reportedDate")
    ]
    return fy_end_month, sorted(entries, key=lambda e: e["reported_date"])


# ---------------------------------------------------------------------------
# 內文季度檢查
# ---------------------------------------------------------------------------

_MONTH_QUARTER_RE = re.compile(r"\b(" + "|".join(MONTH_NAMES) + r")[- ]quarter\b", re.I)
_FISCAL_RES = [
    # 「Q2 Fiscal Year 2025」「Q1 fiscal 2026」
    (re.compile(r"\bQ([1-4]) (?:of )?fiscal (?:year )?(\d{4})\b", re.I), False),
    # 「first quarter of fiscal 2021」「second quarter of fiscal year 2022」
    (re.compile(r"\b(first|second|third|fourth) quarter of fiscal (?:year )?(\d{4})\b", re.I), False),
    # 「Fiscal Year 2024 First Quarter」
    (re.compile(r"\bfiscal (?:year )?(\d{4}) (first|second|third|fourth) quarter\b", re.I), True),
]


def content_quarter_problems(text: str, period_end: str, label: str) -> tuple[list[str], list[str]]:
    """比對內文提到的季度跟這份逐字稿應該對應的季度，回傳 (錯誤, 警告)；兩者皆空 = 一致。

    兩種訊號，有出現才比：
      1. 會計季寫法（「Q2 Fiscal Year 2025」「first quarter of fiscal 2021」「Fiscal Year 2024
         First Quarter」）最常出現的 vs label。開場白一定會講，最可靠。
      2. 「December／March／June／September quarter」等月份季最常出現的月份 vs 季末月份
         （Apple 這類用月份稱呼季度的公司）。會被下季展望拉到：AAPL 2025-05-01（關稅）、
         2026-04-30（記憶體成本）兩場談 June quarter 比 March quarter 多。所以會計季寫法已經
         確認一致時，月份季對不上只算警告；沒有會計季寫法可確認時才算錯誤。
    """
    errors, warnings = [], []
    fiscal = Counter()
    for pattern, year_first in _FISCAL_RES:
        for m in pattern.finditer(text):
            a, b = m.group(1), m.group(2)
            q, y = (b, a) if year_first else (a, b)
            q = _ORDINAL.get(q.lower(), q)
            fiscal[f"{y}Q{q}"] += 1
    fiscal_ok = False
    if fiscal:
        top, n = fiscal.most_common(1)[0]
        fiscal_ok = top == label
        if not fiscal_ok:
            errors.append(f"內文最常提到會計季 {top}（{n} 次），應是 {label}")

    _, expect_month = _period_end_month(period_end)
    months = Counter(m.group(1).capitalize() for m in _MONTH_QUARTER_RE.finditer(text))
    if months:
        top, n = months.most_common(1)[0]
        if top != MONTH_NAMES[expect_month - 1]:
            msg = (f"內文最常提到 {top} quarter（{n} 次，全部 {dict(months)}），"
                   f"季末應是 {MONTH_NAMES[expect_month - 1]}")
            if fiscal_ok:
                warnings.append(msg + f"；但會計季寫法 {label} 一致，多半是下季展望談得多")
            else:
                errors.append(msg)
    return errors, warnings


def transcript_header(ticker: str, label: str, period_end: str, reported_date: str) -> str:
    return (f"{ticker} fiscal {label} Earnings Call Transcript "
            f"(quarter ended {period_end}, call date {reported_date})\n")


def transcript_to_text(data: dict, header: str) -> str:
    """將 Alpha Vantage 的逐字稿 JSON 轉換成純文字格式（第一行是 header）。"""
    parts = [header]

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


def _existing_text(ticker: str, label: str, index: dict) -> str | None:
    """斷點續抓：舊 index 裡已有同一個會計季的檔案就沿用內文（只換 header），不再打 API。

    舊版檔案的第一行是「AAPL 2021Q1 Earnings Call Transcript」，2021Q1 就是當時傳給 API 的
    會計季，所以內文本身是對的，錯的只有日期；確認第一行的季別一致才沿用。
    """
    out_dir = paths.RAW_TRANSCRIPTS / ticker
    for t in index.get("transcripts", []):
        if t.get("quarter") != label:
            continue
        f = out_dir / t["file"]
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        first, _, body = text.partition("\n")
        if re.match(rf"^{re.escape(ticker)} (?:fiscal )?{label} ", first):
            return body.lstrip("\n")
    return None


def select_calls(entries: list[dict], start: str, end: str) -> list[dict]:
    """取公布日落在 [start, end] 的各季，再加上 start 之前最近的一場（讓起始那幾天也有法說會可用）。"""
    in_range = [e for e in entries if start <= e["reported_date"] <= end]
    before = [e for e in entries if e["reported_date"] < start]
    return ([before[-1]] if before else []) + in_range


def fetch_all_transcripts(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    delay_sec: float = 1.0,
) -> int:
    """依 EARNINGS 日曆逐場下載逐字稿，重寫 index.json；回傳新打 API 抓到的篇數。"""
    fy_end_month, entries = fetch_earnings_calendar(ticker, api_key)
    calls = select_calls(entries, start, end)
    print(f"[fetch_transcripts] {ticker} 會計年度結束月={fy_end_month}，範圍 {start} ~ {end}，"
          f"共 {len(calls)} 場法說會")

    out_dir = paths.RAW_TRANSCRIPTS / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    old_index = read_json(index_path) if index_path.exists() else {"transcripts": []}
    today = dt.date.today().isoformat()

    new_entries: list[dict] = []
    texts: dict[str, str] = {}
    fetched = 0
    for e in calls:
        label = fiscal_quarter_label(e["period_end"], fy_end_month)
        event_date = e["reported_date"]
        if label is None:
            print(f"[fetch_transcripts] {ticker} 季末 {e['period_end']} 對不上會計年度結束月 "
                  f"{fy_end_month}，跳過")
            continue

        # 防呆：公布日還沒到，代表這場法說會實際上還沒開。Alpha Vantage 對這種「還沒發生」的
        # 季度，觀察到會回傳看似正常、格式正確、但內容是合成／推測出來的逐字稿（不是空資料、
        # 也不是明確的錯誤訊息），必須主動擋掉，否則會把假資料當成真實資料存進 pipeline
        # （實際發生過一次：AAPL 2026Q3，估算日期 2026-10-25，當時系統日期 2026-09-02）。
        if event_date > today:
            print(f"[fetch_transcripts] {ticker} {label} 的公布日 {event_date} 尚未到（今天 {today}），跳過")
            continue

        header = transcript_header(ticker, label, e["period_end"], event_date)
        body = _existing_text(ticker, label, old_index)
        if body is not None:
            text = header + "\n" + body
            print(f"[fetch_transcripts] {ticker} {label} 沿用既有內文")
        else:
            print(f"[fetch_transcripts] 正在抓取 {ticker} {label}（{event_date}）...")
            data = fetch_transcript_by_quarter(ticker, label, api_key)
            time.sleep(delay_sec)
            if not data:
                print(f"[fetch_transcripts] {ticker} {label} 無資料或下載失敗")
                continue
            if str(data.get("quarter", label)) != label:
                print(f"[fetch_transcripts] {ticker} 要 {label} 卻回傳 {data.get('quarter')}，跳過")
                continue
            text = transcript_to_text(data, header)
            fetched += 1

        errors, _ = content_quarter_problems(text, e["period_end"], label)
        for p in errors:
            print(f"[fetch_transcripts] ⚠ {ticker} {label}（{event_date}）季度對不上：{p}")
        fname = f"EC_{event_date}.txt"
        texts[fname] = text
        new_entries.append({"event_date": event_date, "quarter": label,
                            "period_end": e["period_end"], "report_time": e["report_time"],
                            "file": fname})

    # 範圍外、但日期跟 EARNINGS 日曆一致的舊條目保留；日期錯的舊條目連同檔案一起移除
    calendar = {fiscal_quarter_label(e["period_end"], fy_end_month): e["reported_date"] for e in entries}
    kept = {t["event_date"] for t in new_entries}
    for t in old_index.get("transcripts", []):
        if t["event_date"] not in kept and calendar.get(t.get("quarter")) == t["event_date"] \
                and t.get("period_end"):
            new_entries.append(t)
            kept.add(t["event_date"])

    for fname, text in texts.items():
        (out_dir / fname).write_text(text, encoding="utf-8")
    referenced = {t["file"] for t in new_entries}
    for f in out_dir.glob("EC_*.txt"):
        if f.name not in referenced:
            f.unlink()
            print(f"[fetch_transcripts] 移除日期錯誤的舊檔 {f.name}")
    write_json({"ticker": ticker, "fiscal_year_end_month": fy_end_month,
                "transcripts": sorted(new_entries, key=lambda t: t["event_date"])}, index_path)
    print(f"[fetch_transcripts] {ticker} index 共 {len(new_entries)} 份 -> {index_path}")
    return fetched


def check_transcripts(ticker: str) -> int:
    """不打 API：逐份檢查 index 裡的逐字稿內文季度是否跟 period_end／quarter 一致，回傳對不上的份數。"""
    out_dir = paths.RAW_TRANSCRIPTS / ticker
    index = read_json(out_dir / "index.json")
    bad = 0
    for t in index["transcripts"]:
        if not t.get("period_end"):
            print(f"[check] {t['file']}: index 沒有 period_end（舊版格式），請重抓")
            bad += 1
            continue
        text = (out_dir / t["file"]).read_text(encoding="utf-8")
        errors, warnings = content_quarter_problems(text, t["period_end"], t["quarter"])
        status = "對不上" if errors else ("OK（有警告）" if warnings else "OK")
        print(f"[check] {t['event_date']} {t['quarter']} 季末 {t['period_end']}: {status}"
              + "".join(f"\n    - {p}" for p in errors + warnings))
        bad += bool(errors)
    print(f"[check] {ticker}: {len(index['transcripts'])} 份，{bad} 份對不上")
    return bad


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="使用 Alpha Vantage 下載法說會逐字稿")
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--start", default=cfg["date_range"]["start"])
    ap.add_argument("--end", default=cfg["date_range"]["end"])
    ap.add_argument("--api_key", default=None, help="Alpha Vantage API Key（未提供則讀取環境變數）")
    ap.add_argument("--delay", type=float, default=1.0, help="每次 API 請求間隔秒數（免費方案建議設為 12.0 秒）")
    ap.add_argument("--check", action="store_true", help="只檢查既有逐字稿的季度，不打 API")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(1 if check_transcripts(args.ticker) else 0)
    api_key = get_api_key(args.api_key)
    n = fetch_all_transcripts(args.ticker, args.start, args.end, api_key, args.delay)
    print(f"[fetch_transcripts] {args.ticker}: 本次新打 API 抓了 {n} 篇 -> {paths.RAW_TRANSCRIPTS / args.ticker}")
    check_transcripts(args.ticker)


if __name__ == "__main__":
    main()
