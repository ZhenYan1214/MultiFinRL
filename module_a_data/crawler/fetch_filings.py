"""從 SEC EDGAR 下載財報文件（10-K / 10-Q / 8-K），存到 data/raw/filings/{TICKER}/。

使用 SEC 官方 API（免費、無需金鑰，但必須帶 User-Agent 表明身分）：
  1. company_tickers.json     -> ticker 轉 CIK
  2. submissions/CIK{...}.json -> 該公司所有 filing 清單
  3. 逐份下載主文件 HTML

輸出：
  data/raw/filings/{TICKER}/index.json                 # filing 清單（型別/日期/檔名）
  data/raw/filings/{TICKER}/{TYPE}_{DATE}.html          # 原始文件

8-K（重大訊息即時揭露）：decisions.md #41/#44/#45——10-K/10-Q 走「取最新一份沿用到下一
份發布為止」的背景邏輯不變，8-K 在 build_dataset.py 的 collect_filing_chunks() 另外處理
成標時間戳記的補充事件 chunk，不會互相覆蓋，這裡只負責把三種表格都抓下來、寫進同一份
index.json，下游自己依 form 分流。

用法：
    python -m module_a_data.crawler.fetch_filings --ticker AAPL --start 2021-01-01
"""
import argparse
import datetime as dt
import time
from zoneinfo import ZoneInfo

import requests

from shared import paths
from shared.utils import write_json

# SEC 要求 User-Agent 含聯絡方式，請改成自己的
HEADERS = {"User-Agent": "MultiFinRL research project XXXXXXXXX@gmail.com"}
FORMS = {"10-K", "10-Q", "8-K"}
ET = ZoneInfo("America/New_York")


def _available_date(accepted_at: str, filing_date: str) -> str:
    """SEC 文件在 16:00 ET 後才受理時，下一個平日才可進入收盤後決策狀態。"""
    if accepted_at:
        accepted = dt.datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
        if accepted.tzinfo is None:
            accepted = accepted.replace(tzinfo=ET)
        else:
            accepted = accepted.astimezone(ET)
        day = accepted.date()
        if accepted.time() >= dt.time(16, 0):
            day += dt.timedelta(days=1)
    else:
        # 舊資料或 API 缺欄位時保守延後一天，避免把盤後文件提前到同日。
        day = dt.date.fromisoformat(filing_date) + dt.timedelta(days=1)
    while day.weekday() >= 5:
        day += dt.timedelta(days=1)
    return day.isoformat()


def ticker_to_cik(ticker: str) -> str:
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    for item in r.json().values():
        if item["ticker"].upper() == ticker.upper():
            return str(item["cik_str"]).zfill(10)
    raise ValueError(f"CIK not found for {ticker}")


def list_filings(cik: str, start: str) -> list[dict]:
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS, timeout=30)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    filings = []
    accepted_times = recent.get("acceptanceDateTime", [""] * len(recent["form"]))
    for form, date, accepted_at, accession, doc in zip(
        recent["form"], recent["filingDate"], accepted_times,
        recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form in FORMS and date >= start:
            filings.append({"form": form, "filing_date": date,
                            "accepted_at": accepted_at,
                            "available_date": _available_date(accepted_at, date),
                            "accession": accession.replace("-", ""), "document": doc})
    return filings


def download_filings(ticker: str, start: str) -> None:
    cik = ticker_to_cik(ticker)
    filings = list_filings(cik, start)
    out_dir = paths.RAW_FILINGS / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    downloaded = 0
    skipped = 0
    for f in filings:
        fname = f"{f['form']}_{f['filing_date']}.html"
        out_path = out_dir / fname
        index.append({
            "form": f["form"], "filing_date": f["filing_date"],
            "accepted_at": f["accepted_at"], "available_date": f["available_date"],
            "file": fname,
        })

        if out_path.is_file():
            skipped += 1
            print(f"[fetch_filings] {ticker} {f['form']} {f['filing_date']} 已存在，跳過")
            continue

        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{f['accession']}/{f['document']}"
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        downloaded += 1
        print(f"[fetch_filings] {ticker} {f['form']} {f['filing_date']}")
        time.sleep(0.2)  # SEC 流量限制：每秒 <= 10 requests

    write_json({"ticker": ticker, "cik": cik, "filings": index}, out_dir / "index.json")
    print(f"[fetch_filings] {ticker}: 共 {len(index)} 份，下載 {downloaded} 份，"
          f"跳過 {skipped} 份 -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2021-01-01")
    args = ap.parse_args()
    download_filings(args.ticker, args.start)


if __name__ == "__main__":
    main()
