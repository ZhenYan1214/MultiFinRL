"""從 SEC EDGAR 下載財報文件（10-K / 10-Q），存到 data/raw/filings/{TICKER}/。

使用 SEC 官方 API（免費、無需金鑰，但必須帶 User-Agent 表明身分）：
  1. company_tickers.json     -> ticker 轉 CIK
  2. submissions/CIK{...}.json -> 該公司所有 filing 清單
  3. 逐份下載主文件 HTML

輸出：
  data/raw/filings/{TICKER}/index.json                 # filing 清單（型別/日期/檔名）
  data/raw/filings/{TICKER}/{TYPE}_{DATE}.html          # 原始文件

用法：
    python -m module_a_data.crawler.fetch_filings --ticker AAPL --start 2021-01-01
"""
import argparse
import time

import requests

from shared import paths
from shared.utils import write_json

# SEC 要求 User-Agent 含聯絡方式，請改成自己的
HEADERS = {"User-Agent": "MultiFinRL research project XXXXXXXXX@gmail.com"}
FORMS = {"10-K", "10-Q"}


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
    for form, date, accession, doc in zip(
        recent["form"], recent["filingDate"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form in FORMS and date >= start:
            filings.append({"form": form, "filing_date": date,
                            "accession": accession.replace("-", ""), "document": doc})
    return filings


def download_filings(ticker: str, start: str) -> None:
    cik = ticker_to_cik(ticker)
    filings = list_filings(cik, start)
    out_dir = paths.RAW_FILINGS / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for f in filings:
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{f['accession']}/{f['document']}"
        fname = f"{f['form']}_{f['filing_date']}.html"
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        (out_dir / fname).write_bytes(r.content)
        index.append({"form": f["form"], "filing_date": f["filing_date"], "file": fname})
        print(f"[fetch_filings] {ticker} {f['form']} {f['filing_date']}")
        time.sleep(0.2)  # SEC 流量限制：每秒 <= 10 requests

    write_json({"ticker": ticker, "cik": cik, "filings": index}, out_dir / "index.json")
    print(f"[fetch_filings] {ticker}: {len(index)} filings -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2021-01-01")
    args = ap.parse_args()
    download_filings(args.ticker, args.start)


if __name__ == "__main__":
    main()
