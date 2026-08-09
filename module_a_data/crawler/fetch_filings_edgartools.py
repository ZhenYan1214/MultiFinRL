"""用 edgartools 下載財報文件（10-K / 10-Q），存到 data/raw/filings/{TICKER}/。

docs/decisions.md #25：老師同意先試用 edgartools 這類開源工具，看是否符合需求。
這支是 fetch_filings.py（原生 requests + SEC 官方 API）之外的另一個實作，
輸出格式（index.json 結構、欄位名稱）完全對齊，build_dataset.py 不用改一行，
兩支可以並存比較，之後決定留哪一支就把另一支刪掉即可。

跟原本 fetch_filings.py 的差異：
  - 原本手動組 SEC API 請求、自己解析 JSON、下載原始 HTML。
  - 這支交給 edgartools 處理 CIK 查詢、filing 清單、也用它的 filing.text() 直接拿到
    「已經去除 HTML 標籤的乾淨文字」，所以存檔用 .txt 而非 .html
    （下游 build_dataset.py 的 clean() 兩種都能吃，不影響相容性）。
  - edgartools 內建 rate limit 處理，不用自己 time.sleep()。

安裝：pip install edgartools

用法：
    python -m module_a_data.crawler.fetch_filings_edgartools --ticker AAPL --start 2021-01-01
"""
import argparse

from shared import paths
from shared.utils import write_json

FORMS = ("10-K", "10-Q")
IDENTITY = "MultiFinRL research project XXXXXXXXX@gmail.com"  # 同 fetch_filings.py，改成自己的信箱


def download_filings(ticker: str, start: str) -> None:
    from edgar import Company, set_identity

    set_identity(IDENTITY)
    out_dir = paths.RAW_FILINGS / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    company = Company(ticker)
    index = []
    for form in FORMS:
        filings = company.get_filings(form=form)
        for filing in filings:
            if filing.filing_date < start:
                continue
            text = filing.text()
            if not text:
                continue  # 少數 filing 沒有可解析的正文，跳過
            fname = f"{form}_{filing.filing_date}.txt"
            (out_dir / fname).write_text(text, encoding="utf-8")
            index.append({"form": form, "filing_date": filing.filing_date, "file": fname})
            print(f"[fetch_filings_edgartools] {ticker} {form} {filing.filing_date}")

    write_json({"ticker": ticker, "filings": index}, out_dir / "index.json")
    print(f"[fetch_filings_edgartools] {ticker}: {len(index)} filings -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2021-01-01")
    args = ap.parse_args()
    download_filings(args.ticker, args.start)


if __name__ == "__main__":
    main()
