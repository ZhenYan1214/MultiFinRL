"""下載法說會逐字稿，存到 data/raw/transcripts/{TICKER}/。

docs/decisions.md #25：老師同意先試用 foolcalls（fool.com 爬蟲）這類開源工具。

foolcalls 的限制：它不能像「輸入 ticker 就查得到」那樣使用，
fool.com 的逐字稿清單頁是全公司混在一起、按時間排序，不能只查單一 ticker。
所以做法是：翻清單頁（一頁可能有很多公司、很多不是我們要的），
用 ticker 是否出現在 URL 裡（例如 .../q3-2020-earnings-call-tran.aspx
裡的 "krus" 這段對應 KRUS）過濾出我們要的那幾篇，其餘丟棄。
翻頁是由新到舊，遇到比 --start 更早的日期就停止，不用整個網站翻完。

輸出格式（與 fetch_filings 對齊，供 build_dataset 讀取）：
  data/raw/transcripts/{TICKER}/index.json              # 清單（event_date/檔名）
  data/raw/transcripts/{TICKER}/EC_{YYYY-MM-DD}.txt      # 純文字逐字稿

安裝：foolcalls 在 PyPI 上沒有上架、GitHub 上也沒有 setup.py/pyproject.toml，
不能直接 pip install。已經把它整個複製到本專案根目錄的 foolcalls/
（跟 shared/ 同一層），可以直接 `from foolcalls.scrapers import ...`，不用再自己 clone。
還需要 `pip install boto3 lxml`（foolcalls 內部 import 用到，即使不用它的 S3 功能）。

已修好的問題（2026-07 測試時發現）：foolcalls 目前 GitHub 上的 scrapers.py
會 import 一個叫 extractors_v2 的模組，但整個 repo 裡根本沒有這個檔案，
不修就連 import 都會直接失敗（跟網路、跟我們的程式碼都無關，是 foolcalls
本身目前壞掉了）。已經在 foolcalls/extractors_v2.py 補上一行
`from foolcalls.extractors import *` 當 stub，import 就不會再炸掉
（scrape_transcript_v2 只是備援用的次要路徑，本來就很少被呼叫到，
用 v1 的邏輯頂著沒有太大影響）。

用法：
    python -m module_a_data.crawler.fetch_transcripts --ticker AAPL --start 2021-01-01 --max_pages 200
"""
import argparse
import re

from shared import paths
from shared.utils import write_json, read_json


def save_transcript(ticker: str, event_date: str, text: str) -> None:
    """統一儲存介面：來源不論是誰，最後都呼叫這個函式落地。"""
    out_dir = paths.RAW_TRANSCRIPTS / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"EC_{event_date}.txt"
    (out_dir / fname).write_text(text, encoding="utf-8")

    index_path = out_dir / "index.json"
    index = {"ticker": ticker, "transcripts": []}
    if index_path.exists():
        from shared.utils import read_json
        index = read_json(index_path)
    entries = [t for t in index["transcripts"] if t["event_date"] != event_date]
    entries.append({"event_date": event_date, "file": fname})
    index["transcripts"] = sorted(entries, key=lambda t: t["event_date"])
    write_json(index, index_path)
    print(f"[fetch_transcripts] {ticker} {event_date} -> {out_dir / fname}")


def _extract_event_date(url: str) -> str | None:
    """URL 路徑裡有 /YYYY/MM/DD/，直接取出當作 event_date，比解析頁面 metadata 可靠。"""
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}-{mo}-{d}"


def _matches_ticker(url: str, ticker: str) -> bool:
    """URL slug 裡用連字號包住的小寫 ticker 當關鍵字比對，減少誤判（見檔案開頭說明）。"""
    return f"-{ticker.lower()}-" in url.lower()


def _transcript_to_text(parsed: dict) -> str:
    """把 scrape_transcript() 回傳的結構化 dict 攤平成一段純文字，供 chunker.py 使用。

    scrape_transcript() 回傳的 'call_transcript' 欄位是逐句/逐段的清單，
    每個元素的確切欄位名稱依 foolcalls 版本而定，這裡用防禦性寫法：
    有 speaker 就標出來，有 text/content 類欄位就接上，其餘型別直接字串化，
    確保不論欄位名稱怎麼變，至少不會整支腳本壞掉。
    """
    parts = []
    title = parsed.get("title") or parsed.get("call_title")
    if title:
        parts.append(str(title))

    transcript = parsed.get("call_transcript") or []
    for item in transcript:
        if isinstance(item, dict):
            speaker = item.get("speaker") or item.get("name") or ""
            text = item.get("text") or item.get("content") or item.get("statement") or ""
            parts.append(f"{speaker}: {text}" if speaker else str(text))
        else:
            parts.append(str(item))
    return "\n".join(p for p in parts if p)


def fetch_ticker_transcripts(ticker: str, start: str, max_pages: int) -> int:
    """由新到舊翻 fool.com 的逐字稿清單頁，篩出這支股票的，早於 start 就停止翻頁。"""
    import requests
    from foolcalls.scrapers import scrape_transcript_urls_by_page, scrape_transcript

    n = 0
    for page in range(max_pages):
        urls = scrape_transcript_urls_by_page(page)
        if not urls:
            break  # 翻到底了

        page_dates = [d for u in urls if (d := _extract_event_date(u))]
        stop_after_this_page = bool(page_dates) and max(page_dates) < start

        for url in urls:
            if not _matches_ticker(url, ticker):
                continue
            event_date = _extract_event_date(url)
            if not event_date or event_date < start:
                continue
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            parsed = scrape_transcript(resp.content)
            text = _transcript_to_text(parsed)
            if text:
                save_transcript(ticker, event_date, text)
                n += 1

        if stop_after_this_page:
            print(f"[fetch_transcripts] 第 {page} 頁日期已早於 --start，停止翻頁")
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--max_pages", type=int, default=200,
                    help="最多翻幾頁清單頁（fool.com 全公司混排，翻太少可能漏掉這支股票的逐字稿）")
    args = ap.parse_args()
    n = fetch_ticker_transcripts(args.ticker, args.start, args.max_pages)
    print(f"[fetch_transcripts] {args.ticker}: 共存 {n} 篇 -> {paths.RAW_TRANSCRIPTS / args.ticker}")


if __name__ == "__main__":
    main()
