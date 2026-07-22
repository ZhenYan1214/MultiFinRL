"""下載法說會逐字稿，存到 data/raw/transcripts/{TICKER}/。

來源尚未定案（負責人未指定，自行尋找）。候選：
  - Motley Fool（網頁爬蟲，免費但需解析 HTML）
  - Seeking Alpha（多數需訂閱）
  - API Ninjas earnings-call-transcript API（有免費額度）

輸出格式（與 fetch_filings 對齊，供 build_dataset 讀取）：
  data/raw/transcripts/{TICKER}/index.json              # 清單（event_date/檔名）
  data/raw/transcripts/{TICKER}/EC_{YYYY-MM-DD}.txt      # 純文字逐字稿

實作時：抓回原始 HTML 後用 preprocess.text_cleaner.clean() 清洗，
再依 index.json 格式登錄。下游只讀 index.json 與 txt，來源可隨時替換。
"""
import argparse

from shared import paths
from shared.utils import write_json


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.parse_args()
    raise NotImplementedError(
        "逐字稿來源尚未定案（見 docs/decisions.md）。"
        "定案後實作抓取邏輯，並以 save_transcript() 落地。"
    )


if __name__ == "__main__":
    main()
