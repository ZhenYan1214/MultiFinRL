"""從新聞、財報及法說會文字抽取重大財經事件，評估 Precision / Recall / F1。

基礎版事件類別（暫定，定案後修改 EVENT_KEYWORDS 即可）：
    EARNINGS / MA / PRODUCT_LAUNCH / LAWSUIT / GUIDANCE / DIVIDEND / MANAGEMENT_CHANGE
後續可改成 LLM-based 抽取，只要 extract_events() 的回傳格式不變。

ground truth：150 天分層抽樣，人工（Claude 對話中直接標記）標記，見
`data/labels/event_ground_truth/{ticker}.json` 與 `event_ground_truth_prompt.py`。
評估結果輸出到 data/outputs/metrics/event_extraction_report.json（docs/decisions.md #32）。
"""
import argparse
import re

from shared import paths
from shared.utils import read_json, write_json

EVENT_KEYWORDS: dict[str, list[str]] = {
    "EARNINGS": ["earnings", "quarterly results", "eps", "revenue beat", "revenue miss"],
    "MA": ["acquisition", "merger", "acquire", "takeover", "buyout"],
    "PRODUCT_LAUNCH": ["launch", "unveil", "new product"],
    "LAWSUIT": ["lawsuit", "sue", "litigation", "settlement", "antitrust"],
    "GUIDANCE": ["guidance", "outlook", "forecast", "raised its guidance", "raise its guidance",
                "lowered its guidance", "lower its guidance", "cut its guidance", "cut its forecast"],
    "DIVIDEND": ["dividend", "buyback", "share repurchase", "stock split"],
    "MANAGEMENT_CHANGE": ["resign", "step down", "appoint", "named ceo", "named cfo",
                         "new ceo", "new cfo"],
}

# ticker -> 公司名稱，用於新聞的公司相關性檢查（docs/decisions.md #32）。
# 只有這裡列出的 ticker 有名稱可查；沒列出的話退回只比對 ticker 本身。
TICKER_COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple",
}


def is_company_relevant(ticker: str, headline: str) -> bool:
    """粗略判斷一則新聞是不是真的在講這家公司，而不是別家公司的新聞裡順帶提到。

    規則：標題有出現公司名稱或 ticker 才算——標題通常最能反映一篇文章實際在講誰，
    只在內文順帶提到、比較列表、大盤總覽這類文章的標題通常不會出現公司名稱。
    這是粗略規則，不是完美判斷，見 docs/decisions.md #32 的已知限制。
    """
    name = TICKER_COMPANY_NAMES.get(ticker, ticker)
    headline_lower = headline.lower()
    return ticker.lower() in headline_lower or name.lower() in headline_lower


def extract_events(text: str) -> list[dict]:
    """回傳 [{"event_type": ..., "evidence": 命中的關鍵詞}]，同類型只回報一次。"""
    text_lower = text.lower()
    events = []
    for event_type, keywords in EVENT_KEYWORDS.items():
        hit = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text_lower)]
        if hit:
            events.append({"event_type": event_type, "evidence": hit})
    return events


def extract_from_daily_record(record: dict) -> list[dict]:
    """對 A 的每日 JSON 抽事件（新聞 + 兩類 chunk）。

    新聞：先做公司相關性檢查，不相關的新聞直接跳過，不進關鍵字比對。
    財報／法說會 chunk：collect_filing_chunks()/collect_transcript_chunks()
    是「沿用最新一份到下一份發布為止」（build_dataset.py），所以同一份財報會被
    很多天重複讀到，不代表那天真的有新事件——只有 chunk 的 filing_date/event_date
    等於這天的 record["date"]，才是這天真正新發生的財報/法說會，才進關鍵字比對，
    否則只是拿來當背景脈絡用的舊資料，跳過（docs/decisions.md #32）。
    """
    ticker = record["ticker"]
    date = record["date"]
    events = []
    for n in record["news"]:
        if not is_company_relevant(ticker, n["headline"]):
            continue
        for e in extract_events(f"{n['headline']} {n['content']}"):
            events.append({**e, "source": "news", "ref": n["headline"]})
    for field, date_key in (("filing_chunks", "filing_date"), ("transcript_chunks", "event_date")):
        for c in record[field]:
            if c.get(date_key) != date:
                continue
            for e in extract_events(c["text"]):
                events.append({**e, "source": field, "ref": c["chunk_id"]})
    return events


def evaluate(predictions: list[set], ground_truth: list[set]) -> dict:
    """句/日級多標籤 P/R/F1（micro）。ground truth 定案後接上真實標註即可用。"""
    tp = sum(len(p & g) for p, g in zip(predictions, ground_truth))
    fp = sum(len(p - g) for p, g in zip(predictions, ground_truth))
    fn = sum(len(g - p) for p, g in zip(predictions, ground_truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    args = ap.parse_args()

    dataset_dir = paths.DATASET / args.ticker
    files = sorted(dataset_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"找不到 A 的資料: {dataset_dir}")

    per_day = {}
    per_day_types: dict[str, set] = {}
    for f in files:
        record = read_json(f)
        events = extract_from_daily_record(record)
        per_day[record["date"]] = events
        per_day_types[record["date"]] = {e["event_type"] for e in events}

    report = {
        "ticker": args.ticker,
        "days": len(per_day),
        "events_per_day": {d: e for d, e in per_day.items() if e},
    }

    gt_path = paths.event_ground_truth_path(args.ticker)
    if gt_path.exists():
        gt = read_json(gt_path)
        dates = sorted(d for d in gt if d in per_day_types)
        predictions = [per_day_types[d] for d in dates]
        ground_truth = [set(gt[d]) for d in dates]
        report["evaluation"] = evaluate(predictions, ground_truth)
        report["evaluation"]["n_days"] = len(dates)
        print(f"[event_extraction] P/R/F1（{len(dates)} 天 ground truth）："
             f"precision={report['evaluation']['precision']:.3f}, "
             f"recall={report['evaluation']['recall']:.3f}, "
             f"f1={report['evaluation']['f1']:.3f}")
    else:
        report["evaluation"] = f"pending ground truth（找不到 {gt_path}）"

    out = paths.OUTPUTS / "metrics" / "event_extraction_report.json"
    write_json(report, out)
    print(f"[event_extraction] {len(per_day)} days -> {out}")


if __name__ == "__main__":
    main()
