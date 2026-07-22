"""從新聞、財報及法說會文字抽取重大財經事件，評估 Precision / Recall / F1。

ground truth 來源與事件類別定義尚未定案（docs/decisions.md 待確認事項）。
先完成抽取程式本體（基礎版：關鍵詞規則），評估部分等定案後接上真實標註。
評估結果輸出到 data/outputs/metrics/event_extraction_report.json。

基礎版事件類別（暫定，定案後修改 EVENT_KEYWORDS 即可）：
    EARNINGS / MA / PRODUCT_LAUNCH / LAWSUIT / GUIDANCE / DIVIDEND / MANAGEMENT_CHANGE
後續可改成 LLM-based 抽取，只要 extract_events() 的回傳格式不變。
"""
import argparse
import re

from shared import paths
from shared.utils import read_json, write_json

EVENT_KEYWORDS: dict[str, list[str]] = {
    "EARNINGS": ["earnings", "quarterly results", "eps", "revenue beat", "revenue miss"],
    "MA": ["acquisition", "merger", "acquire", "takeover", "buyout"],
    "PRODUCT_LAUNCH": ["launch", "unveil", "new product", "release", "announce"],
    "LAWSUIT": ["lawsuit", "sue", "litigation", "settlement", "antitrust"],
    "GUIDANCE": ["guidance", "outlook", "forecast", "raise", "lower", "cut"],
    "DIVIDEND": ["dividend", "buyback", "share repurchase", "split"],
    "MANAGEMENT_CHANGE": ["ceo", "cfo", "resign", "appoint", "step down"],
}


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
    """對 A 的每日 JSON 抽事件（新聞 + 兩類 chunk）。"""
    events = []
    for n in record["news"]:
        for e in extract_events(f"{n['headline']} {n['content']}"):
            events.append({**e, "source": "news", "ref": n["headline"]})
    for field in ("filing_chunks", "transcript_chunks"):
        for c in record[field]:
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
    for f in files:
        record = read_json(f)
        per_day[record["date"]] = extract_from_daily_record(record)

    report = {
        "ticker": args.ticker,
        "days": len(per_day),
        "events_per_day": {d: e for d, e in per_day.items() if e},
        "evaluation": "pending ground truth (docs/decisions.md)",
    }
    out = paths.OUTPUTS / "metrics" / "event_extraction_report.json"
    write_json(report, out)
    print(f"[event_extraction] {len(per_day)} days -> {out}")


if __name__ == "__main__":
    main()
