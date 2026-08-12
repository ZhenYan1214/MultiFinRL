"""從新聞、財報及法說會文字抽取重大財經事件，評估 Precision / Recall / F1。

兩種方法，用 --method 切換：
    keyword（預設）：規則式關鍵字比對，免費、隨時可跑，當基準線／對照組。
    llm            ：呼叫 LLM API 判斷，需要金鑰、會產生費用，見
                     docs/spec_b_event_extraction_llm.md 的完整設計說明。

事件類別：EARNINGS / MA / PRODUCT_LAUNCH / LAWSUIT / GUIDANCE / DIVIDEND / MANAGEMENT_CHANGE

ground truth：149 天分層抽樣，人工（Claude 對話中直接標記）標記，見
`data/labels/event_ground_truth/{ticker}.json` 與 `event_ground_truth_prompt.py`。
評估結果輸出到 data/outputs/metrics/event_extraction_report.json（keyword）或
data/outputs/metrics/event_extraction_report_llm_{provider}.json（llm）
（docs/decisions.md #32、docs/spec_b_event_extraction_llm.md）。
"""
import argparse
import re

from shared import paths
from shared.utils import read_json, write_json
from module_b_encoder.llm_client import DEFAULT_MODEL, label_events

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


def _relevant_filing_transcript_chunks(record: dict) -> tuple[list[dict], list[dict]]:
    """只留下日期對得上今天的財報／法說會片段，濾掉沿用中的舊資料背景脈絡——
    跟關鍵字方法同一個邏輯（見上面 extract_from_daily_record 的說明、docs/decisions.md #32
    的主要修正），LLM 方法一樣需要，不然舊資料照樣會被誤判成當天的新事件。
    """
    date = record["date"]
    filing = [c for c in record["filing_chunks"] if c.get("filing_date") == date]
    transcript = [c for c in record["transcript_chunks"] if c.get("event_date") == date]
    return filing, transcript


def extract_llm_from_daily_record(record: dict, provider: str, model: str) -> list[str]:
    """LLM-based 抽取：一天一次呼叫，公司相關性前置過濾 + 沿用資料過濾之後才送進 LLM
    （docs/spec_b_event_extraction_llm.md）。回傳當天的事件類型清單（no 來源追蹤，
    跟 keyword 方法的 events_per_day 格式不同）。"""
    ticker, date = record["ticker"], record["date"]
    news = [n for n in record["news"] if is_company_relevant(ticker, n["headline"])]
    filing_chunks, transcript_chunks = _relevant_filing_transcript_chunks(record)
    return label_events(ticker, date, news, filing_chunks, transcript_chunks, provider, model)


def run_llm_extraction(files: list, provider: str, model: str) -> dict[str, set]:
    """對 files 逐天做 LLM 抽取，中斷續跑：結果邊跑邊存進快取檔，重跑時已經有結果的
    日期直接跳過、不重新呼叫 API、不重複花錢。"""
    ticker = read_json(files[0])["ticker"] if files else None
    cache_path = paths.event_extraction_llm_cache_path(ticker)
    cache: dict[str, list[str]] = read_json(cache_path) if cache_path.exists() else {}

    per_day_types: dict[str, set] = {}
    for i, f in enumerate(files, 1):
        record = read_json(f)
        date = record["date"]
        if date not in cache:
            events = extract_llm_from_daily_record(record, provider, model)
            cache[date] = events
            write_json(cache, cache_path)
            print(f"  [{i}/{len(files)}] {date}: {events}")
        per_day_types[date] = set(cache[date])
    return per_day_types


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
    ap.add_argument("--method", choices=["keyword", "llm"], default="keyword",
                    help="keyword：免費、規則式，隨時可跑；llm：呼叫 LLM API，需要金鑰、"
                         "會產生費用，見 docs/spec_b_event_extraction_llm.md")
    ap.add_argument("--provider", choices=["claude", "openai", "deepseek"], default="deepseek",
                    help="只在 --method llm 時使用")
    ap.add_argument("--model", default=None, help="不指定就用 provider 的預設模型")
    ap.add_argument("--limit_to_ground_truth", action="store_true",
                    help="只處理 ground truth 涵蓋的日期（149 天），用來先驗證 LLM 方法"
                         "效果、不用付全量 1381 天的費用；--method keyword 時無意義")
    args = ap.parse_args()
    model = args.model or DEFAULT_MODEL[args.provider]

    dataset_dir = paths.DATASET / args.ticker
    files = sorted(dataset_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"找不到 A 的資料: {dataset_dir}")

    gt_path = paths.event_ground_truth_path(args.ticker)
    gt = read_json(gt_path) if gt_path.exists() else {}

    if args.limit_to_ground_truth:
        if not gt:
            raise SystemExit(f"--limit_to_ground_truth 需要 ground truth，找不到: {gt_path}")
        files = [f for f in files if f.stem in gt]

    report = {"ticker": args.ticker, "method": args.method, "days": len(files)}

    if args.method == "keyword":
        per_day = {}
        per_day_types: dict[str, set] = {}
        for f in files:
            record = read_json(f)
            events = extract_from_daily_record(record)
            per_day[record["date"]] = events
            per_day_types[record["date"]] = {e["event_type"] for e in events}
        report["events_per_day"] = {d: e for d, e in per_day.items() if e}
        out = paths.OUTPUTS / "metrics" / "event_extraction_report.json"
    else:
        print(f"[event_extraction] method=llm provider={args.provider} model={model}"
             f"（{len(files)} 天，每天一次 API 呼叫，會產生費用，已標記過的日期會跳過）")
        per_day_types = run_llm_extraction(files, args.provider, model)
        report["provider"] = args.provider
        report["model"] = model
        report["events_per_day"] = {d: sorted(t) for d, t in per_day_types.items() if t}
        out = paths.OUTPUTS / "metrics" / f"event_extraction_report_llm_{args.provider}.json"

    if gt:
        dates = sorted(d for d in gt if d in per_day_types)
        predictions = [per_day_types[d] for d in dates]
        ground_truth = [set(gt[d]) for d in dates]
        report["evaluation"] = evaluate(predictions, ground_truth)
        report["evaluation"]["n_days"] = len(dates)
        print(f"[event_extraction] P/R/F1（{len(dates)} 天 ground truth，method={args.method}）："
             f"precision={report['evaluation']['precision']:.3f}, "
             f"recall={report['evaluation']['recall']:.3f}, "
             f"f1={report['evaluation']['f1']:.3f}")
    else:
        report["evaluation"] = f"pending ground truth（找不到 {gt_path}）"

    write_json(report, out)
    print(f"[event_extraction] {len(files)} days -> {out}")


if __name__ == "__main__":
    main()
