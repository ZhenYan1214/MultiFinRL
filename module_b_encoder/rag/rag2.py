"""
RAG-2：Decoder 生成前，把分析日 t 可得的文件縮成 15–20 段短版文字（decisions.md #113）。

不做相似度排序、不學 W_q、不用 K 線敘述當 query、不算 Loss；Decoder 的注意力自己決定參考哪幾段。
候選直接從 A 的每日 JSON 取（build_dataset.py 已經依 available date 過濾，只含 t 當下可得的文件）：

    1. 當天 8-K：全收（每份取 Item 段落開頭，最多 2 段）
    2. 當天新聞（days_ago = 0，盤後新聞已歸到下一個交易日）：全收，標題＋第一句
    3. 當前背景 10-K／10-Q：固定挑 MD&A、風險因素（Item 1A）的開頭各 1 段（跳過目錄）
    4. 最近一場法說會：挑 rag2.transcript_snippets 段（財務關鍵字最多的段落，依原順序）
    5. t 以前的 8-K：最近的幾份各 1 段，補到上限
  總數上限 rag2.max_snippets；超過時從新聞尾端刪。每段壓成約 rag2.snippet_words 個英文字，
  加上「[來源 日期]」標頭。

輸出 data/outputs/rag2/{TICKER}/{date}.json：
    {"ticker", "date", "snippets": [{"source", "doc_type", "date", "section", "chunk_id", "text"}],
     "context": "[8-K 2024-05-02] ...\n[News 2024-05-02] ...\n..."}   # Decoder 直接讀 context

用法：
    python -m module_b_encoder.rag.rag2 --ticker AAPL
    python -m module_b_encoder.rag.rag2 --ticker AAPL --date 2024-05-02 --show
"""
import argparse
import re
from collections import Counter

from shared import paths
from shared.utils import load_config, read_json, write_json

SECTION_PATTERNS = {
    "mda": re.compile(r"Item\s+[27]\.\s+Management.s Discussion and Analysis"
                      r"(\s+of Financial Condition and Results of Operations)?", re.I),
    "risk_factors": re.compile(r"Item\s+1A\.\s+Risk Factors", re.I),
}
SECTION_NAMES = {"mda": "MD&A", "risk_factors": "Risk Factors"}
ITEM_8K = re.compile(r"Item\s+(\d\.\d{2})\s+(?!Financial Statements and Exhibits)([A-Z][^.]{3,90}\.)")
TAG_PREFIX = re.compile(r"^\[[^\]]*\]\s*")      # build_dataset 加的「[N 天前的…]」標籤
TRANSCRIPT_KEYWORDS = re.compile(
    r"\b(revenue|guidance|outlook|margin|growth|quarter|demand|record|expect|forecast|earnings|"
    r"operating income|EPS|billion|percent|year-over-year)\b", re.I)


def shorten(text: str, n_words: int) -> str:
    words = re.sub(r"\s+", " ", TAG_PREFIX.sub("", text)).strip().split(" ")
    return " ".join(words[:n_words]) + (" …" if len(words) > n_words else "")


ABBREVIATIONS = {"inc", "corp", "co", "ltd", "mr", "ms", "dr", "vs", "st", "jr", "no", "u.s", "u.k", "e.g", "i.e"}


def first_sentence(text: str) -> str:
    """第一句；「U.S.」「Inc.」這類縮寫後面的句點不算句尾。"""
    text = re.sub(r"\s+", " ", text).strip()
    for m in re.finditer(r"[.!?](?=\s+[A-Z“\"(]|$)", text):
        word = re.findall(r"[\w.]+$", text[: m.start()])
        if word and (word[0].lower() in ABBREVIATIONS or (len(word[0]) == 1 and word[0].isalpha())
                     or "." in word[0]):
            continue
        return text[: m.end()]
    return text


BOILERPLATE = re.compile(r"forward-looking|Private Securities Litigation|assumes no obligation|"
                         r"Unless otherwise stated|should be read in conjunction", re.I)


def skip_boilerplate(text: str) -> str:
    """章節開頭常是 forward-looking 免責聲明，跳過這類句子，從第一句實質內容開始。"""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", text).strip())
    for i, sentence in enumerate(sentences[:12]):
        if not BOILERPLATE.search(sentence):
            return " ".join(sentences[i:])
    return text


def _doc_key(chunk_id: str) -> str:
    return chunk_id.rsplit("_", 1)[0]


def _section_start(chunks: list[dict], pattern: re.Pattern) -> tuple[dict, int] | None:
    """找章節標題真正開始的位置；目錄裡的同名標題（後面緊接著下一個 Item）跳過。"""
    for c in chunks:
        for m in pattern.finditer(c["text"]):
            after = c["text"][m.end(): m.end() + 160]
            if re.search(r"\bItem\s+\d", after) or re.match(r"\s*\d+\s", after):
                continue
            return c, m.end()
    return None


def _eight_k_snippets(chunks: list[dict], n_words: int, max_per_doc: int) -> list[dict]:
    out = []
    for c in chunks:
        for m in ITEM_8K.finditer(c["text"]):
            text = f"Item {m.group(1)} {m.group(2)} " + c["text"][m.end():]
            out.append({"chunk_id": c["chunk_id"], "text": shorten(text, n_words)})
            break
        if len(out) >= max_per_doc:
            break
    return out


def build_snippets(record: dict, cfg: dict) -> list[dict]:
    rc = cfg["rag2"]
    n_words, limit = rc["snippet_words"], rc["max_snippets"]
    date = record["date"]
    filings = record["filing_chunks"]

    def item(source, doc_type, doc_date, chunk_id, text, section=None):
        return {"source": source, "doc_type": doc_type, "date": doc_date, "section": section,
                "chunk_id": chunk_id, "text": text}

    # 依文件分組，8-K 依日期新到舊
    docs: dict[str, list[dict]] = {}
    for c in filings:
        docs.setdefault(_doc_key(c["chunk_id"]), []).append(c)
    eight_k = sorted((k for k, v in docs.items() if v[0]["doc_type"] == "8-K"),
                     key=lambda k: docs[k][0]["filing_date"], reverse=True)

    today_8k, older_8k = [], []
    for key in eight_k:
        chunks = docs[key]
        filing_date = chunks[0]["filing_date"]
        target = today_8k if filing_date == date else older_8k
        for s in _eight_k_snippets(chunks, n_words, 2 if filing_date == date else 1):
            target.append(item("8-K", "8-K", filing_date, s["chunk_id"], s["text"]))

    news = [item("News", "news", n["published_at"][:10], None,
                 shorten(f"{n['headline']}. {first_sentence(n['content'])}", n_words))
            for n in record["news"] if int(n["days_ago"]) == 0]

    background = []
    periodic = [k for k, v in docs.items() if v[0]["doc_type"] in ("10-K", "10-Q")]
    for key in sorted(periodic, key=lambda k: docs[k][0]["filing_date"], reverse=True)[:1]:
        chunks = docs[key]
        for section in rc["background_sections"]:
            hit = _section_start(chunks, SECTION_PATTERNS[section])
            if hit:
                c, start = hit
                i = chunks.index(c)   # 免責聲明可能跨到下一段 chunk，接著讀兩段
                body = " ".join([c["text"][start:]] + [x["text"] for x in chunks[i + 1:i + 3]])
                background.append(item(c["doc_type"], c["doc_type"], c["filing_date"], c["chunk_id"],
                                       shorten(skip_boilerplate(body), n_words),
                                       SECTION_NAMES[section]))

    transcript = []
    tx = record["transcript_chunks"]
    if tx:
        latest = max(c["event_date"] for c in tx)
        call = [c for c in tx if c["event_date"] == latest]
        scored = sorted(range(len(call)), key=lambda i: -len(TRANSCRIPT_KEYWORDS.findall(call[i]["text"])))
        for i in sorted(scored[: rc["transcript_snippets"]]):
            c = call[i]
            transcript.append(item("Earnings call", "earnings_call", c["event_date"], c["chunk_id"],
                                   shorten(c["text"], n_words)))

    fixed = today_8k + background + transcript
    room = max(0, limit - len(fixed))
    snippets = today_8k + news[:room] + background + transcript
    room = max(0, limit - len(snippets))
    snippets += older_8k[:room]
    return snippets[:limit]


def render(snippets: list[dict]) -> str:
    """Decoder context 用的文字：每段一行，「[來源 日期 章節] 短版文字」。"""
    lines = []
    for s in snippets:
        head = f"{s['source']} {s['date']}" + (f" {s['section']}" if s.get("section") else "")
        lines.append(f"[{head}] {s['text']}")
    return "\n".join(lines)


def load_context(ticker: str, date: str) -> str:
    path = paths.rag2_path(ticker, date)
    if not path.exists():
        raise SystemExit(f"找不到 RAG-2 候選 {path}，先跑 python -m module_b_encoder.rag.rag2 --ticker {ticker}")
    return read_json(path)["context"]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--date", default=None, help="只做這一天")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="印出 context（人工抽查用）")
    args = ap.parse_args()

    files = sorted((paths.DATASET / args.ticker).glob("*.json"))
    if args.date:
        files = [f for f in files if f.stem == args.date]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"找不到 {paths.DATASET / args.ticker} 的每日 JSON")
    counts, sources = [], Counter()
    for f in files:
        record = read_json(f)
        snippets = build_snippets(record, cfg)
        context = render(snippets)
        write_json({"ticker": args.ticker, "date": record["date"], "snippets": snippets, "context": context},
                   paths.rag2_path(args.ticker, record["date"]))
        counts.append(len(snippets))
        sources.update(s["source"] for s in snippets)
        if args.show:
            print(f"===== {record['date']}（{len(snippets)} 段）\n{context}\n")
    print(f"[rag2] {args.ticker}: {len(files)} 天 -> {paths.OUTPUTS / 'rag2' / args.ticker}")
    print(f"[rag2] 每天平均 {sum(counts) / len(counts):.1f} 段（最少 {min(counts)}、最多 {max(counts)}）；"
          f"來源分布 {dict(sources)}")


if __name__ == "__main__":
    main()
