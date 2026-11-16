"""
RAG-2：Decoder 生成前，從分析日 t 可得的文件檢索出要放進 context 的段落（decisions.md #113、#132）。

候選直接從 A 的每日 JSON 取（build_dataset.py 已經依 available date 過濾，只含 t 當下可得的文件）：
當天新聞（days_ago = 0，盤後新聞已歸到下一個交易日）、當天與以前的 8-K、最新一份 10-K／10-Q、
最近一場法說會。不算 Loss，檢索器不訓練。

method: embed（主線，#132）
    1. 切候選：新聞一則一段（標題＋一句；第一句沒提到 Apple 時改取第一個提到 Apple 的句子）；
       8-K、10-K／10-Q、法說會依句子切成約 rag2.snippet_words 字的段落，跳過表格、目錄、
       forward-looking 免責聲明、頁首、附件清單、法說會開場串場。
    2. 打分數：凍結的嵌入模型（rag2.embed_model）把每段和固定 query 編碼，
       分數 = entity_weight × cos(段落, 實體 query) + (1 − entity_weight) × max cos(段落, 面向 query)。
       query 每天一樣；每天不同的資訊來自候選本身。以前的 8-K 分數再乘上 exp(−天數 / older_8k_decay_days)。
    3. 挑段落（select）：不同來源的分數尺度不同，不跨來源比分數。當天 8-K 每份必收分數最高的
       today_8k_per_doc 段；新聞、10-K／10-Q、法說會各自排名，依 quota 輪流放入，某來源不夠時由其他
       來源補位；以前的 8-K 最後補位。新聞分數低於 news_min_score 不收；受段數上限 max_snippets、
       token 上限 max_tokens 限制；跟已選段落 cosine 超過 dedupe_cos 的跳過（同一事件的多則新聞）。
    4. 輸出依來源排列：當天 8-K → 新聞（分數高到低）→ 10-K／10-Q → 法說會 → 以前的 8-K。

method: rule（舊版，對照用）
    當天 8-K 全收、當天新聞全收（標題＋第一句）、背景 10-K／10-Q 固定挑 MD&A 與風險因素開頭、
    法說會挑財務關鍵字最多的段落、以前的 8-K 補到上限；超過時從新聞尾端刪。不做相似度排序。

每段加上「[來源 日期]」標頭。輸出 data/outputs/rag2/{TICKER}/{date}.json：
    {"ticker", "date", "method", "snippets": [{"source", "doc_type", "date", "section", "chunk_id", "text",
     "group", "score"}], "n_candidates",
     "context": "[8-K 2024-05-02] ...\n[News 2024-05-02] ...\n..."}   # Decoder 直接讀 context

用法：
    python -m module_b_encoder.rag.rag2 --ticker AAPL
    python -m module_b_encoder.rag.rag2 --ticker AAPL --date 2024-05-02 --show
    python -m module_b_encoder.rag.rag2 --ticker AAPL --method rule        # 舊規則（對照）
"""
import argparse
import datetime as dt
import math
import re
from collections import Counter

import numpy as np

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
APPLE = re.compile(r"\b(Apple|AAPL|iPhone|iPad|Mac|MacBook|Tim Cook|App Store|Vision Pro|AirPods|"
                   r"Cupertino|iOS)\b")
BOILERPLATE = re.compile(r"forward-looking|Private Securities Litigation|assumes no obligation|"
                         r"Unless otherwise stated|should be read in conjunction|Table of Contents", re.I)
SOURCE_NAMES = {"10-K": "10-K", "10-Q": "10-Q", "8-K": "8-K", "earnings_call": "Earnings call"}
GROUP_ORDER = ["today_8k", "news", "periodic", "transcript", "older_8k"]
# method: embed 切段時額外跳過的段落：附件清單、無變動聲明、法說會開場與主持人串場
SKIP_WINDOW = re.compile(r"Incorporated by Reference|Exhibit (Number|Description)|no material changes|"
                         r"welcome to the|thank you for joining|Earnings Call Transcript|"
                         r"Operator \(Operator\)|SIGNATURES?\b|Pursuant to the requirements|/s/", re.I)
PAGE_HEADER = re.compile(r"Apple Inc\. \| (Q\d )?\d{4} Form 10-[KQ] \| \d+\s*")   # 10-K／10-Q 每頁頁首
SECTION_HEADER = re.compile(r"(PART [IV]+ — [A-Z ]+)|(Item \d+[A-Z]?\.\s*)$")


def shorten(text: str, n_words: int) -> str:
    words = re.sub(r"\s+", " ", TAG_PREFIX.sub("", text)).strip().split(" ")
    return " ".join(words[:n_words]) + (" …" if len(words) > n_words else "")


ABBREVIATIONS = {"inc", "corp", "co", "ltd", "mr", "ms", "dr", "vs", "st", "jr", "no", "u.s", "u.k", "e.g", "i.e"}


def split_sentences(text: str) -> list[str]:
    """切句；「U.S.」「Inc.」這類縮寫後面的句點不算句尾。"""
    text = re.sub(r"\s+", " ", text).strip()
    out, start = [], 0
    for m in re.finditer(r"[.!?](?=\s+[A-Z“\"(]|$)", text):
        word = re.findall(r"[\w.]+$", text[: m.start()])
        if word and (word[0].lower() in ABBREVIATIONS or (len(word[0]) == 1 and word[0].isalpha())
                     or "." in word[0]):
            continue
        out.append(text[start: m.end()].strip())
        start = m.end()
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def first_sentence(text: str) -> str:
    sentences = split_sentences(text)
    return sentences[0] if sentences else ""


def skip_boilerplate(text: str) -> str:
    """章節開頭常是 forward-looking 免責聲明，跳過這類句子，從第一句實質內容開始。"""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", text).strip())
    for i, sentence in enumerate(sentences[:12]):
        if not BOILERPLATE.search(sentence):
            return " ".join(sentences[i:])
    return text


def _available(chunk: dict) -> str:
    """可得交易日：盤後送件的 8-K 從下一個交易日才算「當天」的新文件（build_dataset.py）。
    舊資料沒有 available_date 時退回 filing_date。"""
    return chunk.get("available_date", chunk["filing_date"])


def _doc_key(chunk_id: str) -> str:
    return chunk_id.rsplit("_", 1)[0]


def _days_between(a: str, b: str) -> int:
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def _item(source, doc_type, doc_date, chunk_id, text, section=None, group=None, score=None):
    return {"source": source, "doc_type": doc_type, "date": doc_date, "section": section,
            "chunk_id": chunk_id, "text": text, "group": group, "score": score}


# ---------------------------------------------------------------- method: rule（舊版）

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


def _group_docs(filings: list[dict]) -> dict[str, list[dict]]:
    docs: dict[str, list[dict]] = {}
    for c in filings:
        docs.setdefault(_doc_key(c["chunk_id"]), []).append(c)
    return docs


def build_snippets_rule(record: dict, cfg: dict) -> list[dict]:
    rc = cfg["rag2"]
    n_words, limit = rc["snippet_words"], rc["max_snippets"]
    date = record["date"]
    docs = _group_docs(record["filing_chunks"])
    eight_k = sorted((k for k, v in docs.items() if v[0]["doc_type"] == "8-K"),
                     key=lambda k: docs[k][0]["filing_date"], reverse=True)

    today_8k, older_8k = [], []
    for key in eight_k:
        chunks = docs[key]
        filing_date = chunks[0]["filing_date"]
        is_today = _available(chunks[0]) == date
        target = today_8k if is_today else older_8k
        for s in _eight_k_snippets(chunks, n_words, 2 if is_today else 1):
            target.append(_item("8-K", "8-K", filing_date, s["chunk_id"], s["text"]))

    news = [_item("News", "news", n["published_at"][:10], None,
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
                background.append(_item(c["doc_type"], c["doc_type"], c["filing_date"], c["chunk_id"],
                                        shorten(skip_boilerplate(body), n_words), SECTION_NAMES[section]))

    transcript = []
    tx = record["transcript_chunks"]
    if tx:
        latest = max(c["event_date"] for c in tx)
        call = [c for c in tx if c["event_date"] == latest]
        scored = sorted(range(len(call)), key=lambda i: -len(TRANSCRIPT_KEYWORDS.findall(call[i]["text"])))
        for i in sorted(scored[: rc["transcript_snippets"]]):
            c = call[i]
            transcript.append(_item("Earnings call", "earnings_call", c["event_date"], c["chunk_id"],
                                    shorten(c["text"], n_words)))

    fixed = today_8k + background + transcript
    room = max(0, limit - len(fixed))
    snippets = today_8k + news[:room] + background + transcript
    room = max(0, limit - len(snippets))
    snippets += older_8k[:room]
    return snippets[:limit]


# ---------------------------------------------------------------- method: embed（主線）

def _is_table_like(text: str) -> bool:
    """財報表格、目錄：數字與符號比例高，或連續出現多個 Item 標題。"""
    tokens = text.split()
    if not tokens:
        return True
    numeric = sum(bool(re.fullmatch(r"[\d$,.()%—–-]+", t)) for t in tokens)
    return numeric / len(tokens) > 0.3 or len(re.findall(r"\bItem\s+\d", text)) >= 2


def window_text(text: str, n_words: int, min_words: int = 12) -> list[str]:
    """依句子切成約 n_words 字的段落；太短、像表格、免責聲明、附件清單、開場串場的段落跳過。"""
    out, buf = [], []
    text = PAGE_HEADER.sub("", TAG_PREFIX.sub("", text))
    for sentence in split_sentences(text):
        if SECTION_HEADER.fullmatch(sentence.strip()):
            continue
        buf.append(sentence)
        if sum(len(s.split()) for s in buf) >= int(n_words * 0.75):
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    keep = []
    for w in out:
        if (len(w.split()) < min_words or _is_table_like(w) or BOILERPLATE.search(w)
                or SKIP_WINDOW.search(w)):
            continue
        keep.append(shorten(w, n_words))
    return keep


def news_snippet(n: dict, n_words: int) -> str:
    """標題＋一句：第一句沒提到 Apple 時，改取內文第一個提到 Apple 的句子（同業、大盤新聞常見）。"""
    sentences = split_sentences(n["content"])
    pick = sentences[0] if sentences else ""
    if not APPLE.search(pick):
        pick = next((s for s in sentences if APPLE.search(s)), pick)
    return shorten(f"{n['headline']}. {pick}", n_words)


def build_candidates(record: dict, cfg: dict) -> list[dict]:
    """當天所有候選段落（還沒打分數）。"""
    n_words = cfg["rag2"]["snippet_words"]
    date = record["date"]
    cands = []
    for n in record["news"]:
        if int(n["days_ago"]) == 0:
            cands.append(_item("News", "news", n["published_at"][:10], None, news_snippet(n, n_words),
                               group="news"))
    for chunks in _group_docs(record["filing_chunks"]).values():
        doc_type, filing_date = chunks[0]["doc_type"], chunks[0]["filing_date"]
        if doc_type == "8-K":
            group = "today_8k" if _available(chunks[0]) == date else "older_8k"
        else:
            group = "periodic"
        for c in chunks:
            for w in window_text(c["text"], n_words):
                cands.append(_item(SOURCE_NAMES.get(doc_type, doc_type), doc_type, filing_date,
                                   c["chunk_id"], w, group=group))
    tx = record["transcript_chunks"]
    if tx:
        latest = max(c["event_date"] for c in tx)
        for c in tx:
            if c["event_date"] == latest:
                for w in window_text(c["text"], n_words):
                    cands.append(_item("Earnings call", "earnings_call", c["event_date"], c["chunk_id"], w,
                                       group="transcript"))
    return cands


class Embedder:
    """凍結的嵌入模型（bge 系列：CLS pooling＋L2 正規化；query 加檢索指令）。同一段文字只編碼一次。"""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.cache: dict[str, np.ndarray] = {}

    def _encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=128,
                         return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                h = self.model(**b).last_hidden_state[:, 0]
            out.append(self.torch.nn.functional.normalize(h, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size), np.float32)

    def passages(self, texts: list[str]) -> np.ndarray:
        new = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for t, v in zip(new, self._encode(new)):
            self.cache[t] = v
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.QUERY_PREFIX + t for t in texts])


def score_candidates(cands: list[dict], emb: np.ndarray, q_entity: np.ndarray, q_aspects: np.ndarray,
                     rc: dict, date: str) -> np.ndarray:
    w = rc["entity_weight"]
    scores = w * (emb @ q_entity) + (1 - w) * (emb @ q_aspects.T).max(axis=1)
    for i, c in enumerate(cands):
        if c["group"] == "older_8k":
            scores[i] *= math.exp(-_days_between(c["date"], date) / rc["older_8k_decay_days"])
    return scores


def render_line(s: dict) -> str:
    head = f"{s['source']} {s['date']}" + (f" {s['section']}" if s.get("section") else "")
    return f"[{head}] {s['text']}"


def select(cands: list[dict], emb: np.ndarray, scores: np.ndarray, rc: dict, count_tokens) -> list[dict]:
    """挑段落。不同來源的原始分數尺度不同（新聞寫法常直接點名 Apple，分數普遍較高），所以不跨來源比分數，
    而是每個來源各自依分數排名、再按名額輪流放入：
        1. 當天 8-K 每份必收分數最高的 today_8k_per_doc 段；
        2. 新聞、10-K／10-Q、法說會依 quota 輪流取各自的第 1 名、第 2 名……（新聞原始分數低於
           news_min_score 的不收）；
        3. 某個來源候選不夠時，剩下的位置由其他來源依同樣的輪流順序補上；
        4. 以前的 8-K 最後補位（最多 older_8k_max 段）。
    全程受段數上限 max_snippets、token 上限 max_tokens 限制；跟已選段落 cosine 超過 dedupe_cos 的跳過。"""
    chosen: list[int] = []
    per_group: Counter = Counter()
    used_tokens = 0

    def try_add(i: int) -> bool:
        nonlocal used_tokens
        if i in chosen or len(chosen) >= rc["max_snippets"]:
            return False
        if chosen and float((emb[chosen] @ emb[i]).max()) > rc["dedupe_cos"]:
            return False
        n = count_tokens(render_line(cands[i]) + "\n")
        if used_tokens + n > rc["max_tokens"]:
            return False
        chosen.append(i)
        per_group[cands[i]["group"]] += 1
        used_tokens += n
        return True

    ranked: dict[str, list[int]] = {}
    for i in sorted(range(len(cands)), key=lambda i: -scores[i]):
        c = cands[i]
        if c["group"] == "news" and scores[i] < rc["news_min_score"]:
            continue
        ranked.setdefault(c["group"], []).append(i)

    # 1. 當天 8-K
    taken: Counter = Counter()
    for i in ranked.get("today_8k", []):
        key = _doc_key(cands[i]["chunk_id"])
        if taken[key] < rc["today_8k_per_doc"] and try_add(i):
            taken[key] += 1

    # 2–3. 依名額輪流，名額用完後再無上限地輪流補位
    quota = rc["quota"]
    queues = {g: list(ranked.get(g, [])) for g in quota}
    for limited in (True, False):
        progress = True
        while progress and len(chosen) < rc["max_snippets"]:
            progress = False
            for g in quota:
                if limited and per_group[g] >= quota[g]:
                    continue
                while queues[g]:
                    if try_add(queues[g].pop(0)):
                        progress = True
                        break

    # 4. 以前的 8-K
    for i in ranked.get("older_8k", []):
        if per_group["older_8k"] >= rc["older_8k_max"]:
            break
        try_add(i)

    rank = {g: k for k, g in enumerate(GROUP_ORDER)}
    chosen.sort(key=lambda i: (rank[cands[i]["group"]],
                               -scores[i] if cands[i]["group"] == "news" else 0,
                               cands[i]["date"], cands[i]["chunk_id"] or ""))
    out = []
    for i in chosen:
        s = dict(cands[i])
        s["score"] = round(float(scores[i]), 4)
        out.append(s)
    return out


class EmbedRetriever:
    def __init__(self, cfg: dict):
        from transformers import AutoTokenizer
        self.cfg = cfg
        self.rc = cfg["rag2"]
        self.embedder = Embedder(self.rc["embed_model"])
        self.q_entity = self.embedder.queries([self.rc["query_entity"]])[0]
        self.q_aspects = self.embedder.queries(self.rc["query_aspects"])
        name = cfg.get("decoder", {}).get("base_model", "meta-llama/Llama-2-7b-hf")
        try:   # LLaMA-2 是 gated repo：先用本機快取，沒有才連網（需要 hf auth login）
            llm_tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        except OSError:
            llm_tok = AutoTokenizer.from_pretrained(name)
        self.count_tokens = lambda text: len(llm_tok(text, add_special_tokens=False)["input_ids"])

    def __call__(self, record: dict) -> tuple[list[dict], int]:
        cands = build_candidates(record, self.cfg)
        if not cands:
            return [], 0
        emb = self.embedder.passages([c["text"] for c in cands])
        scores = score_candidates(cands, emb, self.q_entity, self.q_aspects, self.rc, record["date"])
        return select(cands, emb, scores, self.rc, self.count_tokens), len(cands)


# ---------------------------------------------------------------- 共用

def render(snippets: list[dict]) -> str:
    """Decoder context 用的文字：每段一行，「[來源 日期 章節] 短版文字」。"""
    return "\n".join(render_line(s) for s in snippets)


def load_context(ticker: str, date: str) -> str:
    path = paths.rag2_path(ticker, date)
    if not path.exists():
        raise SystemExit(f"找不到 RAG-2 結果 {path}，先跑 python -m module_b_encoder.rag.rag2 --ticker {ticker}")
    return read_json(path)["context"]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--method", choices=["embed", "rule"], default=cfg["rag2"].get("method", "embed"))
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

    retriever = EmbedRetriever(cfg) if args.method == "embed" else None
    counts, sources = [], Counter()
    for f in files:
        record = read_json(f)
        if retriever is not None:
            snippets, n_cand = retriever(record)
        else:
            snippets, n_cand = build_snippets_rule(record, cfg), None
        context = render(snippets)
        write_json({"ticker": args.ticker, "date": record["date"], "method": args.method,
                    "n_candidates": n_cand, "snippets": snippets, "context": context},
                   paths.rag2_path(args.ticker, record["date"]))
        counts.append(len(snippets))
        sources.update(s["source"] for s in snippets)
        if args.show:
            print(f"===== {record['date']}（{len(snippets)} 段，候選 {n_cand}）\n{context}\n")
    print(f"[rag2] {args.ticker}（{args.method}）: {len(files)} 天 -> {paths.OUTPUTS / 'rag2' / args.ticker}")
    print(f"[rag2] 每天平均 {sum(counts) / len(counts):.1f} 段（最少 {min(counts)}、最多 {max(counts)}）；"
          f"來源分布 {dict(sources)}")


if __name__ == "__main__":
    main()
