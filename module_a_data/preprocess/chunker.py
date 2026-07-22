"""把財報 / 法說會文件切成 Chunk。

上限：512 token（以 FinBERT tokenizer 計，configs/config.yaml 的 text.max_chunk_tokens）。
chunk_id 命名規則：{TICKER}_{DOC_TYPE}_{DATE}_{序號三位數}，見 samples/DataStruct.example.json。

若本機尚未安裝 transformers（A 不一定需要跑模型），fallback 用「字數 x 1.3」估 token 數，
正式產出前建議安裝 transformers 以精確計數。
"""
try:
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    def count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text, add_special_tokens=True))
except Exception:  # transformers 未安裝或下載失敗
    _tokenizer = None

    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3) + 2  # 粗估；+2 為 [CLS][SEP]


def chunk_text(text: str, max_tokens: int = 512) -> list[str]:
    """以句子為單位貪婪合併，每個 chunk 不超過 max_tokens。"""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        candidate = f"{current} {s}".strip() if current else s
        if count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # 單句就超長：硬切
            while count_tokens(s) > max_tokens:
                words = s.split()
                head, s = " ".join(words[: max_tokens // 2]), " ".join(words[max_tokens // 2:])
                chunks.append(head)
            current = s
    if current:
        chunks.append(current)
    return chunks


def make_chunks(ticker: str, doc_type: str, date: str, text: str,
                max_tokens: int = 512, date_key: str = "filing_date") -> list[dict]:
    """切 Chunk 並套上 chunk_id，回傳 data_format.md 規定的 chunk dict 列表。

    doc_type: "10-K" / "10-Q" / "earnings_call" 等。
    date_key: filing_chunks 用 "filing_date"、transcript_chunks 用 "event_date"。
    """
    return [
        {"chunk_id": f"{ticker}_{doc_type}_{date}_{i:03d}",
         "doc_type": doc_type, date_key: date, "text": c}
        for i, c in enumerate(chunk_text(text, max_tokens))
    ]
