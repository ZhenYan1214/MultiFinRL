"""LLM 相關性打分（L_ground 的 α* 來源）用的 prompt 與輸出 schema。

設計重點（討論過程見 docs/decisions.md #100）：
- 打分對象是 H_r 當天檢索到的 K 段 chunk（財報／法說會片段），新聞只當背景，不打分。
- 新聞給全文：LLM 能看到模型 H_t 看不到的內容（H_t 截在 512 token），這部分 α* 對模型而言
  是學不到的雜訊，但 L_ground 只要求排序 K 段 chunk，影響有限；H_t 截斷問題另外處理。
- 只打分、不寫敘述：避免「先寫敘述再替自己找依據」的事後合理化，Haiku 成本也低到可以分開呼叫。
- chunk 每天隨機排列後標成 A/B/C，不給檢索分數與名次；代號本身仍有先後，但隨機排序讓位置偏誤
  變成與相關性無關的雜訊。排列由 ticker+日期 決定（固定 seed），需要時可用 build_request 重現。
- 每段附文件類型與發布日期，讓 LLM 自行判斷是否過時。
- 指令用英文（與新聞、文件同語言，打分較穩定），理由用繁體中文方便人工抽查。
- 絕不放入 daily record 的 prices / label（含未來收盤價）。
"""

# Prompt 修改紀錄（不寫進答案卷；改 prompt 後要重打請加 --overwrite）：
# v2：明確定義「分析日」，禁止使用分析日之後的知識；v3：代號改 A/B/C；
# v4：指令改英文（新聞、財報、法說會都是英文），理由仍以繁體中文輸出。實測 AAPL 4 天 × 6 次，
#     同一 prompt 重複打分的平均標準差：全中文 0.22、全英文 0.12、英文指令＋中文理由 0.12。

# chunk 隨機排序後給 LLM 看的代號；同時是輸出 JSON 的欄位名。
# 支援 K ≤ 8（目前 rag.top_k=3）。
DISPLAY_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

DOC_TYPE_NAMES = {
    "10-K": "10-K annual report",
    "10-Q": "10-Q quarterly report",
    "8-K": "8-K current report",
    "earnings_call": "Earnings call transcript",
    "macro_indicator": "Macroeconomic indicator",
    "macro_news": "Macro / FOMC news",
    "unknown": "Document",
}

SYSTEM_PROMPT = """You are a financial analyst reviewing a historical trading day. The user message gives the
"analysis date", the company's news on that date, and several document excerpts retrieved from the
company's filings and earnings calls. Your task: given the news on the analysis date, rate how useful
each excerpt is for judging the company's situation and risks as of the analysis date.

## Time reference (important)
- "That day", "outdated", and "days before the analysis date" always refer to the analysis date given
  by the user, not the real current date.
- Put yourself at the close of the analysis date: use only information public on or before it.
- Do not use anything you know that happened after the analysis date (later price moves, later
  filings or news).

## Scale (0-3)
- 3: directly relevant; explains, corroborates, or adds to important news of the analysis date
- 2: useful; background that is still valid and related to the situation on the analysis date
- 1: loosely related; limited help for judging the analysis date
- 0: not useful (outdated as of the analysis date, routine content, greetings or procedural text, unrelated)

## Notes
- Sharing a topic or wording with the news does not make an excerpt useful; judge whether it adds real information.
- Each excerpt shows its publication date and days before the analysis date; consider whether it was
  already outdated on the analysis date (e.g., old guidance superseded by a newer filing).
- Excerpts are cut from full documents and may start or end mid-sentence; this does not affect the rating.
- The excerpts are in random order, unrelated to importance.
- Rate each excerpt independently; giving every excerpt 0 is fine when none is useful.
- Write one short reason first, then the score. Write the reason in Traditional Chinese (繁體中文).
"""


def build_user_prompt(ticker: str, date: str, news: list[dict],
                      docs: list[dict]) -> str:
    """docs：已依顯示順序排好的 [{"label", "doc_type", "doc_date", "age_days", "text"}]。"""
    parts = [f"Ticker: {ticker}\nAnalysis date: {date}\n"]
    if news:
        parts.append(f"\n## News on the analysis date ({len(news)} articles; context only, do not rate)\n")
        for i, n in enumerate(news, 1):
            published = n.get("published_at", "")
            when = f" (published {published[:16].replace('T', ' ')})" if published else ""
            parts.append(f"\n[News {i}]{when} {n['headline']}\n{n['content']}\n")
    else:
        parts.append("\n## News on the analysis date: none\n")

    parts.append(f"\n## Excerpts to rate ({len(docs)}, randomly ordered)\n")
    for d in docs:
        dated = (f"published {d['doc_date']} ({d['age_days']} days before the analysis date)"
                 if d["doc_date"] else "publication date unknown")
        type_name = DOC_TYPE_NAMES.get(d["doc_type"], d["doc_type"])
        parts.append(f"\n[Document {d['label']}] {type_name} | {dated}\n{d['text']}\n")

    labels = ", ".join(f"Document {d['label']}" for d in docs)
    parts.append(f"\n\nFor each of {labels}, give one reason (in Traditional Chinese) and a 0-3 score.")
    return "".join(parts)


def output_schema(labels: list[str]) -> dict:
    """structured outputs 的 JSON schema：每個代號一個 {reason, score}。"""
    item = {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "score": {"type": "integer", "enum": [0, 1, 2, 3]},
        },
        "required": ["reason", "score"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {label: item for label in labels},
        "required": list(labels),
        "additionalProperties": False,
    }
