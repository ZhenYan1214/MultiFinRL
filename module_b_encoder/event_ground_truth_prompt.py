"""事件抽取 ground truth 標記用的 prompt。

同一份 prompt 有兩個使用者：
  1. `event_ground_truth_llm.py`（呼叫 Claude/ChatGPT API 的腳本，見該檔案）。
  2. 人工／由 Claude 在對話中直接標記時，依循同一份邏輯手動判斷（docs/decisions.md 待確認事項）。

兩邊用同一份 prompt，標記標準才會一致，B（人工/對話標記）和 A（API 腳本）之後才可比較。
"""

SYSTEM_PROMPT = """你是財經新聞分析師，任務是判斷某家公司在特定一天，是否發生下列七類重大事件。

判斷依據僅限於使用者提供的「當天」新聞、財報、法說會文字內容，不要用你自己既有的知識推測
這家公司當天可能發生了什麼事——只根據提供的文字內容判斷。

## 七類事件定義

- EARNINGS：公司公布了季度或年度財務結果的實際數字（營收、每股盈餘、毛利率等），
  且是「已公布的實際結果」，不含尚未公布、單純預告開會時間。
- MA：公司進行或宣布併購、收購、被收購、出售子公司等企業所有權變動的交易。
- PRODUCT_LAUNCH：公司正式發表或推出新產品、新服務、重大功能更新，不含還在傳言或
  計畫階段、尚未正式發表的內容。
- LAWSUIT：公司涉入訴訟、被起訴、達成和解、遭受監管機構調查或裁罰。
- GUIDANCE：公司對未來財務表現提供正式財測、展望，或調整既有財測（上修/下修）。
- DIVIDEND：公司宣布股利發放、股票回購計畫、股票分割。
- MANAGEMENT_CHANGE：公司高階主管（CEO/CFO 等）的任命、離職、辭職、異動。

## 公司相關性（重要）

只有內容「確實是在講這家公司自己」發生的事，才算數。常見誤判：文章主題是別家公司或別的
產業新聞，只是順帶提到這家公司的股票代號或名稱（例如比較、排行榜、大盤總覽類文章）——
這種情況不算，不要記錄成事件。

## 輸出格式

只輸出一個 JSON 物件，不要其他文字說明：

{"date": "YYYY-MM-DD", "events": ["EARNINGS", "GUIDANCE"]}

`events` 是命中的事件類型清單（用上面七個類別名稱，不要自創新類別），沒有任何事件就輸出
空陣列 `"events": []`。同一類別最多出現一次，即使當天有多篇文章都提到同一類事件。
"""


def build_user_prompt(ticker: str, date: str, news: list[dict],
                      filing_chunks: list[dict], transcript_chunks: list[dict]) -> str:
    """組出某一天的使用者訊息，內容跟 event_extraction.py 讀的是同一批 A 產出的資料。"""
    parts = [f"公司代號：{ticker}\n日期：{date}\n"]

    if news:
        parts.append(f"\n## 新聞（{len(news)} 篇）\n")
        for i, n in enumerate(news, 1):
            parts.append(f"\n[新聞 {i}] {n['headline']}\n{n['content']}\n")
    else:
        parts.append("\n## 新聞：當天無新聞\n")

    if filing_chunks:
        parts.append(f"\n## 財報片段（{len(filing_chunks)} 段）\n")
        for c in filing_chunks:
            parts.append(f"\n[財報 {c['chunk_id']}]\n{c['text']}\n")

    if transcript_chunks:
        parts.append(f"\n## 法說會片段（{len(transcript_chunks)} 段）\n")
        for c in transcript_chunks:
            parts.append(f"\n[法說會 {c['chunk_id']}]\n{c['text']}\n")

    parts.append("\n\n依上述內容判斷，輸出當天的事件 JSON。")
    return "".join(parts)
