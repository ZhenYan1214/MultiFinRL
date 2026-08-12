"""共用的 LLM 呼叫工具，給需要用 LLM 判斷事件類型的地方共用：

    event_ground_truth_llm.py  建立事件抽取的 ground truth 答案卷。
    event_extraction.py        --method llm 時的正式事件抽取。

兩邊用同一份 prompt（event_ground_truth_prompt.py）跟同一套呼叫邏輯，判斷標準才會一致，
兩者的結果才能互相比較（docs/spec_b_event_extraction_llm.md）。

支援三種服務商，用 --provider 切換：
    claude    Anthropic API（需要 ANTHROPIC_API_KEY）
    openai    OpenAI API（需要 OPENAI_API_KEY）
    deepseek  DeepSeek API（需要 DEEPSEEK_API_KEY）；介面跟 OpenAI 相容，
              直接用 openai 套件、換一個 base_url 就能打，不用額外裝 SDK。
"""
import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 沒裝 python-dotenv 也沒關係，退回用系統環境變數

from module_b_encoder.event_ground_truth_prompt import SYSTEM_PROMPT, build_user_prompt

EVENT_TYPES = {"EARNINGS", "MA", "PRODUCT_LAUNCH", "LAWSUIT",
               "GUIDANCE", "DIVIDEND", "MANAGEMENT_CHANGE"}

DEFAULT_MODEL = {
    "claude": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
}


def call_claude(system: str, user: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model, max_tokens=200, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def call_openai(system: str, user: str, model: str) -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model, max_tokens=200,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def call_deepseek(system: str, user: str, model: str) -> str:
    """DeepSeek 的 API 是 OpenAI 相容介面，用 openai 套件換 base_url 打過去即可，
    不需要 deepseek 自己的 SDK。"""
    import openai
    client = openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model=model, max_tokens=200,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


CALLERS = {"claude": call_claude, "openai": call_openai, "deepseek": call_deepseek}


def parse_events(raw: str) -> list[str]:
    """從 LLM 回應解析事件清單，過濾掉不在七類定義內的雜訊。"""
    start, end = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[start:end + 1])
    events = [e for e in obj.get("events", []) if e in EVENT_TYPES]
    return sorted(set(events))


def label_events(ticker: str, date: str, news: list[dict],
                 filing_chunks: list[dict], transcript_chunks: list[dict],
                 provider: str, model: str) -> list[str]:
    """呼叫指定服務商，用共用 prompt 判斷某一天發生了哪些事件類型。

    news/filing_chunks/transcript_chunks 由呼叫端組好、篩過——要不要先過濾公司相關性、
    要不要濾掉沿用中的舊資料，兩邊需求不同（ground truth 標記要看完整內容；正式抽取
    為了省成本會先篩過），這支函式只負責組 prompt、呼叫、解析，不做篩選。
    """
    user_prompt = build_user_prompt(ticker, date, news, filing_chunks, transcript_chunks)
    raw = CALLERS[provider](SYSTEM_PROMPT, user_prompt, model)
    return parse_events(raw)
