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
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 沒裝 python-dotenv 也沒關係，退回用系統環境變數

from module_b_encoder.event_ground_truth_prompt import SYSTEM_PROMPT, build_user_prompt

EVENT_TYPES = {"EARNINGS", "MA", "PRODUCT_LAUNCH", "LAWSUIT",
               "GUIDANCE", "DIVIDEND", "MANAGEMENT_CHANGE"}

# 累計 Claude 用掉的 token，跑完可以估實際花費（多執行緒同時呼叫，要上鎖）
USAGE = {"input_tokens": 0, "output_tokens": 0}
_usage_lock = threading.Lock()

DEFAULT_MODEL = {
    "claude": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
}

# 事件標記用的輸出格式：Claude 走 structured outputs，保證回傳的 JSON 一定解析得了、
# 類別一定在七類裡面（openai/deepseek 仍靠 prompt 要求 + parse_events 過濾）
EVENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "events": {"type": "array", "items": {"type": "string", "enum": sorted(EVENT_TYPES)}},
    },
    "required": ["date", "events"],
    "additionalProperties": False,
}

_claude_client = None


def call_claude(system: str, user: str, model: str, schema: dict | None = None) -> str:
    """Haiku 5.5 預設會先思考（adaptive thinking），回應的第一個區塊可能是 thinking 不是 text，
    所以要挑出 text 區塊；max_tokens 也要留給思考用，200 會被截斷。"""
    global _claude_client
    import anthropic
    if _claude_client is None:  # client 可以跨執行緒共用，建一次就好
        _claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=5)
    kwargs = {"output_config": {"format": {"type": "json_schema", "schema": schema}}} if schema else {}
    resp = _claude_client.messages.create(
        model=model, max_tokens=16000, system=system,
        messages=[{"role": "user", "content": user}], **kwargs,
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"Claude 回應異常結束: stop_reason={resp.stop_reason}")
    with _usage_lock:
        USAGE["input_tokens"] += resp.usage.input_tokens
        USAGE["output_tokens"] += resp.usage.output_tokens
    return "".join(b.text for b in resp.content if b.type == "text")


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
    if provider == "claude":
        raw = call_claude(SYSTEM_PROMPT, user_prompt, model, schema=EVENTS_SCHEMA)
    else:
        raw = CALLERS[provider](SYSTEM_PROMPT, user_prompt, model)
    return parse_events(raw)
