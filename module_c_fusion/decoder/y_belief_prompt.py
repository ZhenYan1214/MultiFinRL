"""生成 y_belief（decoder 訓練目標）用的 prompt。

y_belief 是計畫書 3.5 節定義的訓練目標：結構化 token（如 [BULLISH]/[HIGH_VOLATILITY]）
加上敘述文字，序列開頭有 <TREND>/<RISK_LEVEL> 兩個特殊 token。

<TREND> 直接沿用 A 模組已經算好的 label（BULLISH/BEARISH/NEUTRAL，用未來 5 個交易日
報酬分位數嚴謹算出來的，見 configs/config.yaml 的 label 區塊），不讓 LLM 重新用文字內容
猜——LLM 光看新聞/財報文字猜漲跌，準確度遠不如 A 用實際未來報酬算出來的門檻，重新猜
只會是雜訊來源。

LLM 只負責兩件事：
    1. risk_level：這天的內容讀起來風險/波動性高不高（LOW_VOLATILITY / HIGH_VOLATILITY）。
    2. narrative：2-4 句話，總結當天新聞/財報/法說會揭露的重點，說明風險判斷的依據。

跟 event_ground_truth_prompt.py 是同一種性質（LLM-bootstrap 訓練/驗證用的答案卷），
用法上互相獨立，不共用同一份 prompt（任務不同：那邊判斷事件類型，這邊判斷風險與敘述）。
"""

SYSTEM_PROMPT = """你是財經分析師，任務是閱讀某家公司「當天」的新聞、財報片段、法說會片段，
判斷這天的市場風險程度，並寫一段簡短的市場敘述。

## risk_level 判斷標準

- HIGH_VOLATILITY：內容涉及重大不確定性，例如財測下修、訴訟、管理層異動、併購、
  營運指標大幅不如預期、產業性系統風險等，讀起來這天市場情緒可能劇烈波動。
- LOW_VOLATILITY：內容平淡、例行性揭露，沒有重大意外或不確定性。

## narrative 要求

2-4 句話，純粹總結「當天內容揭露了什麼」，不要預測股價漲跌方向（方向已經由其他資料
決定，不是你的任務），只描述事實與其隱含的風險程度。

## 輸出格式

只輸出一個 JSON 物件，不要其他文字說明：

{"risk_level": "HIGH_VOLATILITY", "narrative": "……"}
"""


def build_user_prompt(ticker: str, date: str, news: list[dict],
                      filing_chunks: list[dict], transcript_chunks: list[dict]) -> str:
    """組法跟 event_ground_truth_prompt.py 的 build_user_prompt 一致，確保兩邊看到的是
    同一批 A 產出的資料，只是任務不同。
    """
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

    parts.append("\n\n依上述內容判斷 risk_level，並寫出 narrative，輸出 JSON。")
    return "".join(parts)
