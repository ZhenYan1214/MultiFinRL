"""用 LLM API 生成 decoder 的訓練目標 y_belief（計畫書 3.5 節）。

y_belief 目前整個專案完全不存在（PDF 沒說怎麼取得，見 docs/decisions.md）。這支腳本
用既有的 module_b_encoder/llm_client.py 呼叫邏輯，仿照 event_ground_truth_llm.py 的
做法，LLM-bootstrap 出一份訓練目標——做法上跟 #32 用 LLM 標記事件 ground truth 是同一種
精神（先用 LLM 頂著跑通流程，之後有需要再人工校對，不是宣稱這就是完美答案卷）。

<TREND> 直接沿用 A 的 label，不讓 LLM 重猜（理由見 y_belief_prompt.py 檔頭說明）；
LLM 只生成 risk_level 與 narrative，兩者組成最終的 y_belief 文字。

輸出：data/labels/y_belief/{ticker}.json   {date: y_belief 文字}
（data/labels/ 不在 .gitignore 排除範圍內，答案卷會進版本控制，跟 event_ground_truth 一樣）

要啟用：
    1. pip install openai   （claude/deepseek 都走這個套件；用 claude 則另外 pip install anthropic）
    2. .env 已有 DEEPSEEK_API_KEY（沿用既有設定，不用重設）
    3. python -m module_c_fusion.decoder.generate_y_belief --ticker AAPL --limit 50

先用 --limit 50 跑一小批，人工看過幾筆 narrative 的品質、確認格式沒問題，再拿掉 --limit
跑全量。跟 event_ground_truth_llm.py 一樣支援中斷續跑：已經生成過的日期重跑時會直接跳過。
"""
import argparse
import json

from shared import paths
from shared.utils import load_config, read_json, write_json
from module_b_encoder.llm_client import CALLERS, DEFAULT_MODEL
from module_c_fusion.decoder.y_belief_prompt import SYSTEM_PROMPT, build_user_prompt

VALID_RISK_LEVELS = {"LOW_VOLATILITY", "HIGH_VOLATILITY"}


def render_y_belief(trend: str, risk_level: str, narrative: str) -> str:
    """組出最終的 y_belief 文字：<TREND>/<RISK_LEVEL> 特殊 token + 結構化標籤 + 敘述文字
    （計畫書 3.5 節 y_belief 定義：「structured tokens like [BULLISH]/[HIGH_VOLATILITY]
    plus narrative text, with special tokens <TREND>/<RISK_LEVEL> at sequence start」）。
    """
    return f"<TREND> [{trend}] <RISK_LEVEL> [{risk_level}] {narrative}"


def parse_response(raw: str) -> tuple[str, str]:
    """解析 LLM 回應；risk_level 不在合法值內就保守歸類成 LOW_VOLATILITY，不中斷整批流程。"""
    start, end = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[start:end + 1])
    risk_level = obj.get("risk_level", "LOW_VOLATILITY")
    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "LOW_VOLATILITY"
    narrative = obj.get("narrative", "").strip()
    return risk_level, narrative


def label_day(ticker: str, date: str, provider: str, model: str) -> str | None:
    """回傳 None 代表這天解析失敗（LLM 偶爾會回傳非預期格式，例如速率限制訊息），
    呼叫端跳過即可——這天不會寫進 y_belief，下次重跑同一指令會自動當成待生成補上
    （resume 邏輯是看 y_belief 裡有沒有這個日期，不是看有沒有嘗試過）。
    """
    record = read_json(paths.daily_json(ticker, date))
    user_prompt = build_user_prompt(ticker, date, record["news"],
                                    record["filing_chunks"], record["transcript_chunks"])
    raw = CALLERS[provider](SYSTEM_PROMPT, user_prompt, model)
    try:
        risk_level, narrative = parse_response(raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    [警告] {date} 解析失敗，跳過（{e}）：原始回應開頭 {raw[:120]!r}")
        return None
    return render_y_belief(record["label"], risk_level, narrative)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--provider", choices=["claude", "openai", "deepseek"], default="deepseek")
    ap.add_argument("--model", default=None, help="不指定就用 provider 的預設模型")
    ap.add_argument("--limit", type=int, default=None, help="只生成前 N 天（先小批測試品質用）")
    args = ap.parse_args()
    model = args.model or DEFAULT_MODEL[args.provider]

    manifest_path = paths.dataset_manifest(args.ticker)
    if not manifest_path.exists():
        raise SystemExit(f"找不到 A 的資料 manifest: {manifest_path}，先跑完 A 的 pipeline")
    manifest = read_json(manifest_path)
    files = [paths.daily_json(args.ticker, date) for date in manifest["dates"]]
    if not files or any(not file.exists() for file in files):
        raise SystemExit(f"A 的 manifest/每日資料不完整: {paths.DATASET / args.ticker}")

    out_path = paths.y_belief_path(args.ticker)
    meta_path = paths.y_belief_meta_path(args.ticker)
    y_belief: dict[str, str] = {}
    if out_path.exists():
        if not meta_path.exists():
            raise SystemExit(
                f"既有 {out_path} 缺少 dataset 指紋，無法確認標籤/文本版本。"
                "請先備份或移走舊檔，再依新版 dataset 重新生成"
            )
        meta = read_json(meta_path)
        if meta.get("dataset_records_sha256") != manifest["records_sha256"]:
            raise SystemExit(
                f"既有 {out_path} 對應舊 dataset。為避免混用舊 TREND/敘述，"
                "請先備份或移走舊檔，再重新生成"
            )
        if meta.get("provider") != args.provider or meta.get("model") != model:
            raise SystemExit(
                f"既有 {out_path} 由 {meta.get('provider')}/{meta.get('model')} 生成，"
                f"不可和 {args.provider}/{model} 混在同一份 target；請改回原設定或另行備份重建"
            )
        y_belief = read_json(out_path)  # 已生成的先保留，重跑只補新的（同 event_ground_truth_llm.py）
        valid_dates = set(manifest["dates"])
        y_belief = {date: text for date, text in y_belief.items() if date in valid_dates}

    targets = [f for f in files if f.stem not in y_belief]
    if args.limit:
        targets = targets[:args.limit]
    print(f"[generate_y_belief] {args.ticker}: 待生成 {len(targets)} 天"
         f"（已有 {len(y_belief)} 天），provider={args.provider}, model={model}")

    for i, f in enumerate(targets, 1):
        date = f.stem
        text = label_day(args.ticker, date, args.provider, model)
        if text is None:
            continue  # 這天解析失敗，不落地，下次重跑會自動補（見 label_day 說明）
        y_belief[date] = text
        print(f"  [{i}/{len(targets)}] {date}: {text[:80]}...")
        write_json(y_belief, out_path)  # 每天都落地，避免中途中斷（例如 API 偶發錯誤）把之前的進度全部弄丟
        write_json({"ticker": args.ticker, "provider": args.provider, "model": model,
                    "dataset_records_sha256": manifest["records_sha256"]}, meta_path)

    write_json(y_belief, out_path)
    write_json({"ticker": args.ticker, "provider": args.provider, "model": model,
                "dataset_records_sha256": manifest["records_sha256"]}, meta_path)
    print(f"[generate_y_belief] {len(y_belief)} 天 -> {out_path}")


if __name__ == "__main__":
    main()
