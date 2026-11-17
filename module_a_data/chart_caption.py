"""K 線文字敘述（A 模組）：Claude Haiku 讀當天 K 線圖＋成交量圖，寫成三段英文敘述（decisions.md #109）。

每天產生一次就存檔，之後訓練與推論都讀存檔（結果可重現）。敘述經凍結的 FinBERT 編碼成 H_t
（B 的 module_b_encoder/generate_vectors.py），是 L_align 與 L_ground 的文字端。

輸出 data/labels/chart_caption/{TICKER}.json（答案卷，進版本控制）：
    {date: {
        "pattern": "BREAKOUT",                 # shared/patterns.PATTERN_IDS 七選一（Haiku 判斷）
        "volume_label": "SPIKE",               # 爆量／放量／縮量／量平 四選一（OHLCV 公式決定）
        "kline": "...", "volume": "...", "combined": "...",   # 對應 [K線]、[成交量]、[一起說明]
        "volume_ratios": {"ratio_5_15": 1.42, "recent_max_ratio_20": 2.3},   # 公式算出的倍數，只存檔
        "model": "claude-haiku-5-5"}}

- prompt 只給兩張圖與型態清單，不給股票代號、日期、未來價格或 label（K 線圖本身也關掉座標軸），
  避免 Haiku 憑記憶知道之後的漲跌。
- 成交量標籤由 OHLCV 公式決定（門檻在 shared/patterns.py，待調），不讓 Haiku 判斷；
  prompt 只把狀況告訴 Haiku（例如「最近幾天的量明顯比前面高」），不給倍數或門檻，
  Haiku 照這個狀況寫〔成交量〕那段，標籤跟文字才會一致（decisions.md #128）。

用法：
    python -m module_a_data.chart_caption --ticker AAPL --limit 5          # 先抽幾天看
    python -m module_a_data.chart_caption --ticker AAPL --workers 8        # 全量（可中斷續跑）
    python -m module_a_data.chart_caption --ticker AAPL --check            # 不呼叫 API，只印型態與成交量分布
"""
import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from shared import paths, patterns
from shared.utils import load_config, read_json, write_json

SYSTEM_PROMPT = """You are a technical analyst. You will see two charts of the same stock over the \
same 20 trading days: first a candlestick chart, then a volume chart. The rightmost bar in both \
charts is the most recent trading day. Green means the close was at or above the open, red means \
it was below. The axes are hidden on purpose, so describe levels relatively (e.g. "near the \
window high", "about halfway up the range").

Classify the price action of the 20-day window into exactly one pattern. Check in this order \
and take the first that applies:
1. BREAKOUT: in the last 1-3 days price closes clearly above the highs of the earlier part of the \
window (it leaves the prior range upward). This applies even if a pullback came before it.
2. BREAKDOWN: in the last 1-3 days price closes clearly below the lows of the earlier part of the \
window (it leaves the prior range downward). This applies even if a rally came before it.
3. UPTREND: prices rise across most of the window and the latest close is in the upper part of \
the range. Pullbacks inside the rise do not change this.
4. DOWNTREND: prices fall across most of the window and the latest close is in the lower part of \
the range. Bounces inside the fall do not change this.
5. REVERSAL_TOP: a clear rise lasting several days is followed by a fall that also lasts several \
days, gives back at least about half of the rise, and leaves the latest close in the lower half \
of the range (e.g. an inverted V, a double top, a head and shoulders top).
6. REVERSAL_BOTTOM: a clear fall lasting several days is followed by a rise that also lasts \
several days, recovers at least about half of the fall, and leaves the latest close in the upper \
half of the range (e.g. a V, a double bottom, a head and shoulders bottom).
7. CONSOLIDATION: prices move sideways in a range with no clear direction.
A small pullback after a rise, or a small bounce after a fall, is not a reversal.
In the kline section, name the specific shape when it is visible (e.g. "a double bottom", "a head \
and shoulders top").

You will also be told the volume state of this window, measured from the volume data. Write the \
volume section so that it agrees with that state, and describe what the volume chart shows.

Then write three short sections in English, 1-2 sentences each:
- kline: the price path and the pattern, with position details (where the highs/lows are, \
where the latest close sits in the range).
- volume: how volume changes, which days stand out, and how that lines up with up/down days.
- combined: what price and volume together suggest about the pattern (e.g. "rising price on \
rising volume confirms the breakout").
Do not guess the company, the dates, or future prices. Describe only what the charts show."""

USER_PROMPT = "Volume state of this window: {volume_state}\n\nAnalyze these two charts and return the JSON."

# 告訴 Haiku 的成交量狀況：只講狀況，不給倍數或門檻（decisions.md #128、#129）
VOLUME_STATE_TEXT = {
    "SPIKE": "a volume spike: one of the last three days traded far above the normal level of this window.",
    "EXPANDING": "expanding volume: volume over the most recent days is clearly higher than earlier in the window.",
    "CONTRACTING": "contracting volume: volume over the most recent days is clearly lower than earlier in the window.",
    "FLAT": "flat volume: volume over the most recent days is about the same as earlier in the window.",
}

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "enum": patterns.PATTERN_IDS},
        "kline": {"type": "string"},
        "volume": {"type": "string"},
        "combined": {"type": "string"},
    },
    "required": ["pattern", "kline", "volume", "combined"],
    "additionalProperties": False,
}


def compose_text(caption: dict) -> str:
    """三段串成一份敘述（FinBERT 編碼的就是這段）。答案卷只存三段，用到時才組，不另存。"""
    return (f"[K-line] {caption['kline'].strip()} [Volume] {caption['volume'].strip()} "
            f"[Combined] {caption['combined'].strip()}")


def _window(ohlcv, date: str, window_days: int):
    import pandas as pd
    idx = ohlcv.index.get_loc(pd.Timestamp(date))
    if idx < window_days - 1:
        return None
    return ohlcv.iloc[idx - window_days + 1: idx + 1]


def volume_formula(window) -> dict:
    """成交量標籤的公式版本（四選一，依序判斷）：最後 3 天 ÷ 20 日均量 -> 爆量；
    否則近 5 日均量 ÷ 前 15 日均量 -> 放量／縮量／量平。"""
    vol = window["Volume"].to_numpy(dtype=float)
    k = patterns.VOLUME_RECENT_DAYS
    ratio = float(vol[-k:].mean() / max(vol[:-k].mean(), 1e-9))
    recent_max = float(vol[-patterns.VOLUME_SPIKE_DAYS:].max() / max(vol.mean(), 1e-9))
    if recent_max > patterns.VOLUME_SPIKE_RATIO:
        label = "SPIKE"
    elif ratio > patterns.VOLUME_EXPAND_RATIO:
        label = "EXPANDING"
    elif ratio < patterns.VOLUME_CONTRACT_RATIO:
        label = "CONTRACTING"
    else:
        label = "FLAT"
    return {"label": label, "ratio_5_15": round(ratio, 4), "recent_max_ratio_20": round(recent_max, 4)}


def haiku_caption(record: dict, volume_label: str, model: str, effort: str | None) -> dict:
    """Haiku 判型態、寫三段敘述；成交量標籤由公式決定，只把狀況（不含倍數）告訴 Haiku。"""
    from module_b_encoder.llm_client import call_claude_images
    images = [paths.ROOT / item["path"] for item in record["chart"]["inputs"]]
    user = USER_PROMPT.format(volume_state=VOLUME_STATE_TEXT[volume_label])
    raw = call_claude_images(SYSTEM_PROMPT, user, model, images, schema=CAPTION_SCHEMA, effort=effort)
    obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    if obj["pattern"] not in patterns.PATTERNS:
        raise ValueError(f"Haiku 回傳不在清單裡的型態: {obj}")
    return {"pattern": obj["pattern"], "volume_label": volume_label,
            "kline": obj["kline"], "volume": obj["volume"], "combined": obj["combined"]}


def load_captions(ticker: str) -> dict:
    path = paths.chart_caption_path(ticker)
    if not path.exists():
        raise SystemExit(f"找不到 K 線文字敘述 {path}，先跑 python -m module_a_data.chart_caption --ticker {ticker}")
    return read_json(path)


def distribution_report(captions: dict) -> dict:
    """型態與成交量標籤的分布（成交量標籤由公式決定，不再算一致率）。"""
    rows = list(captions.values())
    return {"n": len(rows),
            "pattern_distribution": {p: sum(c["pattern"] == p for c in rows) for p in patterns.PATTERN_IDS},
            "volume_label_distribution": {v: sum(c["volume_label"] == v for c in rows)
                                          for v in patterns.VOLUME_LABEL_IDS}}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--model", default=cfg["caption"]["model"])
    ap.add_argument("--effort", default=cfg["caption"].get("effort"),
                    help="Haiku 的 output_config.effort（low/medium/high）；不給就用模型預設")
    ap.add_argument("--limit", type=int, default=None, help="只做前 N 個還沒做的日期")
    ap.add_argument("--sample", type=int, default=None, help="隨機抽 N 天（固定 seed），人工抽查用")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="已存在的日期也重做")
    ap.add_argument("--check", action="store_true", help="不呼叫 API，只印出現有答案卷的一致率")
    args = ap.parse_args()

    out_path = paths.chart_caption_path(args.ticker)
    captions = read_json(out_path) if out_path.exists() else {}
    if args.check:
        print(json.dumps(distribution_report(captions), ensure_ascii=False, indent=2))
        return

    from module_a_data.crawler.fetch_ohlcv import load_ohlcv
    ohlcv = load_ohlcv(args.ticker)
    window_days = cfg["chart"]["window_days"]
    files = sorted((paths.DATASET / args.ticker).glob("*.json"))
    if not files:
        raise SystemExit(f"找不到 {paths.DATASET / args.ticker}，先跑 module_a_data.build_dataset")
    dates = [f.stem for f in files]
    if args.sample:
        dates = sorted(random.Random(cfg["seed"]).sample(dates, min(args.sample, len(dates))))
    todo = [d for d in dates if args.overwrite or d not in captions]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[chart_caption] {args.ticker}: {len(dates)} 天，要做 {len(todo)} 天 -> {out_path}")

    lock = threading.Lock()

    def one(date: str) -> dict:
        window = _window(ohlcv, date, window_days)
        if window is None:
            raise ValueError(f"{date} 前面不足 {window_days} 個交易日")
        formula = volume_formula(window)
        obj = haiku_caption(read_json(paths.daily_json(args.ticker, date)), formula.pop("label"),
                            args.model, args.effort)
        obj["volume_ratios"] = formula
        obj["model"] = args.model
        return obj

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, d): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            date = futures[fut]
            try:
                result = fut.result()
            except Exception as error:  # noqa: BLE001  單天失敗不中斷，最後列出來
                failed.append(date)
                print(f"  [{i}/{len(todo)}] {date}: 失敗 {error}")
                continue
            with lock:
                captions[date] = result
                if i % 25 == 0 or i == len(todo):
                    write_json(dict(sorted(captions.items())), out_path)
            print(f"  [{i}/{len(todo)}] {date}: {result['pattern']} / {result['volume_label']}", flush=True)
    write_json(dict(sorted(captions.items())), out_path)

    print(f"[chart_caption] {len(captions)} 天 -> {out_path}")
    print(f"[chart_caption] 型態與成交量分布：{json.dumps(distribution_report(captions), ensure_ascii=False)}")
    from module_b_encoder.llm_client import USAGE
    print(f"[chart_caption] Claude token 用量: input {USAGE['input_tokens']:,}, output {USAGE['output_tokens']:,}")
    if failed:
        print(f"[chart_caption] {len(failed)} 天失敗，再跑一次會自動補上: {failed[:20]}")


if __name__ == "__main__":
    main()
