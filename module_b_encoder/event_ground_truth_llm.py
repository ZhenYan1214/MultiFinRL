"""用 LLM API 幫事件抽取（event_extraction.py）建立 ground truth 答案卷。

現況：149 天的 ground truth 是人工（Claude 在對話中直接讀資料）分批標記完成的
（docs/decisions.md #32）。這支腳本跟事件抽取的正式 LLM 方法（event_extraction.py
--method llm）共用同一套呼叫邏輯（module_b_encoder/llm_client.py）和同一份 prompt
（event_ground_truth_prompt.py），標記標準一致，之後可以互相比較，也可以直接改用
這支腳本重跑或擴大標記範圍。

要啟用：
    1. pip install openai   （claude/deepseek 都走這個套件；用 claude 則 pip install anthropic）
    2. .env 加對應的 key：ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY
    3. python -m module_b_encoder.event_ground_truth_llm --ticker AAPL

費用參考（2026-08 定價）：AAPL 每天平均約 2.8 萬 tokens，抽樣 150 天用 Claude Haiku
（$1/M input tokens）約 4-5 美金，全量 1381 天約 35-40 美金；DeepSeek/OpenAI 費用量級
類似，實際數字以當下官網定價為準。
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

from shared import paths
from shared.utils import load_config, read_json, write_json
from module_b_encoder.llm_client import DEFAULT_MODEL, label_events


def stratified_sample(files: list[Path], n: int, seed: int) -> list[Path]:
    """依年份分層隨機抽樣，讓每一年在樣本裡都按比例出現，不偏向近期或早期。"""
    by_year: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        by_year[f.stem[:4]].append(f)

    rng = random.Random(seed)
    total = len(files)
    sampled: list[Path] = []
    for year, year_files in sorted(by_year.items()):
        quota = max(1, round(n * len(year_files) / total))
        sampled.extend(rng.sample(year_files, min(quota, len(year_files))))
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)  # 隨機修剪超額，不能照日期截斷（會偏向早期年份）
    return sorted(sampled)


def label_day(ticker: str, date: str, provider: str, model: str) -> list[str]:
    record = read_json(paths.daily_json(ticker, date))
    return label_events(ticker, date, record["news"], record["filing_chunks"],
                        record["transcript_chunks"], provider, model)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--provider", choices=["claude", "openai", "deepseek"], default="claude")
    ap.add_argument("--model", default=None, help="不指定就用 provider 的預設模型")
    ap.add_argument("--n_sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model = args.model or DEFAULT_MODEL[args.provider]

    files = sorted((paths.DATASET / args.ticker).glob("*.json"))
    if not files:
        raise SystemExit(f"找不到 A 的資料: {paths.DATASET / args.ticker}")
    sample = stratified_sample(files, args.n_sample, args.seed)
    print(f"[event_ground_truth_llm] {args.ticker}: 抽樣 {len(sample)}/{len(files)} 天,"
         f" provider={args.provider}, model={model}")

    out_path = paths.event_ground_truth_path(args.ticker)
    ground_truth: dict[str, list[str]] = {}
    if out_path.exists():
        ground_truth = read_json(out_path)  # 已有的標記先保留，新抽樣只補新的

    for i, f in enumerate(sample, 1):
        date = f.stem
        if date in ground_truth:
            continue
        events = label_day(args.ticker, date, args.provider, model)
        ground_truth[date] = events
        print(f"  [{i}/{len(sample)}] {date}: {events}")

    write_json(ground_truth, out_path)
    print(f"[event_ground_truth_llm] {len(ground_truth)} 天 -> {out_path}")


if __name__ == "__main__":
    main()
