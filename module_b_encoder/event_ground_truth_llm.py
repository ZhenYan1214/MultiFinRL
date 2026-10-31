"""用 LLM API 幫事件抽取（event_extraction.py）建立 ground truth 答案卷。

最早的 149 天 ground truth 是人工（Claude 在對話中直接讀資料）分批標記完成的
（docs/decisions.md #32）；2026-10 起改用這支腳本以 Claude Haiku 5.5 全量重標 1381 天。
這支腳本跟事件抽取的正式 LLM 方法（event_extraction.py --method llm）共用同一套呼叫邏輯
（module_b_encoder/llm_client.py）和同一份 prompt（event_ground_truth_prompt.py），
標記標準一致，之後可以互相比較。

要啟用：
    1. pip install anthropic   （openai/deepseek 則 pip install openai）
    2. .env 加對應的 key：ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY
    3. python -m module_b_encoder.event_ground_truth_llm --ticker AAPL                  # 分層抽樣 150 天
       python -m module_b_encoder.event_ground_truth_llm --all --overwrite --workers 8  # 全量重標

中途中斷（網路、額度）時，進度每 25 天存一次；去掉 --overwrite 再跑一次就只補還沒標的天。

費用參考（2026-10 定價，Claude Haiku 5.5 $0.10/M input、$0.50/M output）：2026-10-29 全量
1381 天實測 input 2,110 萬、output 64 萬 tokens（每天平均約 1.5 萬），約 2.4 美金，
8 個執行緒約 10 分鐘。實際數字以當下官網定價為準。
"""
import argparse
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from shared import paths
from shared.utils import load_config, read_json, write_json
from module_b_encoder.llm_client import DEFAULT_MODEL, USAGE, label_events

SAVE_EVERY = 25


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
    ap.add_argument("--all", action="store_true", help="標全部天數，不抽樣")
    ap.add_argument("--overwrite", action="store_true", help="不沿用既有標記，全部重標")
    ap.add_argument("--workers", type=int, default=1, help="同時呼叫 API 的執行緒數")
    args = ap.parse_args()
    model = args.model or DEFAULT_MODEL[args.provider]

    files = sorted((paths.DATASET / args.ticker).glob("*.json"))
    if not files:
        raise SystemExit(f"找不到 A 的資料: {paths.DATASET / args.ticker}")
    sample = files if args.all else stratified_sample(files, args.n_sample, args.seed)

    out_path = paths.event_ground_truth_path(args.ticker)
    ground_truth: dict[str, list[str]] = {}
    if out_path.exists() and not args.overwrite:
        ground_truth = read_json(out_path)  # 已有的標記先保留，只補新的
    todo = [f.stem for f in sample if f.stem not in ground_truth]
    print(f"[event_ground_truth_llm] {args.ticker}: 選 {len(sample)}/{len(files)} 天，要標 {len(todo)} 天,"
          f" provider={args.provider}, model={model}, workers={args.workers}")

    def save():
        write_json(dict(sorted(ground_truth.items())), out_path)

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(label_day, args.ticker, d, args.provider, model): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            date = futures[fut]
            try:
                ground_truth[date] = fut.result()
                print(f"  [{i}/{len(todo)}] {date}: {ground_truth[date]}", flush=True)
            except Exception as e:  # 單天失敗不中斷整批，之後重跑（不加 --overwrite）會補上
                failed.append(date)
                print(f"  [{i}/{len(todo)}] {date}: 失敗 {type(e).__name__}: {e}", flush=True)
            if i % SAVE_EVERY == 0:
                save()

    save()
    print(f"[event_ground_truth_llm] {len(ground_truth)} 天 -> {out_path}")
    if args.provider == "claude":
        print(f"[event_ground_truth_llm] Claude token 用量: input {USAGE['input_tokens']:,},"
              f" output {USAGE['output_tokens']:,}")
    if failed:
        print(f"[event_ground_truth_llm] {len(failed)} 天失敗，去掉 --overwrite 再跑一次可補上: {failed}")


if __name__ == "__main__":
    main()
