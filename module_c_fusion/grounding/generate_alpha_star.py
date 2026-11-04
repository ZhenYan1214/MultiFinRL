"""用 Claude Haiku 對每天 H_r 檢索到的 chunk 打相關性分數，作為 L_ground 的 α* 來源。

輸出：data/labels/grounding/{ticker}.json（答案卷，進版本控制，同 y_belief）
    {date: {"chunk_ids": [...],             # 打分當時的 retrieved_chunk_ids（retriever 順序）
            "scores": [1.5, 0.0, 1.0],      # 各次平均，依 chunk_ids 順序；PAD 為 null
            "runs": [{"scores": [2, 0, 1], "reasons": ["...", ...]}, ...],   # 每次原始結果
            "model": "...",
            "usage": {"input_tokens": ..., "output_tokens": ...}}}         # 各次加總
分數是 0–3（平均後可能是小數）；轉成 α* 的溫度與「其他」槽位分數是訓練時的超參數（grounding/alpha.py）。

每天預設打 2 次（--repeats）並取平均（AAPL 100 天試跑時打了 3 次）：Haiku 無法設 temperature，同一段 chunk 換位置重打，分數
標準差約 0.22；只打 1 次時，約 1/3 的日子「最高分段落」會跟 3 次平均的結果不同。
排列順序第 1 次隨機、之後依序輪替（[x,y,z] → [y,z,x] → [z,x,y]），K=3 時每段在前、中、後各
出現一次。試跑中同一段放在中間與放在頭尾的分數平均只差 -0.03，位置偏誤不明顯，輪替只是保險。

要啟用：
    1. pip install anthropic
    2. .env 加上 ANTHROPIC_API_KEY
    3. python -m module_c_fusion.grounding.generate_alpha_star --ticker AAPL --dry_run         # 只印 prompt，不呼叫 API
       python -m module_c_fusion.grounding.generate_alpha_star --ticker AAPL --sample 100      # 先試跑 100 天
       python -m module_c_fusion.grounding.generate_alpha_star --ticker AAPL                   # 全量（可中斷續跑）

已打過分、且 chunk_ids 跟目前向量一致的日期只補不足的次數（例如從 1 次補到 2 次）；
重新產生向量後 chunk 變了的日期會重打。修改 prompt 後要全部重打，加 --overwrite。
"""
import argparse
import datetime as dt
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from shared import paths
from shared.temporal_split import load_temporal_manifest
from shared.utils import read_json
from module_c_fusion.grounding.prompt import (
    DISPLAY_LABELS, SYSTEM_PROMPT, build_user_prompt, output_schema,
)
from module_c_fusion.grounding.retrieval_meta import PAD_CHUNK_ID, parse_chunk_id

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "claude-haiku-5-5"
# 2026-10 牌價（每百萬 token，prompt ≤100K）；只用來在結束時印出估算花費。
PRICE_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50)}


def _dumps(obj, indent: int = 0) -> str:
    """跟 json.dump(indent=2) 相同，但「分數、chunk_ids 這類短的純量陣列」寫在同一行，例如
    "scores": [1.5, 0, 2]，檔案比較好讀；理由這類長字串仍一行一個。只影響檔案排版，讀進來的內容完全一樣。"""
    pad, inner = " " * indent, " " * (indent + 2)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {_dumps(v, indent + 2)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(obj, list):
        scalar = all(x is None or isinstance(x, (int, float, bool)) or (isinstance(x, str) and len(x) <= 60)
                     for x in obj)
        if scalar:
            return json.dumps(obj, ensure_ascii=False)
        return "[\n" + ",\n".join(f"{inner}{_dumps(x, indent + 2)}" for x in obj) + f"\n{pad}]"
    return json.dumps(obj, ensure_ascii=False)


def write_labels(labels: dict, path) -> None:
    """依日期排序後寫出答案卷。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_dumps(dict(sorted(labels.items()))) + "\n")


def load_chunk_texts(ticker: str) -> dict[str, str]:
    """FAISS 索引旁的 meta 檔記錄了每個 chunk 的原文（module_b_encoder/rag/vector_db.py）。"""
    meta = read_json(paths.VECTORS / "_faiss" / f"{ticker}_meta.json")
    return dict(zip(meta["chunk_ids"], meta["texts"]))


def target_dates(ticker: str, split: str) -> list[str]:
    if split == "all":
        return sorted(d.name for d in (paths.VECTORS / ticker).iterdir()
                      if (d / "index.json").exists())
    return sorted(load_temporal_manifest(ticker)["dates"][split])


def display_order(ticker: str, date: str, real: list[int], run: int) -> list[int]:
    """第 run 次打分時，chunk 給 LLM 看的順序（retriever 位置的排列）。固定 seed、可重現。"""
    order = real[:]
    cycle, shift = divmod(run, len(order))
    seed = f"{ticker}|{date}" if cycle == 0 else f"{ticker}|{date}|{cycle}"
    random.Random(seed).shuffle(order)
    return order[shift:] + order[:shift]  # 輪替：每段輪流出現在每個位置


def build_request(ticker: str, date: str, texts: dict[str, str], run: int = 0,
                  vectors_tag: str | None = None) -> dict | None:
    """組出一天第 run 次打分的 prompt；全部都是 PAD（向量庫還沒有文件）時回傳 None。
    vectors_tag 不為 None 時，評的是 data/vectors_{tag}/ 那套檢索結果的 chunk。"""
    index = read_json(paths.vector_dir(ticker, date, vectors_tag) / "index.json")
    chunk_ids = index["retrieved_chunk_ids"]
    real = [i for i, c in enumerate(chunk_ids) if c != PAD_CHUNK_ID]
    if not real:
        return None

    # 只取新聞，絕不碰 record["prices"] / record["label"]（含未來收盤價）
    news = read_json(paths.daily_json(ticker, date))["news"]

    docs = []
    for label, i in zip(DISPLAY_LABELS, display_order(ticker, date, real, run)):
        doc_type, doc_date = parse_chunk_id(chunk_ids[i])
        age = None
        if doc_date is not None:
            age = max((dt.date.fromisoformat(date) - doc_date).days, 0)
        docs.append({
            "label": label, "index": i, "doc_type": doc_type,
            "doc_date": doc_date.isoformat() if doc_date else None,
            "age_days": age, "text": texts[chunk_ids[i]],
        })
    return {
        "chunk_ids": chunk_ids,
        "docs": docs,
        "user_prompt": build_user_prompt(ticker, date, news, docs),
    }


def call_haiku(client, model: str, effort: str, request: dict):
    labels = [d["label"] for d in request["docs"]]
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": request["user_prompt"]}],
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": output_schema(labels)},
        },
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={resp.stop_reason}")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage


def to_run(request: dict, parsed: dict) -> dict:
    """LLM 回傳（依 A/B/C 代號）-> 依 retriever 順序的 {"scores", "reasons"}。"""
    k = len(request["chunk_ids"])
    scores, reasons = [None] * k, [None] * k
    for d in request["docs"]:
        item = parsed[d["label"]]
        scores[d["index"]] = int(item["score"])
        reasons[d["index"]] = item["reason"]
    return {"scores": scores, "reasons": reasons}


def existing_runs(record: dict | None) -> list[dict]:
    """答案卷裡已有的各次結果；舊格式（只打 1 次、沒有 runs 欄位）視為一次。"""
    if not record:
        return []
    if "runs" in record:
        return record["runs"]
    return [{"scores": record["scores"], "reasons": record["reasons"]}]


def to_record(chunk_ids: list[str], runs: list[dict], model: str, usage: dict) -> dict:
    mean = []
    for i in range(len(chunk_ids)):
        values = [run["scores"][i] for run in runs if run["scores"][i] is not None]
        mean.append(round(sum(values) / len(values), 4) if values else None)
    return {"chunk_ids": chunk_ids, "scores": mean, "runs": runs, "model": model, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    ap.add_argument("--sample", type=int, default=None, help="隨機抽 N 天試跑（固定 seed，可重現）")
    ap.add_argument("--limit", type=int, default=None, help="只處理前 N 個待打分日期")
    ap.add_argument("--workers", type=int, default=4, help="同時送出的請求數")
    ap.add_argument("--dry_run", action="store_true", help="只印第一個待打分日期的 prompt，不呼叫 API")
    ap.add_argument("--overwrite", action="store_true", help="已打過分的日期也重打（例如修改 prompt 之後）")
    ap.add_argument("--vectors_tag", default=None,
                    help="評另一套檢索結果 data/vectors_{tag}/ 的 chunk（例如 recency），答案卷另存 {ticker}_{tag}.json")
    ap.add_argument("--repeats", type=int, default=2,
                    help="每天打分次數，取平均；排列順序依序輪替，K=3 時 3 次剛好讓每段在每個位置各出現一次")
    args = ap.parse_args()

    texts = load_chunk_texts(args.ticker)
    dates = target_dates(args.ticker, args.split)
    if args.sample:
        dates = sorted(random.Random(42).sample(dates, min(args.sample, len(dates))))

    out_path = paths.grounding_labels_path(args.ticker, args.vectors_tag)
    labels = read_json(out_path) if out_path.exists() else {}

    if args.repeats < 1:
        raise SystemExit("--repeats 必須至少為 1")
    pending, skipped_pad = [], 0   # [(date, chunk_ids, 已有的 runs, 已有的 usage)]
    for date in dates:
        request = build_request(args.ticker, date, texts, vectors_tag=args.vectors_tag)
        if request is None:
            skipped_pad += 1
            continue
        done = labels.get(date)
        if args.overwrite or not done or done["chunk_ids"] != request["chunk_ids"]:
            done = None
        runs = existing_runs(done)
        if len(runs) >= args.repeats:
            continue
        usage = dict(done["usage"]) if done else {"input_tokens": 0, "output_tokens": 0}
        pending.append((date, request["chunk_ids"], runs, usage))
    if args.limit:
        pending = pending[:args.limit]
    n_calls = sum(args.repeats - len(item[2]) for item in pending)
    print(f"[alpha*] {args.ticker}: 範圍 {len(dates)} 天，待打分 {len(pending)} 天（共 {n_calls} 次呼叫），"
          f"全 PAD 略過 {skipped_pad} 天，model={args.model}，repeats={args.repeats}")

    if args.dry_run:
        if pending:
            date, _, runs, _ = pending[0]
            request = build_request(args.ticker, date, texts, run=len(runs), vectors_tag=args.vectors_tag)
            print(f"\n===== {date} SYSTEM =====\n{SYSTEM_PROMPT}\n===== USER =====\n{request['user_prompt']}")
            print(f"\n[alpha*] user prompt 長度 {len(request['user_prompt'])} 字元")
        return
    if not pending:
        return

    import anthropic
    client = anthropic.Anthropic(max_retries=5)
    lock = threading.Lock()
    totals = {"in": 0, "out": 0, "ok": 0, "fail": 0}

    def work(item):
        date, chunk_ids, runs, usage = item
        runs, usage = list(runs), dict(usage)
        new_in = new_out = 0
        for run in range(len(runs), args.repeats):
            request = build_request(args.ticker, date, texts, run=run, vectors_tag=args.vectors_tag)
            parsed, resp_usage = call_haiku(client, args.model, args.effort, request)
            runs.append(to_run(request, parsed))
            new_in += resp_usage.input_tokens
            new_out += resp_usage.output_tokens
        usage["input_tokens"] += new_in
        usage["output_tokens"] += new_out
        return date, to_record(chunk_ids, runs, args.model, usage), new_in, new_out

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, item): item[0] for item in pending}
        for n, future in enumerate(as_completed(futures), 1):
            date = futures[future]
            try:
                date, record, new_in, new_out = future.result()
            except anthropic.AuthenticationError:
                raise SystemExit("[alpha*] ANTHROPIC_API_KEY 無效，停止")
            except Exception as e:  # 單日失敗不中斷整批，下次重跑會自動補
                totals["fail"] += 1
                print(f"  [警告] {date} 失敗，跳過（{type(e).__name__}: {e}）")
                continue
            with lock:
                labels[date] = record
                totals["in"] += new_in
                totals["out"] += new_out
                totals["ok"] += 1
                write_labels(labels, out_path)  # 每天落地，中斷不會遺失進度
            print(f"  [{n}/{len(pending)}] {date}: 各次={[r['scores'] for r in record['runs']]} 平均={record['scores']}")

    price_in, price_out = PRICE_PER_MTOK.get(args.model, (None, None))
    cost = (f"，估算花費 ${totals['in'] / 1e6 * price_in + totals['out'] / 1e6 * price_out:.3f}"
            if price_in is not None else "")
    print(f"[alpha*] 完成 {totals['ok']} 天、失敗 {totals['fail']} 天；"
          f"input {totals['in']:,} / output {totals['out']:,} tokens{cost} -> {out_path}")


if __name__ == "__main__":
    main()
