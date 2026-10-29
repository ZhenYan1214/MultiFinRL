"""只重做 H_r 的檢索（含時效懲罰），沿用既有 H_v／H_t，不重算 ViT。

用途：比較不同檢索設定（docs/decisions.md #104）。照時間順序重建向量庫——每天只加入當天
daily record 列出的 chunk（與 generate_vectors.py 相同，不會檢索到未來文件），chunk 向量直接
取自 FAISS 存檔，不重新編碼；query 用同一個 build_query。β=0 時應與現有 retrieved_chunk_ids
完全一致，用來驗證重建正確。

    # 1. 只看統計，不寫檔：比較幾個 β 的檢索結果
    python -m module_b_encoder.rerun_retrieval --ticker AAPL --betas 0 0.005 0.01 0.02
    # 2. 選定 β 後產生新的 H_r（存到 data/vectors_{tag}/，不動正式向量）
    python -m module_b_encoder.rerun_retrieval --ticker AAPL --betas 0.01 --write --tag recency
"""
import argparse
import datetime as dt
import statistics as st

import numpy as np

from shared import paths
from shared.doc_meta import parse_chunk_id, superseded_count
from shared.utils import load_config, read_json, write_json

POOL = 50


def trading_days(ticker: str) -> list[str]:
    return sorted(d.name for d in (paths.VECTORS / ticker).iterdir() if (d / "index.json").exists())


def daily_chunks(record: dict) -> list[tuple[str, str]]:
    """與 generate_vectors.py 相同的加入順序：(chunk_id, source)。"""
    from module_b_encoder.generate_vectors import _macro_chunks
    out = [(c["chunk_id"], "filing") for c in record["filing_chunks"]]
    out += [(c["chunk_id"], "transcript") for c in record["transcript_chunks"]]
    macro = _macro_chunks(record)
    out += [(c["chunk_id"], "macro_indicator") for c in macro["indicators"]]
    out += [(c["chunk_id"], "macro_news") for c in macro["news"]]
    return out


def simulate(ticker: str, beta: float, alpha: float, k: int, stored: dict) -> dict[str, list[tuple]]:
    """回傳 {date: [(chunk_id, text, score, source), ...]}（選中的 k 段，可能不足 k）。"""
    from module_b_encoder.rag.retriever import select_hits
    from module_b_encoder.rag.vector_db import ChunkVectorDB

    db = ChunkVectorDB()
    seen: set[str] = set()
    out = {}
    for date in trading_days(ticker):
        record = read_json(paths.daily_json(ticker, date))
        for chunk_id, source in daily_chunks(record):
            if chunk_id in seen or chunk_id not in stored:
                continue
            vec, text = stored[chunk_id]
            db.add(chunk_id, text, vec, source)
            seen.add(chunk_id)
        d = paths.vector_dir(ticker, date)
        h_v, h_t = np.load(d / "H_v.npy"), np.load(d / "H_t.npy")
        staleness = {"beta": beta, "date": date, "ticker": ticker, "pool": POOL} if beta > 0 else None
        out[date] = select_hits(h_v, h_t, db, k, alpha, None, staleness)
    return out


def summarize(ticker: str, result: dict, original: dict, cosine: dict) -> dict:
    sups, ages, same_ids, cos_sel, cos_orig, n_docs = [], [], [], [], [], []
    types = {}
    for date, hits in result.items():
        ids = [h[0] for h in hits]
        for cid in ids:
            src, doc_date = parse_chunk_id(cid)
            sups.append(superseded_count(ticker, src, doc_date, date))
            ages.append((dt.date.fromisoformat(date) - doc_date).days)
            types[src] = types.get(src, 0) + 1
        same_ids.append(len(set(ids) & set(original[date])) / max(len(original[date]), 1))
        cos_sel.append(np.mean([cosine[date][c] for c in ids]))
        cos_orig.append(np.mean([cosine[date][c] for c in original[date] if c in cosine[date]]))
        n_docs.append(len({c.rsplit("_", 1)[0] for c in ids}))
    sups = np.asarray(sups)
    total = sum(types.values())
    return {
        "完全相同的天數": f"{np.mean([s == 1 for s in same_ids]):.1%}",
        "與原 top-3 重疊": f"{np.mean(same_ids):.1%}",
        "被取代 0 次": f"{np.mean(sups == 0):.1%}",
        "被取代 ≥3 次": f"{np.mean(sups >= 3):.1%}",
        "被取代次數中位數": float(np.median(sups)),
        "距今天數中位數": float(np.median(ages)),
        "cosine 平均（原始相似度）": f"{np.mean(cos_sel):.4f}（原 {np.mean(cos_orig):.4f}）",
        "每天來自幾份文件": round(float(np.mean(n_docs)), 2),
        "類型占比": {t: f"{n / total:.0%}" for t, n in sorted(types.items())},
    }


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.005, 0.01, 0.02])
    ap.add_argument("--write", action="store_true", help="用 --betas 的單一 β 產生新 H_r 並寫檔")
    ap.add_argument("--tag", default="recency", help="輸出到 data/vectors_{tag}/")
    args = ap.parse_args()
    if args.write and len(args.betas) != 1:
        raise SystemExit("--write 時 --betas 只能給一個值")

    text_encoder = None
    if args.write:
        # Apple Silicon：先初始化 PyTorch 模型再匯入 FAISS，避免原生層 SIGSEGV（同 generate_vectors.py）
        from module_b_encoder.encoders.text_encoder import TextEncoder
        text_encoder = TextEncoder(cfg["encoders"]["text"])
    import faiss

    k, alpha = cfg["rag"]["top_k"], cfg["rag"].get("query_alpha", 0.5)
    index = faiss.read_index(str(paths.VECTORS / "_faiss" / f"{args.ticker}.index"))
    meta = read_json(paths.VECTORS / "_faiss" / f"{args.ticker}_meta.json")
    stored = {cid: (index.reconstruct(i), text) for i, (cid, text) in enumerate(zip(meta["chunk_ids"], meta["texts"]))}
    original = {d: [c for c in read_json(paths.vector_dir(args.ticker, d) / "index.json")["retrieved_chunk_ids"]
                    if c != "PAD"] for d in trading_days(args.ticker)}

    for beta in args.betas:
        result = simulate(args.ticker, beta, alpha, k, stored)
        if beta == 0:
            same = np.mean([[h[0] for h in hits] == original[d] for d, hits in result.items()])
            print(f"[rerun_retrieval] β=0 重建驗證：與現有 retrieved_chunk_ids 完全一致的天數 {same:.1%}")
        # 原始 cosine（不含懲罰），用來看相似度的代價
        cosine = {}
        from module_b_encoder.rag.retriever import build_query
        for d in result:
            vd = paths.vector_dir(args.ticker, d)
            q = build_query(np.load(vd / "H_v.npy"), np.load(vd / "H_t.npy"))
            q = q / np.linalg.norm(q)
            ids = {h[0] for h in result[d]} | set(original[d])
            cosine[d] = {c: float(stored[c][0] @ q / np.linalg.norm(stored[c][0])) for c in ids}
        print(f"\n[rerun_retrieval] β={beta}")
        for key, value in summarize(args.ticker, result, original, cosine).items():
            print(f"  {key}: {value}")

        if args.write:
            cache: dict[str, np.ndarray] = {}
            for n, (date, hits) in enumerate(result.items(), 1):
                ids = [h[0] for h in hits]
                h_r = np.zeros((k, 512, 768), dtype=np.float32)
                for i, (cid, text, _score, _src) in enumerate(hits):
                    if cid not in cache:
                        cache[cid] = text_encoder.encode(text)
                    h_r[i] = cache[cid]
                out_dir = paths.vector_dir(args.ticker, date, args.tag)
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "H_r.npy", h_r)
                idx = read_json(paths.vector_dir(args.ticker, date) / "index.json")
                idx["retrieved_chunk_ids"] = ids + ["PAD"] * (k - len(ids))
                idx["retrieval"] = {"staleness_beta": beta, "pool": POOL,
                                    "type_weights": "retriever.DEFAULT_TYPE_WEIGHTS",
                                    "note": "H_v／H_t 沿用 data/vectors/，本目錄只有 H_r"}
                write_json(idx, out_dir / "index.json")
                if n % 200 == 0 or n == len(result):
                    print(f"[rerun_retrieval] 寫入 {n}/{len(result)} 天（已編碼 {len(cache)} 段 chunk）")
            print(f"[rerun_retrieval] -> {paths.vectors_root(args.tag) / args.ticker}")


if __name__ == "__main__":
    main()
