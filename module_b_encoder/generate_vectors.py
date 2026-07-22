"""B 模組主程式：對每個交易日產出 H_v / H_t / H_r 三個向量。

輸入：data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json（讀入前先 validate）
輸出：data/vectors/{TICKER}/{YYYY-MM-DD}/{H_v,H_t,H_r}.npy + index.json
寫出 index.json 前必須通過 shared.schemas.validate_vector_index()。

用法：
    python -m module_b_encoder.generate_vectors --fake --n 10       # 第一階段：假資料測通流程
    python -m module_b_encoder.generate_vectors --ticker AAPL       # 第二階段：真實資料
"""
import argparse
import datetime as dt

import numpy as np

from shared import paths, schemas
from shared.utils import load_config, read_json, write_json


def save_vectors(ticker: str, date: str, h_v, h_t, h_r, chunk_ids: list[str],
                 vision_id: str, text_id: str, top_k: int, source_json: str) -> None:
    """三個 .npy + index.json 落地（validate 後）。"""
    out_dir = paths.vector_dir(ticker, date)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "H_v.npy", h_v)
    np.save(out_dir / "H_t.npy", h_t)
    np.save(out_dir / "H_r.npy", h_r)
    index = {
        "ticker": ticker,
        "date": date,
        "vectors": {
            "H_v": {"file": "H_v.npy", "shape": list(h_v.shape), "dtype": "float32", "encoder": vision_id},
            "H_t": {"file": "H_t.npy", "shape": list(h_t.shape), "dtype": "float32", "encoder": text_id},
            "H_r": {"file": "H_r.npy", "shape": list(h_r.shape), "dtype": "float32", "encoder": text_id, "top_k": top_k},
        },
        "retrieved_chunk_ids": chunk_ids,
        "source_json": source_json,
    }
    schemas.validate_vector_index(index)
    write_json(index, out_dir / "index.json")


def run_fake(cfg, ticker: str, n: int) -> None:
    """第一階段：不需要 A 的資料、不需要 GPU，隨機向量測通儲存格式。"""
    from module_b_encoder.encoders.vision_encoder import fake_h_v
    from module_b_encoder.encoders.text_encoder import fake_h_t
    from module_b_encoder.rag.retriever import fake_h_r

    k = cfg["rag"]["top_k"]
    d0 = dt.date(2021, 3, 1)
    for i in range(n):
        date = (d0 + dt.timedelta(days=i)).isoformat()
        h_r, chunk_ids = fake_h_r(k, seed=i)
        save_vectors(ticker, date, fake_h_v(seed=i), fake_h_t(seed=i), h_r, chunk_ids,
                     cfg["encoders"]["vision"], cfg["encoders"]["text"], k,
                     f"data/processed/dataset/{ticker}/{date}.json")
    print(f"[generate_vectors] FAKE {ticker}: {n} days -> {paths.VECTORS / ticker}")


def run_real(cfg, ticker: str, limit: int | None) -> None:
    """第二階段：讀 A 的每日 JSON，跑真模型。"""
    from module_b_encoder.encoders.vision_encoder import VisionEncoder
    from module_b_encoder.encoders.text_encoder import TextEncoder
    from module_b_encoder.rag.vector_db import ChunkVectorDB
    from module_b_encoder.rag.retriever import retrieve

    vision = VisionEncoder(cfg["encoders"]["vision"])
    text = TextEncoder(cfg["encoders"]["text"])
    k = cfg["rag"]["top_k"]
    db = ChunkVectorDB()

    dataset_dir = paths.DATASET / ticker
    files = sorted(dataset_dir.glob("*.json"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"找不到 A 的資料: {dataset_dir}，先跑 module_a_data.build_dataset")

    for f in files:
        record = read_json(f)
        schemas.validate_daily_record(record)
        date = record["date"]

        # 當日文件 chunk 進向量庫（隨時間累積，天然避免 look-ahead）
        db.add_chunks(record["filing_chunks"] + record["transcript_chunks"], text)

        h_v = vision.encode(paths.ROOT / record["chart"]["path"])
        h_t = text.encode_news(record["news"])
        h_r, chunk_ids = retrieve(h_v, h_t, db, text, k)

        save_vectors(ticker, date, h_v, h_t, h_r, chunk_ids,
                     vision.model_id, text.model_id, k,
                     str(f.relative_to(paths.ROOT)).replace("\\", "/"))
        print(f"[generate_vectors] {ticker} {date} done")

    db.save(ticker)
    print(f"[generate_vectors] {ticker}: {len(files)} days -> {paths.VECTORS / ticker}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--fake", action="store_true", help="第一階段：隨機向量測通流程")
    ap.add_argument("--n", type=int, default=10, help="--fake 時產幾天")
    ap.add_argument("--limit", type=int, default=None, help="真實模式只跑前 N 天")
    args = ap.parse_args()

    if args.fake:
        run_fake(cfg, args.ticker, args.n)
    else:
        run_real(cfg, args.ticker, args.limit)


if __name__ == "__main__":
    main()
