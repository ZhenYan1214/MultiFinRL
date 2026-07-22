"""RAG 檢索：以 H_v + H_t 合併成 query，檢索 top-K 相關文件 -> H_r。

- K=3（configs/config.yaml 的 rag.top_k）。
- query 目前做法：H_v 與 H_t 各自 mean-pool 成 [768] 後平均（兩模態等權）。
  B 可自行改進（例如可學習的投影），但 H_r 輸出格式不可變。
- H_r shape [K, 512, 768]：第一維為 K，順序依相似度由高至低；
  檢索到的每份 chunk 原文重新用 text_encoder 編碼成 [512, 768]。
- 檢索不足 K 份（向量庫太小）時，缺的部分以 0 補齊，chunk_id 補 "PAD"。
"""
import numpy as np


def build_query(h_v: np.ndarray, h_t: np.ndarray) -> np.ndarray:
    """H_v [197,768] + H_t [512,768] -> query [768]。"""
    return (h_v.mean(axis=0) + h_t.mean(axis=0)) / 2


def retrieve(h_v: np.ndarray, h_t: np.ndarray, db, text_encoder,
             k: int = 3) -> tuple[np.ndarray, list[str]]:
    """回傳 (H_r [K, 512, 768], retrieved_chunk_ids)。"""
    hits = db.search(build_query(h_v, h_t), k)
    h_r = np.zeros((k, 512, 768), dtype=np.float32)
    chunk_ids = []
    for i, (chunk_id, text, _score) in enumerate(hits):
        h_r[i] = text_encoder.encode(text)
        chunk_ids.append(chunk_id)
    chunk_ids += ["PAD"] * (k - len(chunk_ids))
    return h_r, chunk_ids


def fake_h_r(k: int = 3, shape=(512, 768), seed: int | None = None) -> tuple[np.ndarray, list[str]]:
    """第一階段假資料：隨機 H_r 與假 chunk_ids。"""
    rng = np.random.default_rng(seed)
    h_r = rng.standard_normal((k, *shape)).astype(np.float32)
    return h_r, [f"FAKE_CHUNK_{i:03d}" for i in range(k)]
