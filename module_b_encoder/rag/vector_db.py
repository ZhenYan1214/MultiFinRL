"""RAG 向量資料庫建置：把財報 / 法說會 Chunk 編碼後存入 FAISS。

- Chunk 編碼用與 text_encoder 相同的模型（encode_pooled），確保 query/key 同空間。
- 索引保存 chunk_id 與原文對應，供 retriever 回傳 retrieved_chunk_ids 與重編碼。
- 每個 chunk 額外記一個 `source`（例如 "filing"／"transcript"），供 retriever 做
  按來源分配名額的檢索（`docs/decisions.md #70`：法說會逐字稿在向量相似度競爭裡
  系統性地贏過財報，純粹比相似度排序會讓同一種文體壟斷 top-K，需要能依來源篩選）。
  `source` 是選填欄位，不填則行為與之前完全一樣。
- 持久化：data/vectors/_faiss/{TICKER}.index + {TICKER}_meta.json
"""
import json

import faiss
import numpy as np

from shared import paths

DB_DIR = paths.VECTORS / "_faiss"


class ChunkVectorDB:
    """chunk_id / 原文 / 向量 / 來源 四者對應的 FAISS 內積索引（向量先 L2 normalize = cosine）。"""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.sources: list[str | None] = []

    def add(self, chunk_id: str, text: str, vector: np.ndarray, source: str | None = None) -> None:
        v = vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(v)
        self.index.add(v)
        self.chunk_ids.append(chunk_id)
        self.texts.append(text)
        self.sources.append(source)

    def add_chunks(self, chunks: list[dict], text_encoder, source: str | None = None) -> None:
        """chunks: data_format.md 的 chunk dict 列表；已存在的 chunk_id 跳過。

        source：這批 chunk 的來源標籤（例如 "filing"／"transcript"），同一次呼叫的
        chunks 共用同一個 source；要混合來源請分開呼叫兩次。
        """
        existing = set(self.chunk_ids)
        for c in chunks:
            if c["chunk_id"] in existing:
                continue
            self.add(c["chunk_id"], c["text"], text_encoder.encode_pooled(c["text"]), source)

    def search(self, query: np.ndarray, k: int = 3) -> list[tuple[str, str, float, str | None]]:
        """回傳 [(chunk_id, text, score, source)]，相似度由高至低。"""
        if self.index.ntotal == 0:
            return []
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, min(k, self.index.ntotal))
        return [(self.chunk_ids[i], self.texts[i], float(s), self.sources[i])
                for s, i in zip(scores[0], idxs[0]) if i >= 0]

    # --- 持久化 ---
    def save(self, ticker: str) -> None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(DB_DIR / f"{ticker}.index"))
        meta = {"dim": self.dim, "chunk_ids": self.chunk_ids, "texts": self.texts,
                "sources": self.sources}
        (DB_DIR / f"{ticker}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, ticker: str) -> "ChunkVectorDB":
        meta = json.loads((DB_DIR / f"{ticker}_meta.json").read_text(encoding="utf-8"))
        db = cls(meta["dim"])
        db.index = faiss.read_index(str(DB_DIR / f"{ticker}.index"))
        db.chunk_ids, db.texts = meta["chunk_ids"], meta["texts"]
        # 舊版存檔（#70 之前）沒有 sources 欄位，讀不到時全部當 None，行為等同未分類
        db.sources = meta.get("sources") or [None] * len(db.chunk_ids)
        return db
