"""解析 H_r 檢索結果的 metadata：每段 chunk 的來源類型與文件距今天數。

chunk_id 解析與「被取代次數」的定義見 shared/doc_meta.py（B 的檢索也共用）。

融合層用這些 metadata 做兩件事（fusion/model.py）：
    ① 注意力衰減偏置：每個來源一個可學習的 λ，注意力分數減去 λ_src·log(1+天數)
       或 λ_src·被取代次數（--decay_basis）。
    ② 年齡 embedding：天數分桶後查表加到 chunk token 上。

"""
import datetime as dt

import numpy as np

from shared import paths
from shared.doc_meta import (  # noqa: F401  (re-export：其他模組沿用 retrieval_meta 的名稱)
    PAD_CHUNK_ID, document_dates, parse_chunk_id, superseded_count,
)
from shared.utils import read_json


# 0 保留給 PAD；fusion/model.py 的衰減 λ 依這個編號各自學習。
SOURCE_IDS = {
    "PAD": 0,
    "10-K": 1,
    "10-Q": 2,
    "8-K": 3,
    "earnings_call": 4,
    "macro_indicator": 5,
    "macro_news": 6,
    "unknown": 7,
}
SOURCE_NAMES = {v: k for k, v in SOURCE_IDS.items()}
N_SOURCES = len(SOURCE_IDS)

# 年齡分桶邊界（天）：0–7 / 8–30 / 31–90 / 91–180 / >180
AGE_BUCKET_EDGES = (7, 30, 90, 180)
N_AGE_BUCKETS = len(AGE_BUCKET_EDGES) + 1

def chunk_meta(chunk_ids: list[str], date: str, ticker: str | None = None) -> dict[str, np.ndarray]:
    """retrieved_chunk_ids + 交易日 -> {"age": [K] float32 天數, "sup": [K] float32 被取代次數,
    "src": [K] int64, "pad": [K] bool}。

    日期解析不到時天數記 0（不懲罰），來源記 unknown；文件日期晚於交易日（不應發生）時夾到 0。
    沒給 ticker 時被取代次數一律為 0。
    """
    today = dt.date.fromisoformat(date)
    ages, sups, srcs, pads = [], [], [], []
    for chunk_id in chunk_ids:
        source, doc_date = parse_chunk_id(chunk_id)
        ages.append(max((today - doc_date).days, 0) if doc_date else 0)
        sups.append(superseded_count(ticker, source, doc_date, date))
        srcs.append(SOURCE_IDS[source])
        pads.append(source == "PAD")
    return {
        "age": np.asarray(ages, dtype=np.float32),
        "sup": np.asarray(sups, dtype=np.float32),
        "src": np.asarray(srcs, dtype=np.int64),
        "pad": np.asarray(pads, dtype=bool),
    }


def load_retrieval_meta(ticker: str, date: str, vectors_tag: str | None = None) -> dict[str, np.ndarray]:
    """讀 B 的 index.json 取這天的 retrieved_chunk_ids，轉成 chunk_meta()。"""
    return chunk_meta(retrieved_chunk_ids(ticker, date, vectors_tag), date, ticker)


def retrieved_chunk_ids(ticker: str, date: str, vectors_tag: str | None = None) -> list[str]:
    """vectors_tag 不為 None 時讀 data/vectors_{tag}/ 的另一套檢索結果。"""
    return read_json(paths.vector_dir(ticker, date, vectors_tag) / "index.json")["retrieved_chunk_ids"]


def age_bucket(age):
    """天數 -> 分桶編號（numpy 或 torch 皆可，回傳同型別的整數陣列）。"""
    try:
        import torch
        if isinstance(age, torch.Tensor):
            edges = torch.tensor(AGE_BUCKET_EDGES, dtype=age.dtype, device=age.device)
            return torch.bucketize(age, edges, right=False)
    except ImportError:
        pass
    return np.searchsorted(np.asarray(AGE_BUCKET_EDGES), np.asarray(age), side="left")
