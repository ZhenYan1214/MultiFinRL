"""讀取 generate_alpha_star.py 產生的 LLM 相關性分數，對齊到目前向量的 chunk 順序。

分數檔記錄了打分當時的 retrieved_chunk_ids。若之後重新產生向量（例如開了
--balance_sources、改了 H_t），同一天檢索到的 chunk 可能不同——這種日子的分數已經
不是在評這三段，一律視為無標籤，不拿來算 L_ground，並在載入時印出筆數提醒重新打分。
"""
import numpy as np

from shared import paths
from shared.utils import read_json
from module_c_fusion.grounding.retrieval_meta import PAD_CHUNK_ID, retrieved_chunk_ids


def load_grounding_scores(ticker: str, dates: list[str], *, shuffle: bool = False,
                          seed: int = 0, vectors_tag: str | None = None) -> dict[str, tuple[np.ndarray, bool]]:
    """回傳 {date: (scores [K] float32, has_label)}，scores 依目前 retrieved_chunk_ids 順序。

    shuffle=True 是對照組（隨機打亂 α*）：每天把分數在非 PAD 的 chunk 之間隨機重排，
    用來排除「只是多了一個正則化項就變好」的可能。
    """
    path = paths.grounding_labels_path(ticker, vectors_tag)
    labels = read_json(path) if path.exists() else {}
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[np.ndarray, bool]] = {}
    missing = stale = 0
    for date in dates:
        ids = retrieved_chunk_ids(ticker, date, vectors_tag)
        scores = np.zeros(len(ids), dtype=np.float32)
        record = labels.get(date)
        if record is None:
            missing += 1
            out[date] = (scores, False)
            continue
        if record["chunk_ids"] != ids:
            stale += 1
            out[date] = (scores, False)
            continue
        for i, (chunk_id, score) in enumerate(zip(ids, record["scores"])):
            if chunk_id != PAD_CHUNK_ID and score is not None:
                scores[i] = float(score)
        if shuffle:
            real = np.flatnonzero([chunk_id != PAD_CHUNK_ID for chunk_id in ids])
            scores[real] = scores[rng.permutation(real)]
        out[date] = (scores, True)
    print(f"[grounding] {ticker}: {len(dates) - missing - stale}/{len(dates)} 天有相關性分數"
          f"（未打分 {missing}、chunk 已變動需重打 {stale}）{'，分數已隨機打亂（對照組）' if shuffle else ''}")
    return out
