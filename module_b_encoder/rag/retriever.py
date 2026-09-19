"""RAG 檢索：以 H_v + H_t 合併成 query，檢索 top-K 相關文件 -> H_r。

- K=3（configs/config.yaml 的 rag.top_k）。
- query 做法（2026-08 修正，2026-09 擴充雙圖）：H_v 的圖別與 token 維度、H_t 的 token
  維度各自 mean-pool 成 [768] 後，
  各自做 L2 正規化（避免兩個不同模型出來的向量數值量級不同、其中一個不成比例主導 query），
  再依 alpha 加權合併（alpha 是 H_v 的權重，預設 0.5 對半，可在 configs/config.yaml 的
  rag.query_alpha 調整，不用改程式碼）。真正的可學習/attention-based 加權留待之後。
- H_r shape [K, 512, 768]：第一維為 K，順序依相似度由高至低；
  檢索到的每份 chunk 原文重新用 text_encoder 編碼成 [512, 768]。
- 檢索不足 K 份（向量庫太小）時，缺的部分以 0 補齊，chunk_id 補 "PAD"。
- query 向量本身不會被存進 H_r/index.json，只是內部檢索用的中間產物，
  改這裡不影響 docs/data_format.md 定案的輸出格式。

按來源分配名額（`quota` 參數，`docs/decisions.md #70`）：
  加入法說會逐字稿後發現，逐字稿只佔向量庫 25% 的 chunk，卻拿下 75% 的 top-3 檢索結果
  ——查詢向量有一半來自新聞（H_t），新聞跟逐字稿都是口語敘事文體，在向量空間裡天生
  比較接近，財報因此被結構性排擠，不是每天公平比相關性選出來的。`quota` 讓每個來源
  至少保留幾個名額（例如 {"filing": 1, "transcript": 1}），其餘名額才照相似度自由競爭，
  避免同一種文體壟斷 top-K。預設 `quota=None` 時完全比照原本行為（純相似度排序）。
"""
import numpy as np


def build_query(h_v: np.ndarray, h_t: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """H_v [N_image,N_token,768] + H_t [512,768] -> query [768]。

    各自 mean-pool 後 L2 正規化，再依 alpha 加權合併（alpha 為 H_v 權重）。
    """
    if h_v.ndim < 2:
        raise ValueError(f"H_v 至少需要 token 與 hidden 兩維，實際 shape={h_v.shape}")
    v = h_v.reshape(-1, h_v.shape[-1]).mean(axis=0)
    t = h_t.mean(axis=0)
    v = v / (np.linalg.norm(v) + 1e-8)
    t = t / (np.linalg.norm(t) + 1e-8)
    return alpha * v + (1 - alpha) * t


def _select_with_quota(hits: list[tuple[str, str, float, str | None]], k: int,
                        quota: dict[str, int]) -> list[tuple[str, str, float, str | None]]:
    """hits 已依相似度由高至低排序（candidate pool，通常比 k 大很多）。

    先讓每個 quota 裡的來源，各自挑自己來源裡分數最高、還沒被選過的 chunk，
    數量最多到 quota 給的門檻（該來源候選不足就有多少選多少，不硬湊）；
    剩下名額（k - 已選數量）不分來源，按整體分數由高到低自由競爭遞補。
    最後依分數重新排序（維持「相似度由高至低」的既有輸出契約）。
    """
    selected: list[tuple[str, str, float, str | None]] = []
    selected_ids: set[str] = set()

    for source, min_n in quota.items():
        picked = 0
        for h in hits:
            if picked >= min_n:
                break
            if h[3] != source or h[0] in selected_ids:
                continue
            selected.append(h)
            selected_ids.add(h[0])
            picked += 1

    for h in hits:
        if len(selected) >= k:
            break
        if h[0] in selected_ids:
            continue
        selected.append(h)
        selected_ids.add(h[0])

    selected.sort(key=lambda h: h[2], reverse=True)
    return selected[:k]


def retrieve(h_v: np.ndarray, h_t: np.ndarray, db, text_encoder,
             k: int = 3, alpha: float = 0.5,
             quota: dict[str, int] | None = None) -> tuple[np.ndarray, list[str]]:
    """回傳 (H_r [K, 512, 768], retrieved_chunk_ids)。

    quota 給定時（例如 {"filing": 1, "transcript": 1}），會先對更大的候選池（pool_n）
    搜尋，確保各來源有機會被挑到，不會因為 k 太小、候選池又被單一來源塞滿而挑不到。
    """
    if quota:
        pool_n = db.index.ntotal if hasattr(db, "index") else k * 20
        hits = db.search(build_query(h_v, h_t, alpha), pool_n)
        hits = _select_with_quota(hits, k, quota)
    else:
        hits = db.search(build_query(h_v, h_t, alpha), k)

    h_r = np.zeros((k, 512, 768), dtype=np.float32)
    chunk_ids = []
    for i, hit in enumerate(hits):
        chunk_id, text = hit[0], hit[1]
        h_r[i] = text_encoder.encode(text)
        chunk_ids.append(chunk_id)
    chunk_ids += ["PAD"] * (k - len(chunk_ids))
    return h_r, chunk_ids


def fake_h_r(k: int = 3, shape=(512, 768), seed: int | None = None) -> tuple[np.ndarray, list[str]]:
    """第一階段假資料：隨機 H_r 與假 chunk_ids。"""
    rng = np.random.default_rng(seed)
    h_r = rng.standard_normal((k, *shape)).astype(np.float32)
    return h_r, [f"FAKE_CHUNK_{i:03d}" for i in range(k)]
