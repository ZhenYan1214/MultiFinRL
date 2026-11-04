"""L_ground（計畫書 3.5 節 Evidence Grounding Loss）相關工具。

- retrieval_meta：從 H_r 的 retrieved_chunk_ids 解析每段 chunk 的來源與「距今天數」。
- prompt / generate_alpha_star：用 LLM（Claude Haiku）對每天檢索到的 chunk 打相關性分數。
- labels：讀取打分結果並對齊到目前向量的 chunk 順序。
- alpha：分數 -> α*、L_ground（KL）與驗證指標。
"""
