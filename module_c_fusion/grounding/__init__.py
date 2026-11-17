"""L_align／L_ground（計畫書 3.5 節）相關工具。

- alpha：L_align、α*（cosine → softmax）、L_ground = KL(α* ‖ α̂) 與驗證指標。
- alpha_star_rag1：預先算好 H_t 與每段 RAG-1 定義的 cosine（Q13 選項 A，凍結向量），存成答案卷。
"""
