"""
跨模態融合層（docs/architecture/architecture.html，decisions.md #114–#117）。

    [CLS ; H_t（K 線敘述 token）; 394 個圖 token ; K 個 RAG-1 定義 token] -> 2 層 × 8 頭 -> CLS -> Z_fused

- 共用投影 P_v、P_t、P_r（計畫書 3.2：H = LN(Z P)）：同一組投影同時給 L_align、L_ground 與融合層用，
  兩個 Loss 的梯度才會改到融合層拿到的輸入（#116）。ViT、FinBERT 凍結，輸入是 B 存好的向量。
- RAG-1 檢索（#112）：query = K 線圖與成交量圖 token 經 P_v 後的平均（g_v），key = N 段定義經 P_t 後的
  池化向量；兩者在 L_align 對齊過的共用空間算 cosine，取前 K 段。key 用 P_t 而不是 P_r，是因為定義跟
  K 線敘述是同一個 FinBERT 空間，而 L_align 只對齊 P_v 與 P_t；P_r 還沒被任何 Loss 對齊時，用它算的
  cosine 沒有意義。選段不可微分（top-K），不影響梯度。
- 每段定義經 P_r 後池化成 1 個 token 進融合層（「K 個 RAG-1 定義 token」）。
- α̂（L_ground 的模型端，#115）：最後一層裡 H_t token 對 K 個定義 token 的注意力（各 head 平均、
  dropout 前的機率），對 H_t token 取平均後在 K 段之間正規化。
- mask_ht=True：把 H_t token 從序列中遮掉（key padding），做「遮掉 H_t」消融；此時沒有 α̂。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderLayer(nn.TransformerEncoderLayer):
    """與 nn.TransformerEncoderLayer 相同（參數名稱一致），差別：

    - 一律走一般路徑（不走推論 fast path），訓練與推論同一套計算；
    - record=(q_index, k_index) 時，記錄 q_index 這些位置對 k_index 這些位置的注意力機率
      （softmax 仍在全部 key 上做，含 padding mask；各 head 平均、dropout 前）。
    """

    record = None
    recorded = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        if self.record is not None:
            self.recorded = self._attention(x, key_padding_mask, *self.record)
        return super()._sa_block(x, attn_mask, key_padding_mask, is_causal)

    def _attention(self, x, key_padding_mask, q_index, k_index):
        """x [B,L,d] -> 注意力機率 [B, len(q_index), len(k_index)]，跟 self_attn 內部計算一致。"""
        mha = self.self_attn
        B, L, d = x.shape
        H = mha.num_heads
        hd = d // H
        w, b = mha.in_proj_weight, mha.in_proj_bias
        q = F.linear(x[:, q_index], w[:d], b[:d]).view(B, -1, H, hd).transpose(1, 2)    # [B,H,Q,hd]
        k = F.linear(x, w[d:2 * d], b[d:2 * d]).view(B, L, H, hd).transpose(1, 2)      # [B,H,L,hd]
        logits = q @ k.transpose(-1, -2) / math.sqrt(hd)                              # [B,H,Q,L]
        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            else:
                logits = logits + key_padding_mask[:, None, None, :]
        return logits.softmax(dim=-1).mean(dim=1)[:, :, k_index]                      # [B,Q,K]


def _projection(d_in: int, d_model: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [..., L, d]、mask [..., L] bool（True = 實際 token）-> [..., d]。"""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=-2) / m.sum(dim=-2).clamp_min(1.0)


class CrossModalTransformer(nn.Module):
    def __init__(self, d_in: int = 768, d_model: int = 768, n_heads: int = 8, n_layers: int = 2,
                 z_dim: int = 768, top_k: int = 3, n_vision_inputs: int = 2):
        super().__init__()
        self.P_v = _projection(d_in, d_model)
        self.P_t = _projection(d_in, d_model)
        self.P_r = _projection(d_in, d_model)
        self.vision_slot_emb = nn.Embedding(n_vision_inputs, d_model)
        # 模態 embedding：0=CLS, 1=vision, 2=K 線敘述, 3=RAG-1 定義
        self.modality_emb = nn.Embedding(4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = _EncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, z_dim)
        self.top_k = top_k
        self.n_vision_inputs = n_vision_inputs

    # ---- L_align／L_ground／檢索共用的投影 ----
    def embed_vision(self, h_v: torch.Tensor):
        """h_v [B,I,P,768] -> (圖 token [B,I*P,d], 池化 g_v [B,d])。"""
        if h_v.ndim != 4 or h_v.size(1) != self.n_vision_inputs:
            raise ValueError(f"H_v 應為 [B,{self.n_vision_inputs},P,768]，實際 {tuple(h_v.shape)}")
        t_v = self.P_v(h_v)
        g_v = t_v.mean(dim=(1, 2))
        t_v = (t_v + self.vision_slot_emb.weight[None, :, None, :]).flatten(1, 2)
        return t_v, g_v

    def embed_caption(self, h_t: torch.Tensor, t_mask: torch.Tensor):
        """h_t [B,L,768]、t_mask [B,L] -> (敘述 token [B,L,d], 池化 g_t [B,d])。"""
        t_t = self.P_t(h_t)
        return t_t, masked_mean(t_t, t_mask)

    def embed_kb(self, kb_h: torch.Tensor, kb_mask: torch.Tensor):
        """kb_h [N,Lr,768]、kb_mask [N,Lr] -> (定義 token [N,d]（P_r）, 檢索 key [N,d]（P_t）)。"""
        pooled = masked_mean(kb_h, kb_mask)          # 凍結 FinBERT 的池化向量
        return self.P_r(pooled), self.P_t(pooled)

    @torch.no_grad()
    def retrieve(self, g_v: torch.Tensor, kb_key: torch.Tensor, k: int):
        """cosine(g_v, key) -> (前 k 段的索引 [B,k], 全部分數 [B,N])；不可微分。"""
        scores = F.normalize(g_v, dim=-1) @ F.normalize(kb_key, dim=-1).T
        return scores.topk(k, dim=-1).indices, scores

    def forward(self, h_v: torch.Tensor, h_t: torch.Tensor, t_mask: torch.Tensor,
                kb_h: torch.Tensor, kb_mask: torch.Tensor, k: int | None = None,
                mask_ht: bool = False, record_alpha: bool = True) -> dict:
        """回傳 dict：
            z        [B,z_dim]  Z_fused
            g_v, g_t [B,d]      L_align 用的池化向量（融合層之前）
            r_def    [N,d]      全部定義經 P_r 後的向量（α* 用投影後向量時用）
            idx      [B,K]      RAG-1 這天取到哪幾段（PATTERN_IDS 的索引）
            retrieval_scores [B,N]
            alpha_hat [B,K] 或 None
        """
        B = h_v.size(0)
        k = k or self.top_k
        t_v, g_v = self.embed_vision(h_v)
        t_t, g_t = self.embed_caption(h_t, t_mask)
        r_def, kb_key = self.embed_kb(kb_h, kb_mask)
        idx, scores = self.retrieve(g_v, kb_key, k)
        t_r = r_def[idx]                              # [B,K,d]

        cls = self.cls_token.expand(B, -1, -1) + self.modality_emb.weight[0]
        t_t_in = t_t + self.modality_emb.weight[2]
        t_v_in = t_v + self.modality_emb.weight[1]
        t_r_in = t_r + self.modality_emb.weight[3]
        seq = torch.cat([cls, t_t_in, t_v_in, t_r_in], dim=1)
        Lt, Lv = t_t.size(1), t_v.size(1)
        caption_pad = ~t_mask.bool() if not mask_ht else torch.ones_like(t_mask, dtype=torch.bool)
        pad = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=seq.device),
            caption_pad,
            torch.zeros(B, Lv + k, dtype=torch.bool, device=seq.device),
        ], dim=1)

        last = self.encoder.layers[-1]
        want_alpha = record_alpha and not mask_ht
        q_index = torch.arange(1, 1 + Lt, device=seq.device)
        k_index = torch.arange(1 + Lt + Lv, 1 + Lt + Lv + k, device=seq.device)
        last.record = (q_index, k_index) if want_alpha else None
        try:
            out = self.encoder(seq, src_key_padding_mask=pad)
            att = last.recorded if want_alpha else None              # [B,Lt,K]
        finally:
            last.record, last.recorded = None, None
        alpha_hat = None
        if att is not None:
            m = t_mask.to(att.dtype).unsqueeze(-1)
            att = (att * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # 對 H_t token 平均 -> [B,K]
            alpha_hat = att / att.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return {"z": self.head(out[:, 0]), "g_v": g_v, "g_t": g_t, "r_def": r_def,
                "idx": idx, "retrieval_scores": scores, "alpha_hat": alpha_hat}


def build_model(cfg: dict) -> CrossModalTransformer:
    f = cfg["fusion"]
    return CrossModalTransformer(d_model=f["d_model"], n_heads=f["n_heads"], n_layers=f["n_layers"],
                                 z_dim=f["z_dim"], top_k=cfg["rag1"]["top_k"],
                                 n_vision_inputs=len(cfg["chart"]["vision_inputs"]))
