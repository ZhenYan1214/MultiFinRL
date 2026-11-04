"""Cross-Modal Transformer：H_v / H_t / H_r -> Z_fused。

- 三組向量各自線性投影到 d_model，加上模態 embedding，沿序列軸串接，
  進 multi-head self-attention 融合層，CLS token 池化輸出 Z_fused。
- 輸出 Z_fused 為每日一筆的固定維度向量（config.yaml 的 fusion.z_dim），
  是第二年 RL 的 state。
- H_v 保留每張圖的 197 個 token，並加入可學習的圖別 slot embedding；為控制其餘
  序列長度，H_t 與 H_r 先各自 mean-pool
  成段落級 token；C 可自行改成更精細的做法，但 Z_fused 維度定案後不可變。

L_ground 相關（docs/decisions.md #100，皆為選配，預設關閉時行為與 checkpoint 格式完全不變）：
- r_meta：H_r 每段 chunk 的距今天數、來源、是否 PAD（grounding/retrieval_meta.py）。
  有傳入時 PAD chunk 會被 mask 掉，不再被注意到。
- decay_bias（①）：每個來源一個可學習的 λ ≥ 0（softplus），所有 query 對 chunk token 的
  注意力分數減去 λ_src·log(1+天數)（decay_basis="age"）或 λ_src·被取代次數
  （decay_basis="superseded"，見 grounding/retrieval_meta.py），λ 本身可直接解讀。
- age_embedding（②）：天數分桶（0–7/8–30/31–90/91–180/>180 天），每桶一個可學習向量加到
  chunk token 上，讓模型知道文件多舊、自行決定怎麼用；初始化為 0，訓練起點等同沒加。
- return_attn：回傳最後一層 CLS 對各 chunk 的注意力（各 head 平均、dropout 前的機率），
  即 L_ground 的 α̂。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module_c_fusion.grounding.retrieval_meta import N_AGE_BUCKETS, N_SOURCES, age_bucket


class _EncoderLayer(nn.TransformerEncoderLayer):
    """與 nn.TransformerEncoderLayer 相同（參數名稱一致，舊 checkpoint 可直接載入），差別：

    - 一律走一般路徑（不走推論 fast path），確保訓練與推論套用同一個 attention bias；
    - record_cls=True 時，記錄 CLS 對所有 token 的注意力機率（各 head 平均、dropout 前）。
    """

    record_cls = False
    cls_attn = None

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
        if self.record_cls:
            self.cls_attn = self._cls_attention(x, attn_mask)
        return super()._sa_block(x, attn_mask, key_padding_mask, is_causal)

    def _cls_attention(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """x [B,L,d] -> CLS（位置 0）對每個 token 的注意力 [B,L]，與 self_attn 內部計算一致。"""
        mha = self.self_attn
        B, L, d = x.shape
        H = mha.num_heads
        hd = d // H
        w, b = mha.in_proj_weight, mha.in_proj_bias
        q = F.linear(x[:, :1], w[:d], b[:d]).view(B, 1, H, hd).transpose(1, 2)        # [B,H,1,hd]
        k = F.linear(x, w[d:2 * d], b[d:2 * d]).view(B, L, H, hd).transpose(1, 2)    # [B,H,L,hd]
        logits = q @ k.transpose(-1, -2) / math.sqrt(hd)                            # [B,H,1,L]
        if attn_mask is not None:
            logits = logits + attn_mask.view(B, H, L, L)[:, :, :1, :]
        return logits.softmax(dim=-1).mean(dim=1)[:, 0]                             # [B,L]


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class CrossModalTransformer(nn.Module):
    def __init__(self, d_in: int = 768, d_model: int = 768, n_heads: int = 8,
                 n_layers: int = 2, z_dim: int = 768, k: int = 3,
                 n_vision_inputs: int = 2, decay_bias: bool = False,
                 age_embedding: bool = False, decay_init: float = 0.1,
                 decay_basis: str = "age"):
        super().__init__()
        self.proj_v = nn.Linear(d_in, d_model)
        self.proj_t = nn.Linear(d_in, d_model)
        self.proj_r = nn.Linear(d_in, d_model)
        self.vision_slot_emb = nn.Embedding(n_vision_inputs, d_model)
        # 模態 embedding：0=CLS, 1=vision, 2=text, 3=retrieval
        self.modality_emb = nn.Embedding(4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = _EncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, z_dim)
        self.k = k
        self.n_heads = n_heads
        self.n_vision_inputs = n_vision_inputs
        if decay_basis not in ("age", "superseded"):
            raise ValueError(f"decay_basis 只能是 age 或 superseded，收到 {decay_basis}")
        self.decay_bias = decay_bias
        self.decay_basis = decay_basis
        self.age_embedding = age_embedding
        if decay_bias:
            # λ = softplus(decay_raw[src])，每個來源（10-K/10-Q/8-K/法說會…）各自學
            self.decay_raw = nn.Parameter(torch.full((N_SOURCES,), _inv_softplus(decay_init)))
        if age_embedding:
            self.age_emb = nn.Embedding(N_AGE_BUCKETS, d_model)
            nn.init.zeros_(self.age_emb.weight)

    @property
    def uses_retrieval_meta(self) -> bool:
        return self.decay_bias or self.age_embedding

    def decay_lambdas(self) -> torch.Tensor | None:
        return F.softplus(self.decay_raw) if self.decay_bias else None

    def forward(self, h_v: torch.Tensor, h_t: torch.Tensor, h_r: torch.Tensor,
                r_meta: dict | None = None, return_attn: bool = False):
        """h_v [B,I,P,768], h_t [B,512,768], h_r [B,K,512,768] -> Z_fused [B,z_dim]。

        r_meta：{"age": [B,K] float, "sup": [B,K] float（decay_basis="superseded" 時需要）,
                 "src": [B,K] long, "pad": [B,K] bool}；
        return_attn=True 時回傳 (Z_fused, CLS 對 K 段 chunk 的注意力 [B,K])。
        """
        if h_v.ndim != 4 or h_v.size(1) != self.n_vision_inputs:
            raise ValueError(
                f"H_v 應為 [B,{self.n_vision_inputs},P,hidden]，實際 shape={tuple(h_v.shape)}"
            )
        if self.uses_retrieval_meta and r_meta is None:
            raise ValueError("此模型啟用了 decay_bias／age_embedding，forward 必須傳入 r_meta")
        B = h_v.size(0)
        t_v = self.proj_v(h_v)                       # [B,I,P,d]
        slot = self.vision_slot_emb.weight[None, :, None, :]
        t_v = (t_v + slot).flatten(1, 2)             # [B,I*P,d]
        t_t = self.proj_t(h_t.mean(dim=1, keepdim=True))   # [B,1,d] 段落級
        t_r = self.proj_r(h_r.mean(dim=2))           # [B,K,d]  每份文件一個 token

        t_v = t_v + self.modality_emb.weight[1]
        t_t = t_t + self.modality_emb.weight[2]
        t_r = t_r + self.modality_emb.weight[3]
        if self.age_embedding:
            t_r = t_r + self.age_emb(age_bucket(r_meta["age"]))
        cls = self.cls_token.expand(B, -1, -1) + self.modality_emb.weight[0]

        seq = torch.cat([cls, t_v, t_t, t_r], dim=1)  # [B, 1+I*P+1+K, d]
        mask = self._retrieval_mask(r_meta, B, seq.size(1)) if r_meta is not None else None

        last = self.encoder.layers[-1]
        last.record_cls = return_attn
        try:
            out = self.encoder(seq, mask=mask)
            doc_attn = last.cls_attn[:, -self.k:] if return_attn else None
        finally:
            last.record_cls, last.cls_attn = False, None
        z = self.head(out[:, 0])                      # CLS -> Z_fused
        return (z, doc_attn) if return_attn else z

    def _retrieval_mask(self, r_meta: dict, B: int, L: int) -> torch.Tensor:
        """加在注意力分數上的 [B*heads, L, L] 浮點 mask：只動「對 chunk token 的那 K 欄」。"""
        age, src, pad = r_meta["age"], r_meta["src"], r_meta["pad"]
        cols = torch.zeros(B, self.k, device=age.device, dtype=torch.float32)
        if self.decay_bias:
            staleness = (r_meta["sup"].float() if self.decay_basis == "superseded"
                         else torch.log1p(age.float()))
            cols = cols - F.softplus(self.decay_raw)[src] * staleness
        cols = cols.masked_fill(pad, float("-inf"))
        mask = torch.zeros(B, L, L, device=age.device, dtype=torch.float32)
        mask[:, :, L - self.k:] = cols[:, None, :]
        return mask.repeat_interleave(self.n_heads, dim=0)


def build_model(cfg: dict, decay_bias: bool = False, age_embedding: bool = False,
                decay_basis: str = "age") -> CrossModalTransformer:
    """從 config.yaml 建模；decay_bias／age_embedding／decay_basis 由 fusion/train.py 的 CLI 決定。"""
    f = cfg["fusion"]
    return CrossModalTransformer(
        d_model=f["d_model"], n_heads=f["n_heads"],
        n_layers=f["n_layers"], z_dim=f["z_dim"], k=cfg["rag"]["top_k"],
        n_vision_inputs=len(cfg["chart"]["vision_inputs"]),
        decay_bias=decay_bias, age_embedding=age_embedding, decay_basis=decay_basis,
    )
