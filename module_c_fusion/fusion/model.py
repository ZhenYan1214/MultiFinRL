"""Cross-Modal Transformer：H_v / H_t / H_r -> Z_fused。

- 三組向量各自線性投影到 d_model，加上模態 embedding，沿序列軸串接，
  進 multi-head self-attention 融合層，CLS token 池化輸出 Z_fused。
- 輸出 Z_fused 為每日一筆的固定維度向量（config.yaml 的 fusion.z_dim），
  是第二年 RL 的 state。
- 為控制序列長度（197 + 512 + 3*512 太長），H_t 與 H_r 先各自 mean-pool
  成段落級 token；C 可自行改成更精細的做法，但 Z_fused 維度定案後不可變。
"""
import torch
import torch.nn as nn


class CrossModalTransformer(nn.Module):
    def __init__(self, d_in: int = 768, d_model: int = 768, n_heads: int = 8,
                 n_layers: int = 2, z_dim: int = 768, k: int = 3):
        super().__init__()
        self.proj_v = nn.Linear(d_in, d_model)
        self.proj_t = nn.Linear(d_in, d_model)
        self.proj_r = nn.Linear(d_in, d_model)
        # 模態 embedding：0=CLS, 1=vision, 2=text, 3=retrieval
        self.modality_emb = nn.Embedding(4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, z_dim)
        self.k = k

    def forward(self, h_v: torch.Tensor, h_t: torch.Tensor, h_r: torch.Tensor) -> torch.Tensor:
        """h_v [B,197,768], h_t [B,512,768], h_r [B,K,512,768] -> Z_fused [B,z_dim]。"""
        B = h_v.size(0)
        t_v = self.proj_v(h_v)                       # [B,197,d]
        t_t = self.proj_t(h_t.mean(dim=1, keepdim=True))   # [B,1,d] 段落級
        t_r = self.proj_r(h_r.mean(dim=2))           # [B,K,d]  每份文件一個 token

        t_v = t_v + self.modality_emb.weight[1]
        t_t = t_t + self.modality_emb.weight[2]
        t_r = t_r + self.modality_emb.weight[3]
        cls = self.cls_token.expand(B, -1, -1) + self.modality_emb.weight[0]

        seq = torch.cat([cls, t_v, t_t, t_r], dim=1)  # [B, 1+197+1+K, d]
        out = self.encoder(seq)
        return self.head(out[:, 0])                   # CLS -> Z_fused


def build_model(cfg: dict) -> CrossModalTransformer:
    """從 config.yaml 建模。"""
    f = cfg["fusion"]
    return CrossModalTransformer(
        d_model=f["d_model"], n_heads=f["n_heads"],
        n_layers=f["n_layers"], z_dim=f["z_dim"], k=cfg["rag"]["top_k"],
    )
