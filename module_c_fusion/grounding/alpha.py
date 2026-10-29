"""LLM 分數 -> α*，以及 L_ground = KL(α* ‖ α̂)（計畫書 3.5 節 (2)）。

α̂ 是融合層最後一層 CLS 對 K 段 chunk token 的注意力（各 head 平均、dropout 之前的機率，
fusion/model.py 的 cls_attention）。兩種模式：

- 不含「其他」槽位（null_score=None，預設）：
      α* = softmax(s / T)，α̂ 在 K 段之間重新正規化；三段都 0 分的日子不計入。
  只約束「K 段之間怎麼分」，不約束「分給檢索多少」——後者用 grounding_stats 的 doc_mass 監看。
- 含「其他」槽位（null_score 給數值）：
      α* = softmax([s_1, …, s_K, null_score] / T)，PAD 位置為 0
      α̂ = [CLS→chunk_1, …, CLS→chunk_K, 1 − Σ CLS→chunk]   # 「其他」= 新聞、K 線圖等其餘 token
  三段都不相關的日子，α* 大部分落在「其他」，教模型在證據過時／無關時少依賴檢索；
  同時約束「分給檢索多少」。注意：實測 AAPL 現有 checkpoint 的 CLS 只把約 0.1–0.2% 的
  注意力分給檢索區（其餘在 394 個 K 線圖 token 上），而這個模式的 α* 會要求檢索區拿到
  一半以上，等於強迫大幅改變注意力分配，因此不設為預設（docs/decisions.md #100）。
"""
import torch

EPS = 1e-8


def valid_rows(scores: torch.Tensor, pad: torch.Tensor, has_label: torch.Tensor,
               null_score: float | None) -> torch.Tensor:
    """哪幾筆樣本可以算 L_ground：有標籤、至少一段非 PAD；無「其他」槽位時還要至少一段 > 0 分。"""
    valid = has_label & (~pad).any(dim=-1)
    if null_score is None:
        valid = valid & (scores.masked_fill(pad, 0).amax(dim=-1) > 0)
    return valid


def alpha_star(scores: torch.Tensor, pad: torch.Tensor, temperature: float = 1.0,
               null_score: float | None = None) -> torch.Tensor:
    """scores [B,K]（0–3）-> α* [B,K]（無槽位）或 [B,K+1]（最後一欄是「其他」）。"""
    logits = (scores / temperature).masked_fill(pad, float("-inf"))
    if null_score is not None:
        null = torch.full_like(logits[:, :1], null_score / temperature)
        logits = torch.cat([logits, null], dim=-1)
    return torch.softmax(logits, dim=-1)


def alpha_hat(doc_attn: torch.Tensor, null_score: float | None = None) -> torch.Tensor:
    """doc_attn [B,K]（CLS 對各 chunk 的注意力）-> 與 alpha_star 同形狀的分布。"""
    if null_score is not None:
        other = (1.0 - doc_attn.sum(dim=-1, keepdim=True)).clamp_min(0.0)
        return torch.cat([doc_attn, other], dim=-1)
    return doc_attn / doc_attn.sum(dim=-1, keepdim=True).clamp_min(EPS)


def ground_kl(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """逐筆 KL(target ‖ pred)，target 為 0 的位置不貢獻。回傳 [B]。"""
    terms = target * (target.clamp_min(EPS).log() - pred.clamp_min(EPS).log())
    return torch.where(target > 0, terms, torch.zeros_like(terms)).sum(dim=-1)


def grounding_loss(doc_attn: torch.Tensor, scores: torch.Tensor, pad: torch.Tensor,
                   has_label: torch.Tensor, temperature: float = 1.0,
                   null_score: float | None = None) -> tuple[torch.Tensor, int]:
    """回傳 (有效樣本的平均 L_ground, 有效樣本數)；沒有有效樣本時 loss 為 0（仍可 backward）。"""
    valid = valid_rows(scores, pad, has_label, null_score)
    if not bool(valid.any()):
        return doc_attn.sum() * 0.0, 0
    target = alpha_star(scores[valid], pad[valid], temperature, null_score)
    pred = alpha_hat(doc_attn[valid], null_score)
    return ground_kl(target, pred).mean(), int(valid.sum())


@torch.no_grad()
def grounding_stats(doc_attn: torch.Tensor, scores: torch.Tensor, pad: torch.Tensor,
                    has_label: torch.Tensor, temperature: float = 1.0,
                    null_score: float | None = None) -> dict[str, float]:
    """驗證用的加總統計（呼叫端跨 batch 累加後再除以 n）。

    - kl_sum：KL(α*‖α̂) 加總
    - top1_hits：模型最關注的 chunk 是否落在 LLM 給最高分的 chunk 之一（平手都算對）；
      只統計最高分 > 0 的日子（全部 0 分沒有「最相關」可言）
    - doc_mass_sum：CLS 分給全部 chunk 的注意力總和（檢查檢索區有沒有被犧牲）
    """
    valid = valid_rows(scores, pad, has_label, null_score)
    out = {"n": int(valid.sum()), "kl_sum": 0.0, "top1_n": 0, "top1_hits": 0,
           "doc_mass_sum": 0.0}
    if out["n"] == 0:
        return out
    s, p, a = scores[valid], pad[valid], doc_attn[valid]
    target = alpha_star(s, p, temperature, null_score)
    out["kl_sum"] = float(ground_kl(target, alpha_hat(a, null_score)).sum())
    out["doc_mass_sum"] = float(a.sum(dim=-1).sum())

    masked_scores = s.masked_fill(p, float("-inf"))
    best = masked_scores.amax(dim=-1)
    rated = best > 0
    if bool(rated.any()):
        pick = a.masked_fill(p, float("-inf")).argmax(dim=-1)
        hit = masked_scores.gather(1, pick[:, None]).squeeze(1) == best
        out["top1_n"] = int(rated.sum())
        out["top1_hits"] = int((hit & rated).sum())
    return out
