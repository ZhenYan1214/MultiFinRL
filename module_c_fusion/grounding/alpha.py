"""L_align 與 L_ground（計畫書 3.5 (1)(2)，公式不動；只換了對象，decisions.md #114、#115）。

L_align = −log [ exp(⟨H_v, H_t⟩/τ) / Σ_{t'} exp(⟨H_v, H_{t'}⟩/τ) ]
    H_v、H_t 是融合層之前、經共用投影 P_v、P_t 後的池化向量（g_v、g_t），⟨·,·⟩ 取 L2 正規化後的內積
    （計畫書轉錄註：⟨H_v H_t⟩ 應為 pooled 向量的內積）；分母的 t' 是同一個 batch 裡其他天的敘述。
    同型態的日子不放同一個 batch，從資料端處理（fusion/train.py 的 pattern_distinct_batches）。

L_ground = Σ_i α*_i log(α*_i / α̂_i)
    α̂：融合層裡 H_t token 對當天 K 段 RAG-1 定義的注意力（fusion/model.py）。
    α*（Q13 定案為 label，decisions.md #131）：
        label（預設）= 直接用 K 線敘述的型態標籤（Haiku 判的那一類）：當天取到的 K 段裡，敘述型態那段拿
                       1−ε，其他段平分 ε（label smoothing，loss.ground_smoothing）。正確那段沒被取到的天
                       不算 L_ground（回報 ground_coverage），不改融合層的輸入。
        frozen       = 舊做法（Q5 選 b）：投影前凍結 FinBERT 池化向量的 cosine 經 softmax(cos/τ)，
                       預先算在 data/labels/grounding/{T}_rag1.json。AAPL 實測只有 18.7% 的天 cosine 最高
                       的定義跟敘述型態一致（FinBERT 分不出型態語意），保留只當對照。
        projected    = 投影後的 g_t 與 P_r(定義) 的 cosine，訓練時即時算、detach。
"""
import torch
import torch.nn.functional as F

EPS = 1e-8


def align_loss(g_v: torch.Tensor, g_t: torch.Tensor, tau: float) -> torch.Tensor | None:
    """g_v、g_t [B,d] -> 平均 L_align；batch 只有 1 筆時沒有負樣本，回傳 None。"""
    if g_v.size(0) < 2:
        return None
    logits = F.normalize(g_v, dim=-1) @ F.normalize(g_t, dim=-1).T / tau     # [B,B]
    target = torch.arange(g_v.size(0), device=g_v.device)
    return F.cross_entropy(logits, target)


@torch.no_grad()
def align_top1(g_v: torch.Tensor, g_t: torch.Tensor) -> int:
    """batch 內，每張圖最接近的敘述是不是自己那天的（驗證用，回傳命中數）。"""
    sim = F.normalize(g_v, dim=-1) @ F.normalize(g_t, dim=-1).T
    return int((sim.argmax(dim=-1) == torch.arange(g_v.size(0), device=g_v.device)).sum())


def alpha_star(cos_all: torch.Tensor, idx: torch.Tensor, tau: float) -> torch.Tensor:
    """cos_all [B,N]（H_t 對全部 7 段定義）、idx [B,K]（當天取到的段）-> α* [B,K]。"""
    return torch.softmax(cos_all.gather(1, idx) / tau, dim=-1)


def projected_cos(g_t: torch.Tensor, r_def: torch.Tensor) -> torch.Tensor:
    """Q13 選項 B：投影後的 cosine [B,N]，detach（α* 是目標，不回傳梯度）。"""
    return (F.normalize(g_t, dim=-1) @ F.normalize(r_def, dim=-1).T).detach()


def ground_kl(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """逐筆 KL(target ‖ pred) [B]。"""
    terms = target * (target.clamp_min(EPS).log() - pred.clamp_min(EPS).log())
    return torch.where(target > 0, terms, torch.zeros_like(terms)).sum(dim=-1)


def label_target(pattern: torch.Tensor, idx: torch.Tensor, smoothing: float):
    """pattern [B]（敘述型態的 PATTERN_IDS 索引）、idx [B,K] -> (α* [B,K], valid [B])。
    valid = 敘述型態那段有在當天取到的 K 段裡；沒有的那天 α* 沒有定義，不算 L_ground。"""
    hit = idx == pattern[:, None]
    k = idx.size(1)
    on, off = (1.0 - smoothing, smoothing / (k - 1)) if k > 1 else (1.0, 0.0)
    target = torch.where(hit, torch.tensor(on, device=idx.device), torch.tensor(off, device=idx.device))
    return target.float(), hit.any(dim=-1)


def ground_target(space: str, out: dict, batch: dict, tau: float, smoothing: float):
    """依 alpha_star_space 算 α* -> (α* [B,K], valid [B])。"""
    idx = out["idx"]
    if space == "label":
        return label_target(batch["pattern"], idx, smoothing)
    cos = batch.get("cos") if space == "frozen" else projected_cos(out["g_t"], out["r_def"])
    if cos is None:
        raise SystemExit("α* 用 frozen 向量時需要 data/labels/grounding/{T}_rag1.json（grounding/alpha_star_rag1.py）")
    return alpha_star(cos, idx, tau), torch.ones(idx.size(0), dtype=torch.bool, device=idx.device)


def ground_loss(alpha_hat: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor | None:
    """valid 那幾天的平均 KL(α* ‖ α̂)；整個 batch 都沒有 valid 時回傳 None。"""
    if not bool(valid.any()):
        return None
    return ground_kl(target[valid], alpha_hat[valid]).mean()


@torch.no_grad()
def ground_stats(alpha_hat: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    """加總統計：n（有算 L_ground 的天數）、rows（全部天數）、kl_sum、top1_hits（α̂ 最高的段 = α* 最高的段）。"""
    a, t = alpha_hat[valid], target[valid]
    return {"n": int(valid.sum()), "rows": int(valid.numel()),
            "kl_sum": float(ground_kl(t, a).sum()) if len(a) else 0.0,
            "top1_hits": int((a.argmax(-1) == t.argmax(-1)).sum()) if len(a) else 0}
