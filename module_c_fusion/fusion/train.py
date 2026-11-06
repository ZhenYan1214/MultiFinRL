"""訓練融合層：L_align＋L_ground 先在 Decoder 之前訓練，輸出 Z_fused（decisions.md #114–#117）。

    L = λ1·L_align + λ2·L_ground  （＋選配的 λ_ce·CE 代打，預設 0）

- 輸入：B 的 H_v（兩張圖）、H_t（K 線文字敘述）、RAG-1 的 7 段定義（data/vectors/_patterns/patterns.npz）、
  α* 預設直接用敘述的型態標籤（--alpha_star_space label，#131）；frozen 時才讀 data/labels/grounding/{T}_rag1.json。
- 新聞與舊 H_r 不再進融合層（#110、#111）。
- 組 batch 時同一批不放同型態的日子（L_align 公式不動，從資料端處理，#114）；所以 batch 上限是型態數（6）。
- 前 rag1.retrieve_warmup_epochs 個 epoch 共用空間還沒對齊，RAG-1 先放全部 7 段；之後改取前 K 段。
- 只用 Train 更新參數，Validation 選最佳 epoch（λ1·L_align + λ2·L_ground 最低），Test 不參與。
- 這一階段沒有 L_belief：Z_fused（CLS -> head）只靠注意力層間接被訓練；正式的 Z_fused 由
  decoder/train.py 聯合訓練時的 L_belief 更新。要先看 Z_fused 有沒有資訊，可加 --ce_weight 用 3 類 CE 代打。
- 產出：checkpoint data/outputs/checkpoints/fusion_{T}[_{tag}].pt、Z_fused 到
  data/outputs/z_fused[_{tag}]/{T}/{date}.npy（並彙整索引）、報告 metrics/fusion_train_report_{T}[_{tag}].json。
  報告附「輸入敏感度」：遮掉 H_t、歸零 H_v 時 Z_fused 變多少，以及 Z_fused 兩兩 cosine（檢查是否塌成常數）。

用法：
    python -m module_c_fusion.fusion.train --fake --n 64                       # 假資料測通
    python -m module_c_fusion.fusion.train --ticker AAPL
    python -m module_c_fusion.fusion.train --tickers AAPL NVDA                 # 多股聯合訓練
    python -m module_c_fusion.fusion.train --ticker MSFT --apply_checkpoint data/outputs/checkpoints/fusion_AAPL_NVDA.pt
    python -m module_c_fusion.fusion.train --ticker AAPL --mask_ht --tag mask_ht     # 遮掉 H_t 消融
    python -m module_c_fusion.fusion.train --ticker AAPL --ground_weight 0 --tag align_only
"""
import argparse
import copy
import datetime as dt
import json
import random
import zlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import paths, schemas
from shared.patterns import PATTERN_IDS
from shared.temporal_split import load_temporal_manifest, split_rows
from shared.utils import load_config, read_json, write_json
from module_c_fusion.fusion.model import build_model
from module_c_fusion.grounding.alpha import (align_loss, align_top1, ground_loss, ground_stats,
                                             ground_target)

LABEL_TO_ID = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
PATTERN_TO_ID = {p: i for i, p in enumerate(PATTERN_IDS)}
TRAIN_LOG = paths.OUTPUTS / "logs" / "train_log.jsonl"


@dataclass(frozen=True)
class Day:
    ticker: str
    date: str
    pattern: int      # K 線敘述的型態（PATTERN_IDS 索引）
    label: int        # A 的 3 類漲跌（只有 CE 代打與診斷會用到）


@dataclass
class LossConfig:
    lambda_align: float = 1.0
    lambda_ground: float = 1.0
    ce_weight: float = 0.0
    align_tau: float = 0.07
    ground_tau: float = 0.05
    alpha_star_space: str = "label"
    ground_smoothing: float = 0.1


# ---------------------------------------------------------------- 資料
def load_days(ticker: str) -> list[Day]:
    """B 已產出 H_v＋H_t（K 線敘述）、且 A 有 label 的日期。只有 H_v 的日子（還沒有敘述）跳過。"""
    root = paths.VECTORS / ticker
    if not root.exists():
        return []
    days = []
    for d in sorted(root.iterdir()):
        if not (d / "index.json").exists() or not (d / "H_t.npy").exists():
            continue
        index = read_json(d / "index.json")
        schemas.validate_vector_index(index)
        label_file = paths.daily_json(ticker, d.name)
        if not label_file.exists():
            continue
        days.append(Day(ticker, d.name, PATTERN_TO_ID[index["caption"]["pattern"]],
                        LABEL_TO_ID[read_json(label_file)["label"]]))
    return days


def split_days(ticker: str, days: list[Day]) -> tuple[dict, dict[str, list[Day]]]:
    """沿用唯一 manifest，要求 B 的向量完整覆蓋 manifest 的日期。"""
    manifest = load_temporal_manifest(ticker)
    rows = split_rows([(d.date, d) for d in days], manifest, require_all_manifest_dates=True)
    return manifest, {name: [d for _, d in group] for name, group in rows.items()}


def protocol_days(groups: dict[str, list[Day]]) -> list[Day]:
    """含兩段 gap，排除 manifest 外的舊檔，供輸出 Z_fused。"""
    return sorted((d for group in groups.values() for d in group), key=lambda d: (d.ticker, d.date))


class VectorStore:
    """依日期從磁碟讀 H_v／H_t（每個 batch 才讀，不整批放進記憶體）與 α* 原始分數。"""

    def __init__(self, max_caption_tokens: int, alpha_cos: dict | None = None):
        self.max_tokens = max_caption_tokens
        self.alpha_cos = alpha_cos or {}

    def fetch(self, days: list[Day], device: str, ablate_vision: bool = False) -> dict:
        h_v, h_t = [], []
        for d in days:
            vd = paths.vector_dir(d.ticker, d.date)
            h_v.append(np.load(vd / "H_v.npy"))
            h_t.append(np.load(vd / "H_t.npy"))
        return self._pack(days, np.stack(h_v), np.stack(h_t), device, ablate_vision)

    def _pack(self, days, h_v, h_t, device, ablate_vision):
        mask = np.abs(h_t).sum(-1) > 0
        length = int(min(self.max_tokens, max(1, mask.sum(1).max())))
        if ablate_vision:
            h_v = np.zeros_like(h_v)
        cos = None
        if self.alpha_cos:
            cos = torch.from_numpy(np.stack([self.alpha_cos[(d.ticker, d.date)] for d in days])).to(device)
        return {
            "h_v": torch.from_numpy(h_v).to(device),
            "h_t": torch.from_numpy(np.ascontiguousarray(h_t[:, :length])).to(device),
            "t_mask": torch.from_numpy(mask[:, :length]).to(device),
            "cos": cos,
            "pattern": torch.tensor([d.pattern for d in days], device=device),
            "label": torch.tensor([d.label for d in days], device=device),
            "tickers": [d.ticker for d in days],
        }


class FakeStore(VectorStore):
    """--fake 與單元測試用：每一天的向量由 (ticker, date) 決定的 seed 產生，不讀磁碟。"""

    def __init__(self, max_caption_tokens: int = 32, n_patterns: int = len(PATTERN_IDS)):
        super().__init__(max_caption_tokens)
        self.n_patterns = n_patterns

    def _rng(self, d: Day):
        return np.random.default_rng(zlib.crc32(f"{d.ticker}|{d.date}".encode()))

    def fetch(self, days, device, ablate_vision=False):
        h_v, h_t = [], []
        for d in days:
            rng = self._rng(d)
            h_v.append(rng.standard_normal((2, 197, 768)).astype(np.float32))
            t = np.zeros((self.max_tokens, 768), dtype=np.float32)
            t[: int(rng.integers(8, self.max_tokens))] = rng.standard_normal(768).astype(np.float32)
            h_t.append(t)
        batch = self._pack(days, np.stack(h_v), np.stack(h_t), device, ablate_vision)
        batch["cos"] = torch.from_numpy(np.stack([self._rng(d).uniform(0.7, 0.95, self.n_patterns)
                                                  for d in days]).astype(np.float32)).to(device)
        return batch


def load_kb_tensors(device: str, fake: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """RAG-1 定義 -> (kb_h [7,Lr,768], kb_mask [7,Lr])，Lr 裁到最長那段的長度。"""
    if fake:
        rng = np.random.default_rng(0)
        h = np.zeros((len(PATTERN_IDS), 64, 768), dtype=np.float32)
        h[:, :48] = rng.standard_normal((len(PATTERN_IDS), 48, 768))
    else:
        from module_b_encoder.rag.pattern_kb import load_kb
        h = load_kb()["H_r_all"]
    mask = np.abs(h).sum(-1) > 0
    length = int(mask.sum(1).max())
    return (torch.from_numpy(np.ascontiguousarray(h[:, :length])).to(device),
            torch.from_numpy(mask[:, :length]).to(device))


def pattern_distinct_batches(days: list[Day], batch_size: int, seed: int | None) -> list[list[Day]]:
    """同一個 batch 裡每種型態最多一天（L_align 的負樣本不會是同型態的日子）。

    每次從剩最多的幾個型態各取一天，型態分布不均時也會平均用完；seed=None 代表不打亂（驗證用）。
    最後剩下的日子可能只能湊成很小的 batch，batch 只有 1 天時 L_align 會略過。
    """
    pools = defaultdict(list)
    for d in days:
        pools[d.pattern].append(d)
    rng = random.Random(seed)
    for pool in pools.values():
        if seed is not None:
            rng.shuffle(pool)
    batches = []
    size = min(batch_size, len(PATTERN_IDS))
    while any(pools.values()):
        order = sorted((p for p in pools if pools[p]), key=lambda p: (-len(pools[p]), p))
        batches.append([pools[p].pop() for p in order[:size]])
    if seed is not None:
        rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------- 訓練
def batch_losses(model, batch: dict, kb, k: int, lc: LossConfig, ce_heads=None,
                 mask_ht: bool = False) -> tuple[torch.Tensor, dict]:
    out = model(batch["h_v"], batch["h_t"], batch["t_mask"], *kb, k=k, mask_ht=mask_ht)
    logs = {"n": batch["h_v"].size(0)}
    total = out["z"].sum() * 0.0
    la = align_loss(out["g_v"], out["g_t"], lc.align_tau)
    if la is not None:
        total = total + lc.lambda_align * la
        logs["align"] = la.item()
        logs["align_n"] = logs["n"]
        logs["align_top1"] = align_top1(out["g_v"], out["g_t"])
    if out["alpha_hat"] is not None:
        target, valid = ground_target(lc.alpha_star_space, out, batch, lc.ground_tau, lc.ground_smoothing)
        lg = ground_loss(out["alpha_hat"], target, valid)
        if lg is not None:
            total = total + lc.lambda_ground * lg
        logs.update({f"ground_{key}": v for key, v in ground_stats(out["alpha_hat"], target, valid).items()})
    if ce_heads is not None and lc.ce_weight > 0:
        ce = sum(F.cross_entropy(ce_heads[t](out["z"][i:i + 1]), batch["label"][i:i + 1])
                 for i, t in enumerate(batch["tickers"])) / logs["n"]
        total = total + lc.ce_weight * ce
        logs["ce"] = ce.item()
    idx, pattern = out["idx"], batch["pattern"]
    logs["retrieval_hit1"] = int((idx[:, 0] == pattern).sum())
    logs["retrieval_hitk"] = int((idx == pattern[:, None]).any(-1).sum())
    logs["objective"] = total.item()
    return total, logs


def _accumulate(acc: dict, logs: dict) -> None:
    for key, v in logs.items():
        acc[key] = acc.get(key, 0) + (v * logs["n"] if key in ("objective", "ce") else
                                      v * logs.get("align_n", 0) if key == "align" else v)


def _summary(acc: dict) -> dict:
    n = max(acc.get("n", 0), 1)
    out = {"n": acc.get("n", 0), "objective": round(acc.get("objective", 0) / n, 6)}
    if acc.get("align_n"):
        out["L_align"] = round(acc["align"] / acc["align_n"], 6)
        out["align_top1_in_batch"] = round(acc["align_top1"] / acc["align_n"], 4)
    if acc.get("ground_n"):
        out["L_ground"] = round(acc["ground_kl_sum"] / acc["ground_n"], 6)
        out["ground_top1"] = round(acc["ground_top1_hits"] / acc["ground_n"], 4)
    if acc.get("ground_rows"):
        out["ground_coverage"] = round(acc.get("ground_n", 0) / acc["ground_rows"], 4)
    if "ce" in acc:
        out["ce"] = round(acc["ce"] / n, 6)
    out["retrieval_hit1"] = round(acc.get("retrieval_hit1", 0) / n, 4)
    out["retrieval_hitk"] = round(acc.get("retrieval_hitk", 0) / n, 4)
    return out


def evaluate(model, days, store, kb, k, lc, device, batch_size, ce_heads=None, mask_ht=False) -> dict:
    model.eval()
    acc: dict = {}
    with torch.no_grad():
        for chunk in pattern_distinct_batches(days, batch_size, seed=None):
            _, logs = batch_losses(model, store.fetch(chunk, device), kb, k, lc, ce_heads, mask_ht)
            _accumulate(acc, logs)
    return _summary(acc)


def train_with_validation(model, train_days, val_days, store, kb, *, epochs: int, lr: float,
                          batch_size: int, device: str, lc: LossConfig, top_k: int,
                          warmup_epochs: int = 0, seed: int = 0, mask_ht: bool = False):
    """只用 Train 更新權重；以 Validation 的 objective（λ1·L_align + λ2·L_ground [+ CE]）選最佳 epoch。"""
    model.to(device)
    tickers = sorted({d.ticker for d in train_days})
    ce_heads = nn.ModuleDict({t: nn.Linear(model.head.out_features, 3) for t in tickers}).to(device) \
        if lc.ce_weight > 0 else None
    params = list(model.parameters()) + (list(ce_heads.parameters()) if ce_heads is not None else [])
    optimizer = torch.optim.AdamW(params, lr=lr)
    n_def = kb[0].size(0)
    history, best = [], (float("inf"), 0, None)
    for epoch in range(1, epochs + 1):
        k = n_def if epoch <= warmup_epochs else top_k
        model.train()
        acc: dict = {}
        batches = pattern_distinct_batches(train_days, batch_size, seed=seed + epoch)
        for i, chunk in enumerate(batches, 1):
            loss, logs = batch_losses(model, store.fetch(chunk, device), kb, k, lc, ce_heads, mask_ht)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _accumulate(acc, logs)
            if i % 50 == 0 or i == len(batches):
                print(f"[train] epoch {epoch}/{epochs} batch {i}/{len(batches)} objective={logs['objective']:.4f}",
                      end="\r" if i < len(batches) else "\n", flush=True)
        val = evaluate(model, val_days, store, kb, top_k, lc, device, batch_size, ce_heads, mask_ht)
        row = {"epoch": epoch, "k": k, "train": _summary(acc), "validation": val}
        history.append(row)
        print(f"[train] epoch {epoch}: train={row['train']}")
        print(f"[train]          val  ={val}")
        if val["objective"] < best[0]:
            best = (val["objective"], epoch, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    print(f"[train] 採用 epoch {best[1]}：val objective={best[0]:.4f}")
    return history, best[1]


# ---------------------------------------------------------------- 輸出與診斷
def export_z_fused(model, days, store, kb, k, device, batch: int = 16, tag: str | None = None,
                   mask_ht: bool = False) -> None:
    model.eval()
    out_dirs = {}
    with torch.no_grad():
        for i in range(0, len(days), batch):
            chunk = days[i:i + batch]
            b = store.fetch(chunk, device)
            z = model(b["h_v"], b["h_t"], b["t_mask"], *kb, k=k, mask_ht=mask_ht,
                      record_alpha=False)["z"].cpu().numpy()
            for d, zi in zip(chunk, z):
                out = out_dirs.setdefault(d.ticker, paths.z_fused_dir(tag) / d.ticker)
                out.mkdir(parents=True, exist_ok=True)
                np.save(out / f"{d.date}.npy", zi)
    for t, out in out_dirs.items():
        print(f"[train] Z_fused {t} x{sum(d.ticker == t for d in days)} -> {out}")


def input_sensitivity(model, days, store, kb, k, device, n: int = 128) -> dict:
    """Z_fused 對輸入的敏感度：遮掉 H_t、歸零 H_v 時 ‖ΔZ‖/‖Z‖，以及 Z 兩兩 cosine 平均（≈1 代表塌成常數）。"""
    model.eval()
    sample = days[:: max(1, len(days) // n)][:n]
    zs, d_ht, d_hv = [], [], []
    with torch.no_grad():
        for i in range(0, len(sample), 16):
            chunk = sample[i:i + 16]
            b = store.fetch(chunk, device)
            args = (b["h_v"], b["h_t"], b["t_mask"], *kb)
            z = model(*args, k=k, record_alpha=False)["z"]
            z_ht = model(*args, k=k, mask_ht=True, record_alpha=False)["z"]
            z_hv = model(torch.zeros_like(b["h_v"]), *args[1:], k=k, record_alpha=False)["z"]
            norm = z.norm(dim=-1).clamp_min(1e-8)
            d_ht.append(((z - z_ht).norm(dim=-1) / norm).cpu())
            d_hv.append(((z - z_hv).norm(dim=-1) / norm).cpu())
            zs.append(z.cpu())
    z = F.normalize(torch.cat(zs), dim=-1)
    sim = z @ z.T
    off = sim[~torch.eye(len(z), dtype=torch.bool)]
    return {"n": len(z), "rel_change_mask_ht": round(float(torch.cat(d_ht).mean()), 4),
            "rel_change_zero_hv": round(float(torch.cat(d_hv).mean()), 4),
            "z_pairwise_cosine_mean": round(float(off.mean()), 4)}


def log_run(record: dict) -> None:
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), **record}
    with open(TRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_alpha_cos(tickers: list[str]) -> dict:
    from module_c_fusion.grounding.alpha_star_rag1 import load_alpha_star_cos
    out = {}
    for t in tickers:
        out.update({(t, d): v for d, v in load_alpha_star_cos(t).items()})
    return out


# ---------------------------------------------------------------- CLI
def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--tickers", nargs="+", default=None, help="多股聯合訓練")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--n", type=int, default=64, help="--fake 時樣本數")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=6, help="上限 7（同 batch 不放同型態）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--align_weight", type=float, default=cfg["loss"]["lambda_align"], help="λ1")
    ap.add_argument("--ground_weight", type=float, default=cfg["loss"]["lambda_ground"], help="λ2")
    ap.add_argument("--ce_weight", type=float, default=0.0, help="選配：3 類 CE 代打（附屬診斷用），預設 0")
    ap.add_argument("--align_tau", type=float, default=cfg["loss"]["align_tau"])
    ap.add_argument("--ground_tau", type=float, default=cfg["loss"]["ground_tau"])
    ap.add_argument("--alpha_star_space", choices=["label", "frozen", "projected"],
                    default=cfg["loss"]["alpha_star_space"],
                    help="α*：label＝敘述的型態標籤（預設，#131）；frozen／projected＝cosine 版本（對照）")
    ap.add_argument("--ground_smoothing", type=float, default=cfg["loss"]["ground_smoothing"])
    ap.add_argument("--top_k", type=int, default=cfg["rag1"]["top_k"])
    ap.add_argument("--warmup_epochs", type=int, default=cfg["rag1"]["retrieve_warmup_epochs"])
    ap.add_argument("--mask_ht", action="store_true", help="消融：融合層遮掉 H_t（L_ground 隨之關閉）")
    ap.add_argument("--ablate_vision", action="store_true", help="消融：H_v 歸零")
    ap.add_argument("--apply_checkpoint", default=None, help="套用既有權重產生 Z_fused，不訓練")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default=None, help="實驗組名稱；checkpoint／Z_fused／報告另存")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--weighted", action="store_true", help="（舊參數，已不使用，保留給 run_pipeline 相容）")
    args = ap.parse_args()
    if not 1 <= args.batch <= len(PATTERN_IDS):
        raise SystemExit(f"--batch 必須在 1–{len(PATTERN_IDS)} 之間（同一 batch 不放同型態）")

    seed = cfg["seed"] if args.seed is None else args.seed
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lc = LossConfig(args.align_weight, args.ground_weight, args.ce_weight, args.align_tau,
                    args.ground_tau, args.alpha_star_space, args.ground_smoothing)
    setup = {**vars(args), "seed": seed, "loss": lc.__dict__}
    print(f"[train] device={device} loss={lc}")
    model = build_model(cfg)
    kb = load_kb_tensors(device, fake=args.fake)
    tickers = args.tickers or [args.ticker]

    if args.fake:
        store = FakeStore()
        rng = np.random.default_rng(seed)
        days = [Day("_FAKE", f"d{i:04d}", int(rng.integers(0, len(PATTERN_IDS))), int(rng.integers(0, 3)))
                for i in range(args.n)]
        cut = int(len(days) * 0.8)
        history, best = train_with_validation(
            model, days[:cut], days[cut:], store, kb, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch, device=device, lc=lc, top_k=args.top_k,
            warmup_epochs=args.warmup_epochs, seed=seed, mask_ht=args.mask_ht)
        print(f"[train] 輸入敏感度：{input_sensitivity(model, days, store, kb, args.top_k, device)}")
        ckpt = paths.OUTPUTS / "checkpoints" / "fusion_fake.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print("[train] FAKE 流程測通（loss 數字沒有意義）")
        return

    groups_by_ticker = {}
    for t in tickers:
        days = load_days(t)
        if not days:
            raise SystemExit(f"找不到 {t} 的向量（H_v＋H_t），先跑 module_b_encoder.generate_vectors --ticker {t}")
        groups_by_ticker[t] = split_days(t, days)[1]
    alpha_cos = load_alpha_cos(tickers) if lc.alpha_star_space == "frozen" and not args.mask_ht else {}
    store = VectorStore(cfg["fusion"]["max_caption_tokens"], alpha_cos)
    ckpt = paths.fusion_checkpoint_path(tickers, args.tag)
    export = [d for t in tickers for d in protocol_days(groups_by_ticker[t])]

    if args.apply_checkpoint:
        model.load_state_dict(torch.load(args.apply_checkpoint, map_location=device))
        model.to(device)
        history, best, ckpt = [], None, args.apply_checkpoint
    else:
        train_days = [d for t in tickers for d in groups_by_ticker[t]["train"]]
        val_days = [d for t in tickers for d in groups_by_ticker[t]["validation"]]
        print(f"[train] {tickers}: train={len(train_days)} val={len(val_days)} 天")
        history, best = train_with_validation(
            model, train_days, val_days, store, kb, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch, device=device, lc=lc, top_k=args.top_k,
            warmup_epochs=args.warmup_epochs, seed=seed, mask_ht=args.mask_ht)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"[train] checkpoint -> {ckpt}")

    export_z_fused(model, export, store, kb, args.top_k, device, tag=args.tag, mask_ht=args.mask_ht)
    held_out = [d for t in tickers for d in groups_by_ticker[t]["validation"] + groups_by_ticker[t]["test"]]
    sensitivity = input_sensitivity(model, held_out, store, kb, args.top_k, device)
    print(f"[train] 輸入敏感度（Validation＋Test 抽樣）：{sensitivity}")

    from module_c_fusion.fusion.consolidate import build_index, save_index
    for t in tickers:
        index_data = build_index(t, tag=args.tag)
        if index_data:
            save_index(t, index_data, checkpoint_path=ckpt, run_id=args.run_id, tag=args.tag)
    suffix = f"_{args.tag}" if args.tag else ""
    report = {
        "run_id": args.run_id, "protocol": "strict_temporal_v1",
        "tickers": tickers, "ticker": tickers[0] if len(tickers) == 1 else None,
        "split_manifests": {t: str(paths.temporal_split_path(t)) for t in tickers},
        "split_counts_by_ticker": {t: {k: len(v) for k, v in groups_by_ticker[t].items()} for t in tickers},
        # 單股時給扁平欄位，第一階段報告（reporting/pipeline_report.py）用來做一致性檢查
        "split_manifest": str(paths.temporal_split_path(tickers[0])) if len(tickers) == 1 else None,
        "split_counts": ({k: len(v) for k, v in groups_by_ticker[tickers[0]].items()}
                         if len(tickers) == 1 else None),
        "setup": setup, "best_epoch": best, "history": history,
        "input_sensitivity": sensitivity, "checkpoint": str(ckpt),
    }
    write_json(report, paths.OUTPUTS / "metrics" / f"fusion_train_report_{'_'.join(tickers)}{suffix}.json")
    log_run({"mode": "apply_checkpoint" if args.apply_checkpoint else "real", "tickers": tickers,
             "best_epoch": best, "setup": setup, "input_sensitivity": sensitivity,
             "final_validation": history[-1]["validation"] if history else None})


if __name__ == "__main__":
    main()
