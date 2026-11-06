"""Decoder 聯合訓練：L = λ1·L_align + λ2·L_ground + λ3·L_belief，一次 backward
（計畫書 3.5 (4)；decisions.md #117、#118）。

每一步：
    1. 融合層（可訓練）讀 H_v、H_t（K 線敘述）、RAG-1 定義 -> Z_fused、L_align、L_ground
    2. Decoder：[4 個 prefix（由 Z_fused 投影）; RAG-2 候選短版文字; y_belief] -> 只對 y_belief 算 L_belief
    3. 三個 Loss 加權相加一次 backward：梯度更新 LoRA、Projector、融合層與共用投影 P_v/P_t/P_r；
       ViT、FinBERT、LLaMA 本體凍結
組 batch 時同一批不放同型態的日子（L_align）。舊版是讀預先存好的 Z_fused，L_belief 到不了融合層。

兩個版本（#118）：
    主線     --rag2（預設）：context 約 1,100–1,300 token，等有 GPU 再跑
    選配     --no_rag2：不放候選文字，y_belief 上限 512 token，Mac 可以先跑

沿用唯一的 strict temporal manifest：Train 更新參數、Validation 選最佳 epoch、Test 留給 evaluate.py。

前置：
    1. module_b_encoder.generate_vectors（H_v、H_t）、module_b_encoder.rag.pattern_kb（RAG-1）
    2. （α* 預設用敘述的型態標籤，不需要前置；--alpha_star_space frozen 才要先跑 grounding.alpha_star_rag1）
    3. module_b_encoder.rag.rag2（主線才需要）
    4. module_c_fusion.decoder.generate_y_belief（y_belief 答案卷）
    5. 建議先跑 module_c_fusion.fusion.train 當融合層的起點（--fusion_init，預設自動找 fusion_{T}.pt）

用法：
    python -m module_c_fusion.decoder.train --ticker AAPL --epochs 3                 # 主線（GPU）
    python -m module_c_fusion.decoder.train --ticker AAPL --no_rag2 --epochs 3       # 選配：Mac 512 版本
    python -m module_c_fusion.decoder.train --ticker AAPL --tiny_random --limit 24 --epochs 1   # smoke test
    python -m module_c_fusion.decoder.train --ticker AAPL --resume
Mac（MPS）請帶 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 HF_HUB_OFFLINE=1。
"""
import argparse
import random

import torch
from transformers import get_cosine_schedule_with_warmup

from shared import paths
from shared.temporal_split import load_temporal_manifest, split_rows
from shared.utils import load_config, read_json, write_json
from module_c_fusion.fusion.model import build_model
from module_c_fusion.fusion.train import (LossConfig, VectorStore, load_alpha_cos, load_days,
                                          load_kb_tensors, pattern_distinct_batches)
from module_c_fusion.grounding.alpha import align_loss, ground_loss, ground_target

CONTEXT_HEADER = "Documents available on the analysis date:\n"
BELIEF_HEADER = "\nBelief state:\n"


def checkpoint_dirs(tag: str | None, rag2: bool) -> tuple:
    name = "decoder" + ("" if rag2 else "_512") + (f"_{tag}" if tag else "")
    root = paths.OUTPUTS / "checkpoints"
    return root / name, root / f"{name}_latest"


def load_samples(ticker: str, rag2: bool, limit: int | None = None) -> list[dict]:
    """(Day, y_belief, RAG-2 context) 的交集，依日期排序。"""
    y_path = paths.y_belief_path(ticker)
    if not y_path.exists():
        raise SystemExit(f"找不到 y_belief，先跑 python -m module_c_fusion.decoder.generate_y_belief --ticker {ticker}")
    y_belief = read_json(y_path)
    samples = []
    for day in load_days(ticker):
        if day.date not in y_belief:
            continue
        context = ""
        if rag2:
            p = paths.rag2_path(ticker, day.date)
            if not p.exists():
                raise SystemExit(f"找不到 {p}，先跑 python -m module_b_encoder.rag.rag2 --ticker {ticker}")
            context = read_json(p)["context"]
        samples.append({"day": day, "y": y_belief[day.date], "context": context})
    if not samples:
        raise SystemExit("向量跟 y_belief 沒有交集")
    return samples[:limit] if limit else samples


def split_samples(samples: list[dict], ticker: str) -> dict[str, list[dict]]:
    manifest = load_temporal_manifest(ticker)
    groups = split_rows([(s["day"].date, s) for s in samples], manifest, require_all_manifest_dates=False)
    return {name: [s for _, s in rows] for name, rows in groups.items()}


def collate_text(samples: list[dict], tokenizer, max_ctx: int, max_y: int, device) -> dict:
    """候選文字＋y_belief 串成一段、右側 padding；labels 只保留 y_belief（候選與 padding 為 -100）。"""
    rows = []
    for s in samples:
        ctx_ids = []
        if s["context"]:
            ctx_ids = tokenizer(CONTEXT_HEADER + s["context"], add_special_tokens=False,
                                truncation=True, max_length=max_ctx)["input_ids"]
        head = tokenizer(BELIEF_HEADER, add_special_tokens=False)["input_ids"] if ctx_ids else []
        y_ids = tokenizer(s["y"], add_special_tokens=False, truncation=True,
                          max_length=max_y - 1)["input_ids"] + [tokenizer.eos_token_id]
        ids = ctx_ids + head + y_ids
        rows.append((ids, [-100] * (len(ctx_ids) + len(head)) + y_ids))
    length = max(len(ids) for ids, _ in rows)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(rows), length), pad, dtype=torch.long)
    labels = torch.full((len(rows), length), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.long)
    for i, (ids, lab) in enumerate(rows):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        mask[i, :len(ids)] = 1
    return {"input_ids": input_ids.to(device), "labels": labels.to(device), "attention_mask": mask.to(device)}


def context_ids(tokenizer, context: str, max_ctx: int, device):
    """evaluate.py 生成時用：候選文字＋「Belief state:」標頭的 token（沒有候選時為 None）。"""
    if not context:
        return None
    ids = tokenizer(CONTEXT_HEADER + context, add_special_tokens=False, truncation=True,
                    max_length=max_ctx)["input_ids"] + tokenizer(BELIEF_HEADER, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids], device=device)


def joint_step(fusion, decoder, store, kb, batch, lc: LossConfig, lam_belief: float, top_k: int,
               tokenizer, max_ctx: int, max_y: int, device, shuffle_z: bool = False) -> tuple:
    """一個 batch 的三個 Loss；回傳 (加權總和, 各項數值)。shuffle_z=True 時把 Z_fused 在 batch 內換位置
    （評估「打亂 Z_fused」用）。"""
    days = [s["day"] for s in batch]
    b = store.fetch(days, device)
    out = fusion(b["h_v"], b["h_t"], b["t_mask"], *kb, k=top_k)
    z = out["z"]
    if shuffle_z and z.size(0) > 1:
        z = z.roll(1, dims=0)
    text = collate_text(batch, tokenizer, max_ctx, max_y, device)
    l_belief = decoder(z, text["input_ids"], text["attention_mask"], text["labels"])
    total = lam_belief * l_belief
    logs = {"belief": l_belief.item()}
    la = align_loss(out["g_v"], out["g_t"], lc.align_tau)
    if la is not None:
        total = total + lc.lambda_align * la
        logs["align"] = la.item()
    if out["alpha_hat"] is not None:
        target, valid = ground_target(lc.alpha_star_space, out, b, lc.ground_tau, lc.ground_smoothing)
        lg = ground_loss(out["alpha_hat"], target, valid)
        if lg is not None:
            total = total + lc.lambda_ground * lg
            logs["ground"] = lg.item()
    logs["total"] = total.item()
    return total, logs


def _mean_logs(rows: list[dict]) -> dict:
    keys = {k for r in rows for k in r}
    return {k: round(sum(r[k] for r in rows if k in r) / max(1, sum(k in r for r in rows)), 6) for k in sorted(keys)}


def evaluate_split(fusion, decoder, store, kb, samples, args, lc, cfg, tokenizer, device,
                   shuffle_z: bool = False) -> dict:
    fusion.eval()
    decoder.eval()
    rows = []
    with torch.no_grad():
        for chunk in pattern_distinct_batches([s["day"] for s in samples], args.batch_size, seed=None):
            by_day = {s["day"]: s for s in samples}
            _, logs = joint_step(fusion, decoder, store, kb, [by_day[d] for d in chunk], lc,
                                 args.belief_weight, args.top_k, tokenizer, args.max_ctx, args.max_y,
                                 device, shuffle_z)
            rows.append(logs)
    fusion.train()
    decoder.train()
    return _mean_logs(rows)


def save_checkpoint(out_dir, decoder, llm, fusion, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.projector.state_dict(), out_dir / "projector.pt")
    torch.save(fusion.state_dict(), out_dir / "fusion.pt")
    llm.save_pretrained(out_dir / "lora_adapter")
    if meta.get("tiny_random"):   # 小模型的 backbone 是隨機初始化，評估時要用同一份權重
        torch.save(llm.state_dict(), out_dir / "tiny_full.pt")
    write_json(meta, out_dir / "config.json")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--base_model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--rag2", dest="rag2", action="store_true", default=True, help="主線：放 RAG-2 候選（預設）")
    ap.add_argument("--no_rag2", dest="rag2", action="store_false", help="選配：不放候選（512 版本）")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2, help="上限 7（同 batch 不放同型態）")
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA 與 Projector")
    ap.add_argument("--fusion_lr", type=float, default=1e-4, help="融合層與共用投影")
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--n_prefix_tokens", type=int, default=4)
    ap.add_argument("--max_ctx", type=int, default=1024, help="RAG-2 候選文字的 token 上限")
    ap.add_argument("--max_y", type=int, default=512, help="y_belief 的 token 上限（中文敘述，不要再調低）")
    ap.add_argument("--align_weight", type=float, default=cfg["loss"]["lambda_align"])
    ap.add_argument("--ground_weight", type=float, default=cfg["loss"]["lambda_ground"])
    ap.add_argument("--belief_weight", type=float, default=cfg["loss"]["lambda_belief"])
    ap.add_argument("--alpha_star_space", choices=["label", "frozen", "projected"], default=cfg["loss"]["alpha_star_space"])
    ap.add_argument("--top_k", type=int, default=cfg["rag1"]["top_k"])
    ap.add_argument("--fusion_init", default="auto",
                    help="融合層起點 checkpoint；auto = 有 fusion_{T}.pt 就用，none = 隨機初始化")
    ap.add_argument("--freeze_fusion", action="store_true", help="消融：融合層不更新（只有 L_belief 訓練 LoRA）")
    ap.add_argument("--tiny_random", action="store_true", help="smoke test：隨機初始化的小 LLaMA，CPU 可跑")
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 天（smoke test 用）")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", action="store_true", help="從 *_latest/ 接續訓練")
    args = ap.parse_args()
    from shared.patterns import PATTERN_IDS
    if not 1 <= args.batch_size <= len(PATTERN_IDS):
        raise SystemExit(f"--batch_size 必須在 1–{len(PATTERN_IDS)} 之間（同 batch 不放同型態）")

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    lc = LossConfig(args.align_weight, args.ground_weight, 0.0, cfg["loss"]["align_tau"],
                    cfg["loss"]["ground_tau"], args.alpha_star_space, cfg["loss"]["ground_smoothing"])
    out_dir, latest_dir = checkpoint_dirs(args.tag if not args.tiny_random else f"tiny{'_' + args.tag if args.tag else ''}",
                                          args.rag2)
    state_path = latest_dir / "resume_state.json"
    report_path = paths.OUTPUTS / "metrics" / f"{out_dir.name}_train_report_{args.ticker}.json"

    samples = load_samples(args.ticker, args.rag2)
    groups = split_samples(samples, args.ticker)
    if args.limit:
        groups = {k: v[: max(2, args.limit * len(v) // len(samples))] for k, v in groups.items()}
    train_rows, val_rows, test_rows = groups["train"], groups["validation"], groups["test"]
    print(f"[decoder.train] {args.ticker} rag2={'on' if args.rag2 else 'off'}: train={len(train_rows)} "
          f"val={len(val_rows)} test={len(test_rows)}（依 manifest，不打散）")
    if not train_rows or not val_rows:
        raise SystemExit("train／validation 有空集合，先補齊 y_belief 或向量")

    # ---- 模型 ----
    from module_c_fusion.decoder.model import (ZFusedDecoder, load_backbone, load_backbone_for_resume,
                                               load_tiny_random)
    if args.tiny_random:
        tokenizer, llm = load_tiny_random(args.base_model)
    elif args.resume:
        tokenizer, llm = load_backbone_for_resume(latest_dir, args.base_model)
    else:
        tokenizer, llm = load_backbone(args.base_model)
    device = llm.device if not args.tiny_random else torch.device("cpu")
    decoder = ZFusedDecoder(llm, tokenizer, z_dim=cfg["fusion"]["z_dim"], n_prefix_tokens=args.n_prefix_tokens)
    fusion = build_model(cfg).to(device)
    init = None
    if args.resume:
        decoder.projector.load_state_dict(torch.load(latest_dir / "projector.pt", map_location=device))
        init = latest_dir / "fusion.pt"
    elif args.fusion_init == "auto":
        candidate = paths.fusion_checkpoint_path(args.ticker)
        init = candidate if candidate.exists() else None
    elif args.fusion_init != "none":
        init = args.fusion_init
    if init:
        fusion.load_state_dict(torch.load(init, map_location=device))
    print(f"[decoder.train] 融合層起點：{init or '隨機初始化'}；融合層{'凍結' if args.freeze_fusion else '一起訓練'}")
    if args.freeze_fusion:
        for p in fusion.parameters():
            p.requires_grad_(False)

    kb = load_kb_tensors(device)
    alpha_cos = load_alpha_cos([args.ticker]) if lc.alpha_star_space == "frozen" else {}
    store = VectorStore(cfg["fusion"]["max_caption_tokens"], alpha_cos)

    dec_params = [p for p in decoder.parameters() if p.requires_grad]
    groups_opt = [{"params": dec_params, "lr": args.lr}]
    if not args.freeze_fusion:
        groups_opt.append({"params": list(fusion.parameters()), "lr": args.fusion_lr})
    optimizer = torch.optim.AdamW(groups_opt)

    start_epoch, best_val, best_epoch, history = 0, float("inf"), None, []
    if args.resume:
        state = read_json(state_path)
        start_epoch, best_val, best_epoch, history = (state["epoch"], state["best_val"],
                                                      state["best_epoch"], state["history"])
        if start_epoch >= args.epochs:
            raise SystemExit(f"已完成 {start_epoch} 個 epoch，>= --epochs {args.epochs}")
    by_day = {s["day"]: s for s in train_rows}
    n_batches = len(pattern_distinct_batches([s["day"] for s in train_rows], args.batch_size, seed=0))
    total_steps = n_batches * (args.epochs - start_epoch)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    meta = {"ticker": args.ticker, "rag2": args.rag2, "max_ctx": args.max_ctx,
            "max_y": args.max_y, "n_prefix_tokens": args.n_prefix_tokens, "top_k": args.top_k,
            "base_model": args.base_model, "tiny_random": args.tiny_random, "loss": lc.__dict__,
            "lambda_belief": args.belief_weight, "fusion_init": str(init) if init else None,
            "freeze_fusion": args.freeze_fusion}

    decoder.train()      # backbone 預設 eval；不切 train 的話 gradient checkpointing／LoRA dropout 不會啟用
    fusion.train()
    trainable = dec_params + ([] if args.freeze_fusion else list(fusion.parameters()))
    for epoch in range(start_epoch, args.epochs):
        rows = []
        batches = pattern_distinct_batches([s["day"] for s in train_rows], args.batch_size, seed=cfg["seed"] + epoch)
        for step, chunk in enumerate(batches):
            optimizer.zero_grad()
            loss, logs = joint_step(fusion, decoder, store, kb, [by_day[d] for d in chunk], lc,
                                    args.belief_weight, args.top_k, tokenizer, args.max_ctx, args.max_y, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            rows.append(logs)
            print(f"  epoch {epoch + 1}/{args.epochs} step {step + 1}/{len(batches)} " +
                  " ".join(f"{k}={v:.4f}" for k, v in logs.items()), flush=True)
        val = evaluate_split(fusion, decoder, store, kb, val_rows, args, lc, cfg, tokenizer, device)
        row = {"epoch": epoch + 1, "train": _mean_logs(rows), "validation": val}
        history.append(row)
        print(f"[decoder.train] epoch {epoch + 1}: train={row['train']} val={val}")
        if val["total"] < best_val:
            best_val, best_epoch = val["total"], epoch + 1
            save_checkpoint(out_dir, decoder, llm, fusion, {**meta, "epoch": best_epoch})
            print(f"[decoder.train] val total 創新低 {best_val:.4f}，最佳 checkpoint -> {out_dir}")
        save_checkpoint(latest_dir, decoder, llm, fusion, {**meta, "epoch": epoch + 1})
        write_json({"epoch": epoch + 1, "best_val": best_val, "best_epoch": best_epoch, "history": history},
                   state_path)
        write_json({"protocol": "strict_temporal_v1", **meta,
                    "split_manifest": str(paths.temporal_split_path(args.ticker)),
                    "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
                    "best_epoch": best_epoch, "best_val_total": best_val, "history": history}, report_path)
    print(f"[decoder.train] 完成：最佳 epoch {best_epoch}（val total={best_val:.4f}）-> {out_dir}")
    print(f"[decoder.train] 報告 -> {report_path}")


if __name__ == "__main__":
    main()
