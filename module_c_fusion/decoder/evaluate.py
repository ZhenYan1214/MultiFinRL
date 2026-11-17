"""評估聯合訓練後的 Decoder（decisions.md #119）：只用 Test（訓練與選模都沒看過的天）。

    (1) Test loss：L_belief／L_align／L_ground；並跟「打亂 Z_fused」比（同一 batch 內把 Z_fused 換到別天）。
        L_belief 幾乎不變，代表 Decoder 沒在用 Z_fused（舊版就是這樣，見 decisions.md 與 Z_fused 診斷）。
    (2) 實際生成 --n_generate 天：
        - <TREND>／<RISK_LEVEL> 準確率（跟 y_belief 比）
        - 型態一致率（主指標）：生成敘述裡提到的 K 線型態，跟 Haiku 的 K 線敘述型態是否一致；
          另外回報「有提到型態的比例」——y_belief 本身不一定寫型態，提到率低時一致率參考價值有限
        - 打亂 Z_fused（輔助）：同一天換成別天的 Z_fused 再生成一次，看型態／TREND 有沒有跟著變
    (3)（選配，--llm_judge，會花錢）LLM-as-judge 評敘述品質 1–5 分。
    L_belief 下降怎麼解讀，等測試完再決定（#119）；報告裡同時放有無 RAG-2 的設定，方便對照消融。

用法：
    python -m module_c_fusion.decoder.evaluate --ticker AAPL --n_generate 30
    python -m module_c_fusion.decoder.evaluate --ticker AAPL --checkpoint data/outputs/checkpoints/decoder_512
    python -m module_c_fusion.decoder.evaluate --ticker AAPL --n_generate 30 --llm_judge --n_judge 15
"""
import argparse
import json
import re
from pathlib import Path

import torch

from shared import paths
from shared.patterns import PATTERN_IDS, PATTERN_KEYWORDS
from shared.utils import load_config, read_json, write_json
from module_c_fusion.fusion.model import build_model
from module_c_fusion.fusion.train import LossConfig, VectorStore, load_alpha_cos, load_kb_tensors
from module_c_fusion.decoder.train import context_ids, evaluate_split, load_samples, split_samples

TAG_PATTERN = re.compile(r"<TREND>\s*\[(\w+)\]\s*<RISK_LEVEL>\s*\[(\w+)\]\s*(.*)", re.DOTALL)


def parse_tags(text: str):
    """y_belief 格式 -> (trend, risk_level, narrative)；解析不出來回傳 (None, None, "")。"""
    m = TAG_PATTERN.search(text)
    if not m:
        return None, None, ""
    return m.group(1), m.group(2), m.group(3).strip()


def find_pattern(text: str) -> str | None:
    """敘述裡第一個出現的 K 線型態（PATTERN_KEYWORDS 中英文關鍵字）；沒提到回傳 None。
    頭肩底的關鍵字比頭肩頂、突破長，先比對長的避免被短關鍵字吃掉。"""
    lowered = text.lower()
    best = None
    for pid in PATTERN_IDS:
        for kw in sorted(PATTERN_KEYWORDS[pid], key=len, reverse=True):
            pos = lowered.find(kw.lower())
            if pos >= 0 and (best is None or pos < best[0] or (pos == best[0] and len(kw) > best[2])):
                best = (pos, pid, len(kw))
    return best[1] if best else None


def load_decoder(checkpoint_dir: Path, cfg: dict, device_hint=None):
    """讀 decoder/train.py 存的 checkpoint：fusion.pt + projector.pt + lora_adapter/ + config.json。"""
    from peft import PeftModel
    from module_c_fusion.decoder.model import ZFusedDecoder, load_base_only, load_tiny_random
    meta = read_json(checkpoint_dir / "config.json")
    if meta.get("tiny_random"):
        tokenizer, llm = load_tiny_random(meta["base_model"])
        llm.load_state_dict(torch.load(checkpoint_dir / "tiny_full.pt", map_location="cpu"))
        device = torch.device("cpu")
    else:
        tokenizer, base = load_base_only(meta["base_model"])
        llm = PeftModel.from_pretrained(base, checkpoint_dir / "lora_adapter")
        device = llm.device
    decoder = ZFusedDecoder(llm, tokenizer, z_dim=cfg["fusion"]["z_dim"], n_prefix_tokens=meta["n_prefix_tokens"])
    decoder.projector.load_state_dict(torch.load(checkpoint_dir / "projector.pt", map_location=device))
    fusion = build_model(cfg).to(device)
    fusion.load_state_dict(torch.load(checkpoint_dir / "fusion.pt", map_location=device))
    decoder.eval()
    fusion.eval()
    return decoder, fusion, tokenizer, meta, device


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--checkpoint", default=str(paths.OUTPUTS / "checkpoints" / "decoder"))
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--n_generate", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--llm_judge", action="store_true", help="用 LLM API 評分（會花錢），預設關閉")
    ap.add_argument("--n_judge", type=int, default=15)
    ap.add_argument("--judge_provider", choices=["claude", "openai", "deepseek"], default="deepseek")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    decoder, fusion, tokenizer, meta, device = load_decoder(ckpt, cfg)
    samples = load_samples(args.ticker, meta["rag2"])
    test_rows = split_samples(samples, args.ticker)["test"]
    if not test_rows:
        raise SystemExit("test set 是空的")
    print(f"[decoder.evaluate] {args.ticker} {ckpt.name}（rag2={'on' if meta['rag2'] else 'off'}）："
          f"test {len(test_rows)} 天 {test_rows[0]['day'].date} ~ {test_rows[-1]['day'].date}")

    lc = LossConfig(**meta["loss"])
    kb = load_kb_tensors(device)
    store = VectorStore(cfg["fusion"]["max_caption_tokens"],
                        load_alpha_cos([args.ticker]) if lc.alpha_star_space == "frozen" else {})
    targs = argparse.Namespace(batch_size=args.batch_size, belief_weight=meta["lambda_belief"],
                               top_k=meta["top_k"], max_ctx=meta["max_ctx"], max_y=meta["max_y"])

    # ---- (1) Test loss：真的 Z_fused vs 打亂 ----
    real = evaluate_split(fusion, decoder, store, kb, test_rows, targs, lc, cfg, tokenizer, device)
    shuffled = evaluate_split(fusion, decoder, store, kb, test_rows, targs, lc, cfg, tokenizer, device,
                              shuffle_z=True)
    print(f"[decoder.evaluate] test loss：真的 Z_fused {real}；打亂 Z_fused {shuffled}")

    # ---- (2) 生成：TREND/RISK、型態一致率、打亂 Z_fused ----
    captions = {d: v["pattern"] for d, v in _caption_patterns(args.ticker).items()}
    n_gen = min(args.n_generate, len(test_rows))
    rows = test_rows[:n_gen]
    with torch.no_grad():
        b = store.fetch([s["day"] for s in rows], device)
        z_all = torch.cat([fusion(b["h_v"][i:i + 1], b["h_t"][i:i + 1], b["t_mask"][i:i + 1], *kb,
                                  k=meta["top_k"], record_alpha=False)["z"] for i in range(len(rows))])
    records = []
    for i, s in enumerate(rows):
        ctx = context_ids(tokenizer, s["context"], meta["max_ctx"], device)
        gen = decoder.generate(z_all[i:i + 1], ctx, args.max_new_tokens)
        gen_shuf = decoder.generate(z_all[(i + 1) % len(rows)][None], ctx, args.max_new_tokens) if len(rows) > 1 else ""
        gt_trend, gt_risk, gt_narr = parse_tags(s["y"])
        trend, risk, narr = parse_tags(gen)
        trend_s, _, _ = parse_tags(gen_shuf)
        date = s["day"].date
        records.append({
            "date": date, "caption_pattern": captions.get(date), "ground_truth": s["y"], "generated": gen,
            "generated_shuffled_z": gen_shuf, "gt_trend": gt_trend, "gen_trend": trend,
            "gt_risk_level": gt_risk, "gen_risk_level": risk, "gt_narrative": gt_narr, "gen_narrative": narr,
            "gen_pattern": find_pattern(gen), "gen_pattern_shuffled_z": find_pattern(gen_shuf),
            "trend_shuffled_z": trend_s,
        })
        print(f"  {date}: trend {gt_trend}->{trend} | pattern {captions.get(date)}->{records[-1]['gen_pattern']}"
              f" | 打亂 Z：trend {trend_s}, pattern {records[-1]['gen_pattern_shuffled_z']}")

    def rate(pred) -> float | None:
        vals = [pred(r) for r in records]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    mentioned = [r for r in records if r["gen_pattern"]]
    summary = {
        "n_generated": len(records),
        "format_ok_rate": rate(lambda r: r["gen_trend"] is not None),
        "trend_accuracy": rate(lambda r: r["gen_trend"] == r["gt_trend"]),
        "risk_level_accuracy": rate(lambda r: r["gen_risk_level"] == r["gt_risk_level"]),
        "pattern_mention_rate": round(len(mentioned) / len(records), 4),
        "pattern_consistency": rate(lambda r: (r["gen_pattern"] == r["caption_pattern"]) if r["gen_pattern"] else None),
        "shuffled_z": {
            "generated_text_changed": rate(lambda r: r["generated"] != r["generated_shuffled_z"]),
            "trend_changed": rate(lambda r: r["gen_trend"] != r["trend_shuffled_z"]),
            "pattern_changed": rate(lambda r: (r["gen_pattern"] != r["gen_pattern_shuffled_z"])
                                    if (r["gen_pattern"] or r["gen_pattern_shuffled_z"]) else None),
        },
    }
    print(f"[decoder.evaluate] 生成摘要：{json.dumps(summary, ensure_ascii=False)}")

    # ---- (3) LLM-as-judge（選配）----
    judge = None
    if args.llm_judge:
        from module_b_encoder.llm_client import CALLERS, DEFAULT_MODEL
        from module_c_fusion.decoder.judge_prompt import SYSTEM_PROMPT, build_user_prompt
        model_name = DEFAULT_MODEL[args.judge_provider]
        scores = []
        for r in records[: args.n_judge]:
            raw = CALLERS[args.judge_provider](SYSTEM_PROMPT, build_user_prompt(r["gt_narrative"], r["gen_narrative"]),
                                               model_name)
            try:
                obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
                scores.append({"date": r["date"], "score": obj.get("score"), "reason": obj.get("reason", "")})
            except (json.JSONDecodeError, ValueError):
                scores.append({"date": r["date"], "score": None, "reason": "解析失敗"})
        valid = [s["score"] for s in scores if isinstance(s["score"], (int, float))]
        judge = {"provider": args.judge_provider, "model": model_name, "n_valid": len(valid),
                 "avg_score": sum(valid) / len(valid) if valid else None, "details": scores}

    report = {"ticker": args.ticker, "checkpoint": str(ckpt), "setup": meta, "n_test": len(test_rows),
              "test_period": [test_rows[0]["day"].date, test_rows[-1]["day"].date],
              "test_loss": {"real_z": real, "shuffled_z": shuffled}, "generation": summary,
              "llm_judge": judge, "samples": records}
    out = paths.OUTPUTS / "metrics" / f"{ckpt.name}_evaluation_{args.ticker}.json"
    write_json(report, out)
    print(f"[decoder.evaluate] 報告 -> {out}")


def _caption_patterns(ticker: str) -> dict:
    """每天 index.json 記錄的敘述型態（跟 H_t 用的是同一份敘述）。"""
    out = {}
    for d in (paths.VECTORS / ticker).iterdir():
        if (d / "index.json").exists() and (d / "H_t.npy").exists():
            out[d.name] = read_json(d / "index.json")["caption"]
    return out


if __name__ == "__main__":
    main()
