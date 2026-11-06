"""α* 的 cosine 對照版本：每天 K 線敘述（H_t）與 7 段 RAG-1 定義的 cosine（decisions.md #115）。

11/05 起 α* 預設改用敘述的型態標籤（#131，grounding/alpha.py 的 label_target），這份檔案只在
--alpha_star_space frozen 時使用，或拿來診斷「凍結 FinBERT 分不分得出型態」（AAPL 實測一致率 18.7%）。

用投影前、凍結 FinBERT 的池化向量算，跟訓練無關，可以預先算好存成答案卷：
    data/labels/grounding/{TICKER}_rag1.json
        {"_meta": {"kb_sha256": [...], "encoder": ...},
         date: {"cos": [7 個 cosine，順序 = shared/patterns.PATTERN_IDS], "caption_pattern": "BREAKOUT"}}
訓練時依當天 RAG-1 取到的 K 段，取出對應的 cosine 做 softmax(cos/τ)（grounding/alpha.py）。
選項 B（投影後向量）在訓練時即時算，不需要這份檔案：fusion/train.py --alpha_star_space projected。

跑完印出兩個檢查：α* 最高的定義跟 Haiku 判斷的型態一致的比例，以及 τ 下 α* 的平均最大機率
（cosine 差距很小時 α* 會接近均勻分布，L_ground 學不到東西，要調低 loss.ground_tau）。

用法：
    python -m module_c_fusion.grounding.alpha_star_rag1 --ticker AAPL
"""
import argparse

import numpy as np

from shared import paths, patterns
from shared.utils import load_config, read_json, write_json


def load_alpha_star_cos(ticker: str) -> dict[str, np.ndarray]:
    path = paths.rag1_alpha_star_path(ticker)
    if not path.exists():
        raise SystemExit(f"找不到 α* 答案卷 {path}，先跑 python -m module_c_fusion.grounding.alpha_star_rag1 --ticker {ticker}")
    data = read_json(path)
    from module_b_encoder.rag.pattern_kb import load_kb
    kb_sha = [e["sha256"] for e in load_kb()["meta"].get("entries", [])]
    if kb_sha and data.get("_meta", {}).get("kb_sha256") != kb_sha:
        raise SystemExit(f"{path} 是用舊版的 RAG-1 定義算的，定義改過後要重跑 alpha_star_rag1")
    return {d: np.asarray(v["cos"], dtype=np.float32) for d, v in data.items() if d != "_meta"}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    args = ap.parse_args()

    from module_b_encoder.encoders.text_encoder import masked_mean
    from module_b_encoder.rag.pattern_kb import load_kb
    kb = load_kb()
    keys = kb["pooled"] / np.linalg.norm(kb["pooled"], axis=-1, keepdims=True)
    out = {"_meta": {"kb_sha256": [e["sha256"] for e in kb["meta"]["entries"]],
                     "encoder": kb["meta"]["encoder"], "space": "frozen",
                     "patterns": patterns.PATTERN_IDS}}
    days = sorted(d for d in (paths.VECTORS / args.ticker).iterdir() if (d / "H_t.npy").exists())
    if not days:
        raise SystemExit(f"找不到 {paths.VECTORS / args.ticker} 的 H_t，先跑 module_b_encoder.generate_vectors")
    hits, top_probs = 0, []
    tau = cfg["loss"]["ground_tau"]
    for d in days:
        g = masked_mean(np.load(d / "H_t.npy"))
        cos = keys @ (g / np.linalg.norm(g))
        caption = read_json(d / "index.json")["caption"]
        pattern = caption["pattern"]
        out[d.name] = {"cos": [round(float(x), 6) for x in cos], "caption_pattern": pattern}
        hits += patterns.PATTERN_IDS[int(cos.argmax())] == pattern
        p = np.exp((cos - cos.max()) / tau)
        top_probs.append(float(np.sort(p / p.sum())[-3:][::-1][0]))
    write_json(out, paths.rag1_alpha_star_path(args.ticker))
    print(f"[alpha_star_rag1] {args.ticker}: {len(days)} 天 -> {paths.rag1_alpha_star_path(args.ticker)}")
    print(f"[alpha_star_rag1] cosine 最高的定義 = 敘述的型態：{hits}/{len(days)}（{hits / len(days):.1%}）")
    n = len(patterns.PATTERN_IDS)
    print(f"[alpha_star_rag1] τ={tau} 時 α*（{n} 段）最大機率平均 {np.mean(top_probs):.3f}（越接近 1/{n}≈{1 / n:.3f} 越接近均勻）")


if __name__ == "__main__":
    main()
