"""訓練融合模型並輸出 Z_fused。

- 基礎版：以情緒分類為訓練目標（cross-entropy），端到端訓練融合層。
  QLoRA / 對比損失（L_align + L_ground + L_belief）為後續強化，介面不變。
- 產出：Z_fused 到 data/outputs/z_fused/{TICKER}/{date}.npy，
  checkpoint 到 data/outputs/checkpoints/fusion.pt。

用法：
    python -m module_c_fusion.fusion.train --fake --n 32     # 第一階段：假向量測通
    python -m module_c_fusion.fusion.train --ticker AAPL     # 第二階段：B 的真實向量
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from shared import paths, schemas
from shared.utils import load_config, read_json
from module_c_fusion.fusion.model import build_model

LABEL_TO_ID = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}


def load_day(ticker: str, date: str):
    """讀 B 的一天向量，回傳 (h_v, h_t, h_r) numpy。"""
    d = paths.vector_dir(ticker, date)
    index = read_json(d / "index.json")
    schemas.validate_vector_index(index)
    return (np.load(d / "H_v.npy"), np.load(d / "H_t.npy"), np.load(d / "H_r.npy"))


def load_dataset(ticker: str):
    """列出 B 已產出的所有日期，配上 A 的 label（沒有 label 的日期跳過）。"""
    days = []
    for d in sorted((paths.VECTORS / ticker).iterdir()):
        if not (d / "index.json").exists():
            continue
        date = d.name
        label_file = paths.daily_json(ticker, date)
        if not label_file.exists():
            continue
        days.append((date, LABEL_TO_ID[read_json(label_file)["label"]]))
    return days


def make_fake_batch(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    h_v = rng.standard_normal((n, 197, 768)).astype(np.float32)
    h_t = rng.standard_normal((n, 512, 768)).astype(np.float32)
    h_r = rng.standard_normal((n, k, 512, 768)).astype(np.float32)
    y = rng.integers(0, 3, n)
    return h_v, h_t, h_r, y


def train(model, batches, epochs: int, lr: float, device: str):
    """batches: [(h_v, h_t, h_r, y)]，皆為 numpy。"""
    model.to(device).train()
    clf_head = nn.Linear(model.head.out_features, 3).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(clf_head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        total = 0.0
        for h_v, h_t, h_r, y in batches:
            h_v = torch.from_numpy(h_v).to(device)
            h_t = torch.from_numpy(h_t).to(device)
            h_r = torch.from_numpy(h_r).to(device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            z = model(h_v, h_t, h_r)
            loss = loss_fn(clf_head(z), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[train] epoch {ep + 1}/{epochs} loss={total / len(batches):.4f}")
    return clf_head


def export_z_fused(model, ticker: str, days, device: str, batch: int = 8):
    """對所有日期輸出 Z_fused 到 data/outputs/z_fused/。"""
    model.eval()
    out_dir = paths.OUTPUTS / "z_fused" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for date, _y in days:
            h_v, h_t, h_r = load_day(ticker, date)
            z = model(
                torch.from_numpy(h_v[None]).to(device),
                torch.from_numpy(h_t[None]).to(device),
                torch.from_numpy(h_r[None]).to(device),
            )[0].cpu().numpy()
            np.save(out_dir / f"{date}.npy", z)
    print(f"[train] Z_fused x{len(days)} -> {out_dir}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--n", type=int, default=32, help="--fake 時樣本數")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = cfg["rag"]["top_k"]
    model = build_model(cfg)

    if args.fake:
        h_v, h_t, h_r, y = make_fake_batch(args.n, k, cfg["seed"])
        batches = [
            (h_v[i:i + args.batch], h_t[i:i + args.batch], h_r[i:i + args.batch], y[i:i + args.batch])
            for i in range(0, args.n, args.batch)
        ]
        train(model, batches, args.epochs, args.lr, device)
        print("[train] FAKE 流程測通：三個向量進、Z_fused 出、分類 loss 有下降即可")
    else:
        days = load_dataset(args.ticker)
        if not days:
            raise SystemExit("找不到 B 的向量，先跑 module_b_encoder.generate_vectors")
        batches = []
        for i in range(0, len(days), args.batch):
            chunk = days[i:i + args.batch]
            arrs = [load_day(args.ticker, d) for d, _ in chunk]
            batches.append((
                np.stack([a[0] for a in arrs]),
                np.stack([a[1] for a in arrs]),
                np.stack([a[2] for a in arrs]),
                np.array([y for _, y in chunk]),
            ))
        train(model, batches, args.epochs, args.lr, device)
        export_z_fused(model, args.ticker, days, device)

    ckpt = paths.OUTPUTS / "checkpoints" / "fusion.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[train] checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
