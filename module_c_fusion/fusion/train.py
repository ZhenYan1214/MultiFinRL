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
import datetime as dt
import json

import numpy as np
import torch    
import torch.nn as nn

from shared import paths, schemas
from shared.utils import load_config, read_json
from module_c_fusion.fusion.model import build_model

LABEL_TO_ID = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
TRAIN_LOG = paths.OUTPUTS / "logs" / "train_log.jsonl"


def log_run(mode: str, ticker: str, n_days: int, epochs: int, batch: int, lr: float,
           losses: list[float], date_range: list[str] | None = None,
           note: str = "") -> None:
    """把這次訓練的參數與每個 epoch 的 loss 附加寫進 train_log.jsonl，一行一筆紀錄。

    不用手動記，每次跑 train.py 都會自動留下一筆，之後要比較不同次執行的結果，
    直接打開這個檔案看，或用 pandas 讀成表格分析。
    """
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,              # "fake" 或 "real"
        "ticker": ticker,
        "n_days": n_days,
        "date_range": date_range,  # real 模式才有；[起始日, 結束日]
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "losses": losses,          # 每個 epoch 結束時的平均 loss，依序排列
        "final_loss": losses[-1] if losses else None,
        "note": note,
    }
    with open(TRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[train] 訓練紀錄 -> {TRAIN_LOG}")


def load_day(ticker: str, date: str, ablate_news: bool = False):
    """讀 B 的一天向量，回傳 (h_v, h_t, h_r) numpy。

    ablate_news=True 時，H_t 讀進來後在記憶體裡直接歸零（不改動磁碟上的檔案）。
    這是 docs/spec_c_accuracy_diagnostics.md 的新聞歸零對照實驗用的，
    只影響這次執行的結果，不會留下任何需要事後還原的殘留狀態。"""
    d = paths.vector_dir(ticker, date)
    index = read_json(d / "index.json")
    schemas.validate_vector_index(index)
    h_v = np.load(d / "H_v.npy")
    h_t = np.load(d / "H_t.npy")
    h_r = np.load(d / "H_r.npy")
    if ablate_news:
        h_t = np.zeros_like(h_t)
    return (h_v, h_t, h_r)


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


def train(model, batches, epochs: int, lr: float, device: str, class_weights=None):
    """batches: [(h_v, h_t, h_r, y)]，皆為 numpy。回傳 (clf_head, 每個 epoch 的平均 loss 列表)。

    class_weights（可選）：長度 3、依 LABEL_TO_ID 順序排列的 tensor，
    用於 docs/spec_c_accuracy_diagnostics.md 的類別加權對照實驗，預設 None（不加權，
    行為與加入這個選項之前完全一樣）。"""
    model.to(device).train()
    clf_head = nn.Linear(model.head.out_features, 3).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(clf_head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    losses = []
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
        avg = total / len(batches)
        losses.append(round(avg, 4))
        print(f"[train] epoch {ep + 1}/{epochs} loss={avg:.4f}")
    return clf_head, losses


def export_z_fused(model, ticker: str, days, device: str, batch: int = 8, ablate_news: bool = False):
    """對所有日期輸出 Z_fused 到 data/outputs/z_fused/。"""
    model.eval()
    out_dir = paths.OUTPUTS / "z_fused" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for date, _y in days:
            h_v, h_t, h_r = load_day(ticker, date, ablate_news=ablate_news)
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
    ap.add_argument("--ablate_news", action="store_true",
                    help="診斷用：H_t 讀進來後在記憶體歸零，不影響磁碟上的向量檔案（見 spec_c_accuracy_diagnostics.md）")
    ap.add_argument("--weighted", action="store_true",
                    help="診斷用：訓練 loss 依類別出現頻率加權，預設關閉（見 spec_c_accuracy_diagnostics.md）")
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
        _, losses = train(model, batches, args.epochs, args.lr, device)
        print("[train] FAKE 流程測通：三個向量進、Z_fused 出、分類 loss 有下降即可")
        log_run("fake", args.ticker, args.n, args.epochs, args.batch, args.lr, losses,
               note="假向量測通流程，loss 數字沒有實質意義")
    else:
        days = load_dataset(args.ticker)
        if not days:
            raise SystemExit("找不到 B 的向量，先跑 module_b_encoder.generate_vectors")
        batches = []
        for i in range(0, len(days), args.batch):
            chunk = days[i:i + args.batch]
            arrs = [load_day(args.ticker, d, ablate_news=args.ablate_news) for d, _ in chunk]
            batches.append((
                np.stack([a[0] for a in arrs]),
                np.stack([a[1] for a in arrs]),
                np.stack([a[2] for a in arrs]),
                np.array([y for _, y in chunk]),
            ))

        class_weights = None
        if args.weighted:
            counts = np.bincount([y for _, y in days], minlength=3).astype(np.float32)
            counts[counts == 0] = 1  # 避免除以 0（理論上三類都該有資料）
            weights = len(days) / (3.0 * counts)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            print(f"[train] 類別加權啟用，權重（BEARISH/NEUTRAL/BULLISH 順序）={weights.tolist()}")

        _, losses = train(model, batches, args.epochs, args.lr, device, class_weights=class_weights)
        export_z_fused(model, args.ticker, days, device, ablate_news=args.ablate_news)

        # 整段時間範圍都產完了，自動彙整成單一索引檔，供分類驗證/回測/RL 訓練直接讀取
        from module_c_fusion.fusion.consolidate import build_index, save_index
        index_data = build_index(args.ticker)
        if index_data:
            save_index(args.ticker, index_data)

        date_range = [days[0][0], days[-1][0]] if days else None
        note = f"news={'off' if args.ablate_news else 'on'}, weighted={args.weighted}"
        log_run("real", args.ticker, len(days), args.epochs, args.batch, args.lr, losses,
               date_range=date_range, note=note)

    ckpt = paths.OUTPUTS / "checkpoints" / "fusion.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[train] checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
