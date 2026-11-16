"""ViT＋分類頭能不能看出 K 線型態（AAPL，2026-11-16；--ticker 可換股票，例如 NVDA）。

目標：Haiku 讀圖判的 7 類型態（data/labels/chart_caption/AAPL.json 的 pattern）。
切分：全專案共用的 strict temporal manifest（Train／Validation／Test），Test 只評一次。

A. frozen：ViT 凍結，讀 B 存好的 H_v [2,197,768]（K 線圖、成交量圖），
   特徵取 CLS 或 patch 平均，Logistic Regression；Validation 選 C（macro F1），Train＋Validation refit。
B. finetune：google/vit-base-patch16-224 整個微調＋線性分類頭，只用 K 線圖（或 K 線＋成交量兩張圖的
   CLS 串接）；class-weighted CE；不做翻轉等資料增強（左右翻轉會把上升變下降）；
   依 Validation macro F1 選 epoch，Test 只評選中的那個 epoch。

結果寫到 data/outputs/experiments/vit_pattern/（不進版控）；實驗紀錄：docs/vit_pattern_experiments_2026-11.md。

用法（repo 根目錄）：
    PYTHONPATH=. python experiments/vision/vit_pattern.py frozen
    PYTHONPATH=. python experiments/vision/vit_pattern.py finetune --images candlestick --seed 42
    PYTHONPATH=. python experiments/vision/vit_pattern.py finetune --unfreeze 2 --seed 42   # 只解凍最後 2 層
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from shared import paths
from shared.patterns import PATTERN_IDS
from shared.temporal_split import load_temporal_manifest, split_rows

OUT = paths.OUTPUTS / "experiments" / "vit_pattern"   # 結果不進版控
TICKER = "AAPL"   # 由 --ticker 覆寫；多支股票用 + 連接（例如 AAPL+NVDA）＝合併訓練，Test 分股票各評一次


def tag():
    """AAPL 的輸出檔名維持原樣，其他股票加上前綴。"""
    return "" if TICKER == "AAPL" else f"{TICKER}_"


def load_groups():
    """每一列是 ((ticker, date), 型態 index)；每支股票用自己的 manifest 切，再把同名的組合併。"""
    groups = {}
    for t in TICKER.split("+"):
        caps = json.load(open(paths.ROOT / "data" / "labels" / "chart_caption" / f"{t}.json"))
        rows = [((t, d), PATTERN_IDS.index(v["pattern"])) for d, v in sorted(caps.items())
                if (paths.vector_dir(t, d) / "H_v.npy").exists()]
        for name, part in split_rows(rows, load_temporal_manifest(t), date_getter=lambda r: r[0][1],
                                     require_all_manifest_dates=False).items():
            groups.setdefault(name, []).extend(part)
    return groups


def metrics(y, pred, labels):
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, labels=labels, average="macro", zero_division=0), 4)}


def run_frozen(groups):
    labels = list(range(len(PATTERN_IDS)))
    feats = {}
    for name, rows in groups.items():
        hv = np.stack([np.load(paths.vector_dir(*d) / "H_v.npy") for d, _ in rows])   # [N,2,197,768]
        feats[name] = {
            "candle_cls": hv[:, 0, 0], "candle_mean": hv[:, 0, 1:].mean(1),
            "both_cls": hv[:, :, 0].reshape(len(rows), -1),
            "both_mean": hv[:, :, 1:].mean(2).reshape(len(rows), -1),
        }
    y = {n: np.array([p for _, p in rows]) for n, rows in groups.items()}
    majority = Counter(y["train"].tolist()).most_common(1)[0][0]
    report = {"ticker": TICKER, "n": {k: len(v) for k, v in y.items()},
              "majority_test_accuracy": round(float(np.mean(y["test"] == majority)), 4),
              "test_distribution": {PATTERN_IDS[k]: v for k, v in sorted(Counter(y["test"].tolist()).items())},
              "results": {}}
    for key in feats["train"]:
        xtr, xva, xte = feats["train"][key], feats["validation"][key], feats["test"][key]
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        xtr, xva, xte = (xtr - mu) / sd, (xva - mu) / sd, (xte - mu) / sd
        by_c = {}
        for c in (0.001, 0.01, 0.1, 1.0):
            clf = LogisticRegression(max_iter=3000, C=c, class_weight="balanced").fit(xtr, y["train"])
            by_c[c] = f1_score(y["validation"], clf.predict(xva), labels=labels, average="macro", zero_division=0)
        best = max(by_c, key=by_c.get)
        clf = LogisticRegression(max_iter=3000, C=best, class_weight="balanced").fit(
            np.vstack([xtr, xva]), np.concatenate([y["train"], y["validation"]]))
        pred = clf.predict(xte)
        report["results"][key] = {"C": best, "val_macro_f1": round(by_c[best], 4), **metrics(y["test"], pred, labels)}
        print(key, report["results"][key])
    json.dump(report, open(OUT / f"{tag()}frozen_report.json", "w"), indent=2, ensure_ascii=False)
    print("majority", report["majority_test_accuracy"])


def run_finetune(groups, images: str, seed: int, epochs: int, lr: float, batch: int, unfreeze: int = 0):
    import torch
    import torch.nn as nn
    from PIL import Image
    from transformers import ViTImageProcessor, ViTModel

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    name = "google/vit-base-patch16-224"
    proc = ViTImageProcessor.from_pretrained(name)
    kinds = ["candlestick"] if images == "candlestick" else ["candlestick", "volume"]

    def img_path(key, kind):
        t, d = key
        return paths.RAW_CHARTS / t / f"{d}.png" if kind == "candlestick" else \
            paths.RAW_CHARTS / t / "volume" / f"{d}.png"

    def tensors(rows):
        x = []
        for kind in kinds:
            ims = [Image.open(img_path(d, kind)).convert("RGB") for d, _ in rows]
            x.append(proc(images=ims, return_tensors="pt")["pixel_values"])
        return torch.stack(x, 1), torch.tensor([p for _, p in rows])      # [N,I,3,224,224]

    data = {n: tensors(r) for n, r in groups.items()}

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = ViTModel.from_pretrained(name, add_pooling_layer=False)
            self.head = nn.Linear(768 * len(kinds), len(PATTERN_IDS))

        def forward(self, x):
            B, I = x.shape[:2]
            h = self.vit(pixel_values=x.flatten(0, 1)).last_hidden_state[:, 0]    # CLS
            return self.head(h.reshape(B, I * 768))

    net = Net().to(dev)
    if unfreeze:   # 只訓練最後 unfreeze 層＋最後的 LayerNorm＋分類頭；embedding 與前面的層凍結
        for prm in net.vit.parameters():
            prm.requires_grad = False
        for blk in net.vit.encoder.layer[-unfreeze:]:
            for prm in blk.parameters():
                prm.requires_grad = True
        for prm in net.vit.layernorm.parameters():
            prm.requires_grad = True
    n_train = sum(prm.numel() for prm in net.parameters() if prm.requires_grad)
    print("trainable params", n_train, flush=True)
    ytr = data["train"][1]
    counts = torch.bincount(ytr, minlength=len(PATTERN_IDS)).float()
    weight = (counts.sum() / (len(PATTERN_IDS) * counts.clamp_min(1))).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW([{"params": [q for q in net.vit.parameters() if q.requires_grad], "lr": lr},
                             {"params": net.head.parameters(), "lr": lr * 50}], weight_decay=0.01)
    steps = epochs * ((len(ytr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 50], total_steps=steps, pct_start=0.1)
    labels = list(range(len(PATTERN_IDS)))

    def predict(x):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 32):
                out.append(net(x[i:i + 32].to(dev)).argmax(-1).cpu())
        return torch.cat(out).numpy()

    history, best = [], None
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(ytr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = loss_fn(net(data["train"][0][idx].to(dev)), ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        va = metrics(data["validation"][1].numpy(), predict(data["validation"][0]), labels)
        tr = metrics(ytr.numpy(), predict(data["train"][0]), labels)
        history.append({"epoch": ep, "train_loss": round(tot / len(ytr), 4), "train": tr, "validation": va})
        print(history[-1], flush=True)
        if best is None or va["macro_f1"] > best[0]:
            best = (va["macro_f1"], ep, predict(data["test"][0]))
    yte = data["test"][1].numpy()
    pred = best[2]
    report = {"ticker": TICKER, "images": images, "seed": seed, "unfreeze_last_layers": unfreeze or "all", "trainable_params": n_train, "epochs": epochs, "lr": lr, "batch": batch,
              "selected_epoch": best[1], "val_macro_f1": round(best[0], 4),
              "test": metrics(yte, pred, labels),
              "test_by_ticker": {t: {**metrics(yte[m], pred[m], labels),
                                     "confusion": confusion_matrix(yte[m], pred[m], labels=labels).tolist()}
                                 for t in TICKER.split("+")
                                 for m in [np.array([k[0] == t for k, _ in groups["test"]])]},
              "confusion_test": {"labels": PATTERN_IDS, "matrix": confusion_matrix(yte, pred, labels=labels).tolist()},
              "history": history}
    json.dump(report, open(OUT / f"{tag()}finetune_{images}{f'_unfreeze{unfreeze}' if unfreeze else ''}_seed{seed}.json", "w"), indent=2, ensure_ascii=False)
    print("selected epoch", best[1], "test", report["test"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["frozen", "finetune"])
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--images", choices=["candlestick", "both"], default="candlestick")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--unfreeze", type=int, default=0, help="只解凍最後 N 層 Transformer（0 = 全部微調）")
    args = ap.parse_args()
    TICKER = args.ticker
    groups = load_groups()
    print({k: len(v) for k, v in groups.items()})
    if args.mode == "frozen":
        run_frozen(groups)
    else:
        run_finetune(groups, args.images, args.seed, args.epochs, args.lr, args.batch, args.unfreeze)
