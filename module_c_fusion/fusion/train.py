"""訓練融合模型並輸出 Z_fused。

- 基礎版：以情緒分類為訓練目標（cross-entropy），端到端訓練融合層。
  QLoRA / 對比損失（L_align + L_ground + L_belief）為後續強化，介面不變。
- 真實資料一律沿用 A 建立的 strict temporal manifest：只用 Train 更新參數，
  Validation 選最佳 epoch，Test 不參與訓練或選模。兩個邊界的 gap 日也不參與。
- 產出：Z_fused 到 data/outputs/z_fused/{TICKER}/{date}.npy，真實訓練 checkpoint
  到 data/outputs/checkpoints/fusion.pt；--fake 獨立寫到 fusion_fake.pt。

用法：
    python -m module_c_fusion.fusion.train --fake --n 32     # 第一階段：假向量測通
    python -m module_c_fusion.fusion.train --ticker AAPL     # 第二階段：B 的真實向量
    python -m module_c_fusion.fusion.train --tickers AAPL NVDA        # 多股票聯合訓練（留 MSFT 不訓練）
    python -m module_c_fusion.fusion.train --ticker MSFT \
        --apply_checkpoint data/outputs/checkpoints/fusion_AAPL_NVDA.pt   # 泛化測試：套用既有權重，不訓練

留一支股票測泛化能力（`--apply_checkpoint`，王崇穎確認的實驗設計）：
  要驗證融合層對「完全沒訓練過的股票」表現好不好，不能把這支股票也丟進聯合訓練——
  正確做法是融合層只用要訓練的股票（例如 AAPL、NVDA）聯合訓練，訓練完之後，把這組權重
  原封不動套用在被留下來的股票（例如 MSFT）的向量上，只做前向運算產生 Z_fused，完全不
  更新任何權重（不呼叫 train()/train_multi()）。`--apply_checkpoint <path>` 就是做這件事：
  讀進指定的 checkpoint、直接對 `--ticker` 指定的那支股票跑 `export_z_fused()`，其餘流程
  （分類驗證用各自的 `classifier.py` 訓練）不變。

多股票聯合訓練（`--tickers`，見 docs/decisions.md 多股票擴充實驗相關決議）：
  王崇穎（負責人）的架構建議是「共用權重值（固定前面的模型跟 weighting），但輸出層
  （最後一層）可以分開」。融合層本體（`CrossModalTransformer`，H_v/H_t/H_r -> Z_fused
  這部分）是「前面的模型」，多支股票的資料混在一起聯合訓練同一組權重；但訓練時用來算
  分類 loss 的最後一層（`clf_head`），每支股票各自有自己的一個（`nn.Linear` 存在
  `dict` 裡，key 是 ticker），不共用——一來每支股票的漲跌標籤門檻是各自算的分位數
  （decisions.md #30），硬要用同一層去預測不同股票的標籤不合理；二來這樣才是忠實
  照著「共用主幹、分開輸出層」的架構做，不是只共用主幹卻仍然只有一個輸出層。
  這個 `clf_head` 本身只是訓練時提供梯度訊號用的鷹架，不會被存進 checkpoint（只存
  `model.state_dict()`），所以每支股票各自一個不會影響到之後怎麼載入這個 checkpoint。
  單一 ticker（`--ticker` 或只給一個 `--tickers`）時，行為與加入這個功能之前完全一樣，
  checkpoint 仍存到 `fusion.pt`；給多個 tickers 時，checkpoint 改存到
  `fusion_{TICKER1}_{TICKER2}_....pt`，不會覆蓋掉單一 ticker 訓練出來的 `fusion.pt`，
  兩者可以並存比較，不需要手動備份。
"""
import argparse
import copy
import datetime as dt
import json

import numpy as np
import torch
import torch.nn as nn

from shared import paths, schemas
from shared.temporal_split import load_temporal_manifest, split_rows
from shared.utils import load_config, read_json, write_json
from module_c_fusion.fusion.model import build_model

LABEL_TO_ID = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
TRAIN_LOG = paths.OUTPUTS / "logs" / "train_log.jsonl"


def log_run(mode: str, ticker: str, n_days: int, epochs: int, batch: int, lr: float,
           losses: list[float], date_range: list[str] | None = None,
           note: str = "", validation_history: list[dict] | None = None,
           best_epoch: int | None = None, split_manifest: str | None = None) -> None:
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
        "validation_history": validation_history,
        "best_epoch": best_epoch,
        "split_manifest": split_manifest,
        "note": note,
    }
    with open(TRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[train] 訓練紀錄 -> {TRAIN_LOG}")


def load_day(ticker: str, date: str, ablate_news: bool = False, ablate_vision: bool = False):
    """讀 B 的一天向量，回傳 (h_v, h_t, h_r) numpy。

    ablate_news=True 時，H_t 讀進來後在記憶體裡直接歸零；ablate_vision=True 時，
    H_v（全部視覺圖）比照辦理歸零——這是 ViT domain gap 對照實驗用的（decisions.md #46，
    沿用 #28 新聞歸零對照實驗同一套手法，這次換成歸零視覺輸入，比較「有無 K 線圖」對
    分類準確度的影響）。兩者皆只影響這次執行的記憶體內容，不改動磁碟上的向量檔案，
    不會留下任何需要事後還原的殘留狀態。"""
    d = paths.vector_dir(ticker, date)
    index = read_json(d / "index.json")
    schemas.validate_vector_index(index)
    h_v = np.load(d / "H_v.npy")
    h_t = np.load(d / "H_t.npy")
    h_r = np.load(d / "H_r.npy")
    arrays = {"H_v": h_v, "H_t": h_t, "H_r": h_r}
    for name, array in arrays.items():
        expected = tuple(index["vectors"][name]["shape"])
        if array.shape != expected:
            raise ValueError(f"{d / (name + '.npy')} shape={array.shape}，index.json 記錄為 {expected}")
    if ablate_news:
        h_t = np.zeros_like(h_t)
    if ablate_vision:
        h_v = np.zeros_like(h_v)
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


def load_split_dataset(ticker: str, days: list[tuple[str, int]]) -> tuple[dict, dict]:
    """讀取唯一 manifest，並要求 B 向量完整覆蓋 manifest 的所有日期。"""
    manifest = load_temporal_manifest(ticker)
    groups = split_rows(days, manifest, require_all_manifest_dates=True)
    return manifest, groups


def make_batches(ticker: str, days: list[tuple[str, int]], batch_size: int,
                 *, ablate_news: bool = False, ablate_vision: bool = False,
                 split_name: str = "dataset") -> list:
    print(f"[train] 載入 {ticker} {split_name}：{len(days)} 天，batch={batch_size}")
    batches = []
    for i in range(0, len(days), batch_size):
        chunk = days[i:i + batch_size]
        arrs = [
            load_day(ticker, date, ablate_news=ablate_news, ablate_vision=ablate_vision)
            for date, _ in chunk
        ]
        batches.append((
            np.stack([item[0] for item in arrs]),
            np.stack([item[1] for item in arrs]),
            np.stack([item[2] for item in arrs]),
            np.asarray([label for _, label in chunk]),
        ))
        loaded = min(i + len(chunk), len(days))
        if loaded == len(days) or loaded % 100 < batch_size:
            finished = loaded == len(days)
            print(
                f"[train] 載入 {ticker} {split_name}: {loaded}/{len(days)} 天",
                end="\n" if finished else "\r",
                flush=True,
            )
    return batches


def class_weights_for_days(days: list[tuple[str, int]], device: str) -> torch.Tensor:
    counts = np.bincount([label for _, label in days], minlength=3).astype(np.float32)
    counts[counts == 0] = 1
    weights = len(days) / (3.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def protocol_days(groups: dict[str, list[tuple[str, int]]]) -> list[tuple[str, int]]:
    """包含兩段 gap、但排除 manifest 外的舊檔，供凍結後輸出 Z_fused。"""
    return sorted(row for rows in groups.values() for row in rows)


def multi_batches(ticker: str, batches: list) -> list:
    return [(ticker, *batch) for batch in batches]


def make_fake_batch(n: int, k: int, n_vision_inputs: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    h_v = rng.standard_normal((n, n_vision_inputs, 197, 768)).astype(np.float32)
    h_t = rng.standard_normal((n, 512, 768)).astype(np.float32)
    h_r = rng.standard_normal((n, k, 512, 768)).astype(np.float32)
    y = rng.integers(0, 3, n)
    return h_v, h_t, h_r, y


def train(model, batches, epochs: int, lr: float, device: str, class_weights=None):
    """batches: [(h_v, h_t, h_r, y)]，皆為 numpy。回傳 (clf_head, 每個 epoch 的平均 loss 列表)。

    class_weights（可選）：長度 3、依 LABEL_TO_ID 順序排列的 tensor，
    用於類別加權對照實驗（結果見 docs/decisions.md #28、docs/data_and_experiments_log.md
    第三節），預設 None（不加權，行為與加入這個選項之前完全一樣）。"""
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


def train_multi(model, batches, epochs: int, lr: float, device: str, tickers: list[str],
                class_weights_by_ticker: dict | None = None):
    """多股票聯合訓練：batches 是 [(ticker, h_v, h_t, h_r, y)]，混合多支股票的批次
    （每個 batch 內部仍是同一支股票，不同 batch 可能屬於不同股票）。

    融合層本體（model）共用、梯度來自所有股票；分類 loss 用的 clf_head 每支股票各自一個
    （只是訓練時的梯度鷹架，不存進 checkpoint），對應王崇穎「共用主幹、分開輸出層」的建議。
    回傳 (clf_heads dict, 每個 epoch 的平均 loss 列表)。
    """
    model.to(device).train()
    clf_heads = {t: nn.Linear(model.head.out_features, 3).to(device) for t in tickers}
    params = list(model.parameters())
    for head in clf_heads.values():
        params += list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    losses = []
    for ep in range(epochs):
        total = 0.0
        for ticker, h_v, h_t, h_r, y in batches:
            h_v = torch.from_numpy(h_v).to(device)
            h_t = torch.from_numpy(h_t).to(device)
            h_r = torch.from_numpy(h_r).to(device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            weight = (class_weights_by_ticker or {}).get(ticker)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
            z = model(h_v, h_t, h_r)
            loss = loss_fn(clf_heads[ticker](z), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(batches)
        losses.append(round(avg, 4))
        print(f"[train] epoch {ep + 1}/{epochs} loss={avg:.4f}")
    return clf_heads, losses


def _macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    scores = []
    for label in range(3):
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def evaluate_head(model, clf_head, batches, device: str) -> dict:
    model.eval()
    clf_head.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, n_items = 0.0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for h_v, h_t, h_r, y in batches:
            h_v = torch.from_numpy(h_v).to(device)
            h_t = torch.from_numpy(h_t).to(device)
            h_r = torch.from_numpy(h_r).to(device)
            labels = torch.as_tensor(y, dtype=torch.long, device=device)
            logits = clf_head(model(h_v, h_t, h_r))
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            n_items += len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "loss": total_loss / max(n_items, 1),
        "macro_f1": _macro_f1(y_true, y_pred),
        "n": n_items,
    }


def train_with_validation(model, train_batches, validation_batches, epochs: int, lr: float,
                          device: str, class_weights=None):
    """只用 train 更新權重，以 validation macro F1 選回最佳 epoch。"""
    model.to(device)
    clf_head = nn.Linear(model.head.out_features, 3).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(clf_head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    history = []
    best_score, best_loss, best_epoch = -1.0, float("inf"), 0
    best_model_state = None
    best_head_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        clf_head.train()
        total_loss, n_items = 0.0, 0
        for batch_index, (h_v, h_t, h_r, y) in enumerate(train_batches, start=1):
            h_v = torch.from_numpy(h_v).to(device)
            h_t = torch.from_numpy(h_t).to(device)
            h_r = torch.from_numpy(h_r).to(device)
            labels = torch.as_tensor(y, dtype=torch.long, device=device)
            loss = loss_fn(clf_head(model(h_v, h_t, h_r)), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            n_items += len(labels)
            if batch_index == len(train_batches) or batch_index % 50 == 0:
                print(f"[train] epoch {epoch}/{epochs} batch "
                      f"{batch_index}/{len(train_batches)}")

        train_loss = total_loss / max(n_items, 1)
        validation = evaluate_head(model, clf_head, validation_batches, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "validation_loss": round(validation["loss"], 6),
            "validation_macro_f1": round(validation["macro_f1"], 6),
        }
        history.append(row)
        print(
            f"[train] epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} val_macro_f1={validation['macro_f1']:.4f}"
        )
        score = validation["macro_f1"]
        if score > best_score or (score == best_score and validation["loss"] < best_loss):
            best_score, best_loss, best_epoch = score, validation["loss"], epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_head_state = copy.deepcopy(clf_head.state_dict())

    model.load_state_dict(best_model_state)
    clf_head.load_state_dict(best_head_state)
    print(f"[train] 採用 epoch {best_epoch}：val_macro_f1={best_score:.4f}, val_loss={best_loss:.4f}")
    return clf_head, history, best_epoch


def evaluate_multi_heads(model, clf_heads: dict, batches, device: str) -> dict:
    model.eval()
    for head in clf_heads.values():
        head.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, n_items = 0.0, 0
    y_true = {ticker: [] for ticker in clf_heads}
    y_pred = {ticker: [] for ticker in clf_heads}
    with torch.no_grad():
        for ticker, h_v, h_t, h_r, y in batches:
            h_v = torch.from_numpy(h_v).to(device)
            h_t = torch.from_numpy(h_t).to(device)
            h_r = torch.from_numpy(h_r).to(device)
            labels = torch.as_tensor(y, dtype=torch.long, device=device)
            logits = clf_heads[ticker](model(h_v, h_t, h_r))
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            n_items += len(labels)
            y_true[ticker].extend(labels.cpu().tolist())
            y_pred[ticker].extend(logits.argmax(dim=1).cpu().tolist())
    by_ticker = {
        ticker: _macro_f1(y_true[ticker], y_pred[ticker]) for ticker in clf_heads
    }
    return {
        "loss": total_loss / max(n_items, 1),
        "macro_f1": float(np.mean(list(by_ticker.values()))),
        "macro_f1_by_ticker": by_ticker,
        "n": n_items,
    }


def train_multi_with_validation(model, train_batches, validation_batches, epochs: int, lr: float,
                                device: str, tickers: list[str],
                                class_weights_by_ticker: dict | None = None):
    model.to(device)
    clf_heads = {ticker: nn.Linear(model.head.out_features, 3).to(device) for ticker in tickers}
    parameters = list(model.parameters())
    for head in clf_heads.values():
        parameters.extend(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    history = []
    best_score, best_loss, best_epoch = -1.0, float("inf"), 0
    best_model_state = None
    best_head_states = None

    for epoch in range(1, epochs + 1):
        model.train()
        for head in clf_heads.values():
            head.train()
        total_loss, n_items = 0.0, 0
        for batch_index, (ticker, h_v, h_t, h_r, y) in enumerate(train_batches, start=1):
            h_v = torch.from_numpy(h_v).to(device)
            h_t = torch.from_numpy(h_t).to(device)
            h_r = torch.from_numpy(h_r).to(device)
            labels = torch.as_tensor(y, dtype=torch.long, device=device)
            loss_fn = nn.CrossEntropyLoss(weight=(class_weights_by_ticker or {}).get(ticker))
            loss = loss_fn(clf_heads[ticker](model(h_v, h_t, h_r)), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            n_items += len(labels)
            if batch_index == len(train_batches) or batch_index % 50 == 0:
                print(f"[train] epoch {epoch}/{epochs} batch "
                      f"{batch_index}/{len(train_batches)}")

        train_loss = total_loss / max(n_items, 1)
        validation = evaluate_multi_heads(model, clf_heads, validation_batches, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "validation_loss": round(validation["loss"], 6),
            "validation_macro_f1": round(validation["macro_f1"], 6),
            "validation_macro_f1_by_ticker": {
                ticker: round(score, 6)
                for ticker, score in validation["macro_f1_by_ticker"].items()
            },
        }
        history.append(row)
        print(
            f"[train] epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} val_macro_f1={validation['macro_f1']:.4f}"
        )
        score = validation["macro_f1"]
        if score > best_score or (score == best_score and validation["loss"] < best_loss):
            best_score, best_loss, best_epoch = score, validation["loss"], epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_head_states = {
                ticker: copy.deepcopy(head.state_dict()) for ticker, head in clf_heads.items()
            }

    model.load_state_dict(best_model_state)
    for ticker, state in best_head_states.items():
        clf_heads[ticker].load_state_dict(state)
    print(f"[train] 採用 epoch {best_epoch}：val_macro_f1={best_score:.4f}, val_loss={best_loss:.4f}")
    return clf_heads, history, best_epoch


def export_z_fused(model, ticker: str, days, device: str, batch: int = 8,
                   ablate_news: bool = False, ablate_vision: bool = False):
    """對所有日期輸出 Z_fused 到 data/outputs/z_fused/。"""
    model.eval()
    out_dir = paths.OUTPUTS / "z_fused" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, (date, _y) in enumerate(days, start=1):
            h_v, h_t, h_r = load_day(ticker, date, ablate_news=ablate_news, ablate_vision=ablate_vision)
            z = model(
                torch.from_numpy(h_v[None]).to(device),
                torch.from_numpy(h_t[None]).to(device),
                torch.from_numpy(h_r[None]).to(device),
            )[0].cpu().numpy()
            np.save(out_dir / f"{date}.npy", z)
            if index == len(days) or index % 100 == 0:
                print(f"[train] 輸出 {ticker} Z_fused: {index}/{len(days)} 天")
    print(f"[train] Z_fused x{len(days)} -> {out_dir}")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="多股票聯合訓練：給多個 ticker 時融合層本體聯合訓練、"
                         "分類 loss 的輸出層各股票分開；只給一個等同 --ticker")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--n", type=int, default=32, help="--fake 時樣本數")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ablate_news", action="store_true",
                    help="診斷用：H_t 讀進來後在記憶體歸零，不影響磁碟上的向量檔案（結果見 docs/decisions.md #28）")
    ap.add_argument("--ablate_vision", action="store_true",
                    help="診斷用：H_v（全部視覺圖）讀進來後在記憶體歸零，不影響磁碟上的向量檔案"
                         "（ViT domain gap 對照實驗，見 decisions.md #46）")
    ap.add_argument("--weighted", action="store_true",
                    help="診斷用：訓練 loss 依類別出現頻率加權，預設關閉（結果見 docs/decisions.md #28）")
    ap.add_argument("--apply_checkpoint", default=None,
                    help="套用既有 checkpoint 產生 Z_fused，完全不訓練——測試共用融合層對"
                         "沒訓練過的股票的泛化能力用，搭配 --ticker 指定要套用在哪支股票")
    args = ap.parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs 必須至少為 1")
    if args.batch < 1:
        raise SystemExit("--batch 必須至少為 1")
    if args.fake and args.n < 1:
        raise SystemExit("--n 必須至少為 1")

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")
    k = cfg["rag"]["top_k"]
    model = build_model(cfg)
    requested_tickers = args.tickers if args.tickers else [args.ticker]
    if args.fake:
        ckpt = paths.OUTPUTS / "checkpoints" / "fusion_fake.pt"
    elif len(requested_tickers) > 1:
        ckpt = paths.OUTPUTS / "checkpoints" / f"fusion_{'_'.join(requested_tickers)}.pt"
    else:
        ckpt = paths.OUTPUTS / "checkpoints" / "fusion.pt"

    if args.apply_checkpoint:
        # ---- 泛化測試：套用既有權重，不訓練，只對指定股票跑前向運算產生 Z_fused ----
        ticker = args.tickers[0] if args.tickers else args.ticker
        days = load_dataset(ticker)
        if not days:
            raise SystemExit(f"找不到 {ticker} 的 B 向量，先跑 module_b_encoder.generate_vectors --ticker {ticker}")
        _manifest, groups = load_split_dataset(ticker, days)
        export_days = protocol_days(groups)
        state = torch.load(args.apply_checkpoint, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        export_z_fused(model, ticker, export_days, device,
                       ablate_news=args.ablate_news, ablate_vision=args.ablate_vision)

        from module_c_fusion.fusion.consolidate import build_index, save_index
        index_data = build_index(ticker)
        if index_data:
            save_index(ticker, index_data, checkpoint_path=args.apply_checkpoint)

        date_range = [export_days[0][0], export_days[-1][0]] if export_days else None
        log_run("apply_checkpoint", ticker, len(export_days), 0, args.batch, args.lr, [],
               date_range=date_range,
               split_manifest=str(paths.temporal_split_path(ticker)),
               note=f"泛化測試，套用 checkpoint={args.apply_checkpoint}，未訓練")
        print(f"[train] 已套用 {args.apply_checkpoint} 對 {ticker} 產生 Z_fused（未訓練，權重完全沿用該 checkpoint）")
        return

    if args.fake:
        h_v, h_t, h_r, y = make_fake_batch(
            args.n, k, len(cfg["chart"]["vision_inputs"]), cfg["seed"]
        )
        batches = [
            (h_v[i:i + args.batch], h_t[i:i + args.batch], h_r[i:i + args.batch], y[i:i + args.batch])
            for i in range(0, args.n, args.batch)
        ]
        _, losses = train(model, batches, args.epochs, args.lr, device)
        print("[train] FAKE 流程測通：三個向量進、Z_fused 出、分類 loss 有下降即可")
        log_run("fake", args.ticker, args.n, args.epochs, args.batch, args.lr, losses,
               note="假向量測通流程，loss 數字沒有實質意義")
    elif args.tickers and len(args.tickers) > 1:
        # ---- 多股票聯合訓練：融合層本體共用，分類 loss 的輸出層各股票分開 ----
        tickers = args.tickers
        groups_by_ticker = {}
        for t in tickers:
            d = load_dataset(t)
            if not d:
                raise SystemExit(f"找不到 {t} 的 B 向量，先跑 module_b_encoder.generate_vectors --ticker {t}")
            _, groups = load_split_dataset(t, d)
            groups_by_ticker[t] = groups

        train_batches, validation_batches = [], []
        for t in tickers:
            train_batches.extend(multi_batches(t, make_batches(
                t, groups_by_ticker[t]["train"], args.batch,
                ablate_news=args.ablate_news, ablate_vision=args.ablate_vision,
                split_name="Train",
            )))
            validation_batches.extend(multi_batches(t, make_batches(
                t, groups_by_ticker[t]["validation"], args.batch,
                ablate_news=args.ablate_news, ablate_vision=args.ablate_vision,
                split_name="Validation",
            )))

        class_weights_by_ticker = None
        if args.weighted:
            class_weights_by_ticker = {}
            for t in tickers:
                weights = class_weights_for_days(groups_by_ticker[t]["train"], device)
                class_weights_by_ticker[t] = weights
                print(f"[train] {t} Train-only 類別權重="
                      f"{weights.detach().cpu().tolist()}")

        _, history, best_epoch = train_multi_with_validation(
            model, train_batches, validation_batches, args.epochs, args.lr, device, tickers,
            class_weights_by_ticker=class_weights_by_ticker,
        )
        losses = [row["train_loss"] for row in history]

        for t in tickers:
            export_days = protocol_days(groups_by_ticker[t])
            export_z_fused(model, t, export_days, device,
                           ablate_news=args.ablate_news, ablate_vision=args.ablate_vision)
        exported_tickers = tickers

        total_days = sum(len(groups_by_ticker[t]["train"]) for t in tickers)
        note = (f"joint training tickers={tickers}, "
               f"news={'off' if args.ablate_news else 'on'}, "
               f"vision={'off' if args.ablate_vision else 'on'}, weighted={args.weighted}, "
               "strict temporal split; test labels never used")
        log_run("real_multi", "+".join(tickers), total_days, args.epochs, args.batch, args.lr, losses,
               note=note, validation_history=history, best_epoch=best_epoch,
               split_manifest=", ".join(str(paths.temporal_split_path(t)) for t in tickers))

        write_json({
            "protocol": "strict_temporal_v1",
            "tickers": tickers,
            "split_manifests": {
                t: str(paths.temporal_split_path(t)) for t in tickers
            },
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "history": history,
            "checkpoint": str(ckpt),
        }, paths.OUTPUTS / "metrics" / f"fusion_train_report_{'_'.join(tickers)}.json")

    else:
        ticker = args.tickers[0] if args.tickers else args.ticker
        days = load_dataset(ticker)
        if not days:
            raise SystemExit("找不到 B 的向量，先跑 module_b_encoder.generate_vectors")
        manifest, groups = load_split_dataset(ticker, days)
        train_batches = make_batches(
            ticker, groups["train"], args.batch,
            ablate_news=args.ablate_news, ablate_vision=args.ablate_vision,
            split_name="Train",
        )
        validation_batches = make_batches(
            ticker, groups["validation"], args.batch,
            ablate_news=args.ablate_news, ablate_vision=args.ablate_vision,
            split_name="Validation",
        )

        class_weights = None
        if args.weighted:
            class_weights = class_weights_for_days(groups["train"], device)
            print(f"[train] Train-only 類別權重（BEARISH/NEUTRAL/BULLISH）="
                  f"{class_weights.detach().cpu().tolist()}")

        _, history, best_epoch = train_with_validation(
            model, train_batches, validation_batches, args.epochs, args.lr, device,
            class_weights=class_weights,
        )
        losses = [row["train_loss"] for row in history]
        export_days = protocol_days(groups)
        export_z_fused(model, ticker, export_days, device,
                       ablate_news=args.ablate_news, ablate_vision=args.ablate_vision)

        exported_tickers = [ticker]

        date_range = [export_days[0][0], export_days[-1][0]] if export_days else None
        note = (f"news={'off' if args.ablate_news else 'on'}, "
               f"vision={'off' if args.ablate_vision else 'on'}, weighted={args.weighted}, "
               "strict temporal split; test labels never used")
        log_run("real", ticker, len(groups["train"]), args.epochs, args.batch, args.lr, losses,
               date_range=date_range, note=note, validation_history=history,
               best_epoch=best_epoch,
               split_manifest=str(paths.temporal_split_path(ticker)))
        write_json({
            "protocol": manifest["protocol"],
            "ticker": ticker,
            "split_manifest": str(paths.temporal_split_path(ticker)),
            "split_counts": manifest["counts"],
            "split_periods": manifest["periods"],
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "history": history,
            "checkpoint": str(ckpt),
        }, paths.OUTPUTS / "metrics" / f"fusion_train_report_{ticker}.json")

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[train] checkpoint -> {ckpt}")
    if not args.fake:
        # 先存最佳 checkpoint，再建索引；metadata 會寫入實際權重的 SHA256，
        # 防止之後把舊 Z_fused 與新權重混用。
        from module_c_fusion.fusion.consolidate import build_index, save_index
        for exported_ticker in exported_tickers:
            index_data = build_index(exported_ticker)
            if index_data:
                save_index(exported_ticker, index_data, checkpoint_path=ckpt)


if __name__ == "__main__":
    main()
