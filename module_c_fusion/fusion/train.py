"""訓練融合模型並輸出 Z_fused。

- 基礎版：以情緒分類為訓練目標（cross-entropy），端到端訓練融合層。
  QLoRA / 對比損失（L_align + L_ground + L_belief）為後續強化，介面不變。
- 產出：Z_fused 到 data/outputs/z_fused/{TICKER}/{date}.npy，
  checkpoint 到 data/outputs/checkpoints/fusion_{TICKER}.pt。

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
  單一 ticker 存 `fusion_{TICKER}.pt`；多個 tickers 存
  `fusion_{TICKER1}_{TICKER2}_....pt`；fake smoke test 則存 `fusion_fake.pt`，彼此不覆蓋。
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch    
import torch.nn as nn

from shared import paths, schemas
from shared.utils import load_config, read_json, stable_json_sha256, write_json
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
    """列出 B 已產出的日期，配上 A 的 label/split/target_date。"""
    manifest_path = paths.dataset_manifest(ticker)
    if not manifest_path.exists():
        raise SystemExit(f"缺少 {manifest_path}；請先重跑 build_dataset")
    manifest = read_json(manifest_path)
    if manifest.get("protocol_version") != 2 or not manifest.get("records_sha256"):
        raise SystemExit(f"{manifest_path} 是舊版格式；請先重跑 build_dataset")

    days = []
    for date in manifest["dates"]:
        d = paths.vector_dir(ticker, date)
        if not (d / "index.json").exists():
            raise SystemExit(
                f"缺少 {d / 'index.json'}；請完整重跑 module_b_encoder.generate_vectors "
                f"--ticker {ticker}"
            )
        label_file = paths.daily_json(ticker, date)
        if not label_file.exists():
            raise SystemExit(f"manifest 指向不存在的 {label_file}；請重跑 build_dataset")
        record = read_json(label_file)
        if "split" not in record or "target_date" not in record.get("prices", {}):
            raise SystemExit(
                f"{label_file} 缺少防洩漏 split/target_date；請先重跑 module_a_data.build_dataset"
            )
        vector_index_path = d / "index.json"
        vector_index = read_json(vector_index_path)
        source_hash = vector_index.get("source_record_sha256")
        if (not source_hash or source_hash != stable_json_sha256(record)
                or vector_index.get("dataset_records_sha256") != manifest["records_sha256"]):
            raise SystemExit(
                f"{vector_index_path} 與目前 {label_file} 版本不一致；"
                f"請重跑 module_b_encoder.generate_vectors --ticker {ticker}"
            )
        days.append((date, LABEL_TO_ID[record["label"]],
                     record["split"], record["prices"]["target_date"]))
    return days


def _model_days(days, split: str = "train"):
    return [day for day in days if day[2] == split]


def _common_multi_stock_splits(days_by_ticker: dict[str, list], train_ratio: float,
                               validation_ratio: float) -> dict[str, dict[str, str]]:
    """多股票共用日期邊界，避免某股票的 test 時段進入另一股票的 shared fusion 訓練。"""
    common_dates = sorted(set.intersection(*(set(day[0] for day in days) for days in days_by_ticker.values())))
    if len(common_dates) < 3:
        raise ValueError("多股票沒有足夠的共同日期可建立共用時間切分")
    i = int(len(common_dates) * train_ratio)
    j = int(len(common_dates) * (train_ratio + validation_ratio))
    if i == 0 or j <= i or j >= len(common_dates):
        raise ValueError("多股票共同日期不足以切出 train/validation/test")
    validation_start, test_start = common_dates[i], common_dates[j]

    result = {}
    for ticker, days in days_by_ticker.items():
        stored_validation = min((day[0] for day in days if day[2] == "validation"), default=None)
        stored_test = min((day[0] for day in days if day[2] == "test"), default=None)
        if stored_validation != validation_start or stored_test != test_start:
            raise SystemExit(
                f"{ticker} 的 dataset split 邊界 ({stored_validation}, {stored_test}) 與多股票共同"
                f"邊界 ({validation_start}, {test_start}) 不一致。為避免 shared fusion/label threshold "
                "跨期洩漏，請先用相同交易日範圍重建各股票 dataset。"
            )
        by_date = {}
        for date, _label, _stored_split, target_date in days:
            if date < validation_start:
                split = "train" if target_date < validation_start else "purged"
            elif date < test_start:
                split = "validation" if target_date < test_start else "purged"
            else:
                split = "test"
            by_date[date] = split
        result[ticker] = by_date
    print(f"[train] 多股票共用切分邊界：validation={validation_start}, test={test_start}")
    return result


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


def export_z_fused(model, ticker: str, days, device: str, batch: int = 8,
                   ablate_news: bool = False, ablate_vision: bool = False):
    """對所有日期輸出 Z_fused 到 data/outputs/z_fused/。"""
    model.eval()
    out_dir = paths.OUTPUTS / "z_fused" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for day in days:
            date = day[0]
            h_v, h_t, h_r = load_day(ticker, date, ablate_news=ablate_news, ablate_vision=ablate_vision)
            z = model(
                torch.from_numpy(h_v[None]).to(device),
                torch.from_numpy(h_t[None]).to(device),
                torch.from_numpy(h_r[None]).to(device),
            )[0].cpu().numpy()
            np.save(out_dir / f"{date}.npy", z)
    dataset_manifest = read_json(paths.dataset_manifest(ticker))
    write_json({
        "ticker": ticker,
        "dates": [day[0] for day in days],
        "dataset_records_sha256": dataset_manifest["records_sha256"],
    }, out_dir / "run.json")
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

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = cfg["rag"]["top_k"]
    model = build_model(cfg)

    if args.apply_checkpoint:
        # ---- 泛化測試：套用既有權重，不訓練，只對指定股票跑前向運算產生 Z_fused ----
        ticker = args.tickers[0] if args.tickers else args.ticker
        checkpoint_path = Path(args.apply_checkpoint)
        checkpoint_meta_path = checkpoint_path.with_suffix(".meta.json")
        if not checkpoint_meta_path.exists():
            raise SystemExit(f"{checkpoint_path} 缺少訓練 metadata，不可作正式泛化測試")
        checkpoint_meta = read_json(checkpoint_meta_path)
        if checkpoint_meta.get("training_split") != "train":
            raise SystemExit(f"{checkpoint_path} 不是 train-only 正式 checkpoint")
        if ticker in checkpoint_meta.get("trained_tickers", []):
            raise SystemExit(
                f"{ticker} 已參與 {checkpoint_path} 訓練，不是 held-out ticker；"
                "不可把這次套用宣稱為跨股票泛化測試"
            )
        days = load_dataset(ticker)
        if not days:
            raise SystemExit(f"找不到 {ticker} 的 B 向量，先跑 module_b_encoder.generate_vectors --ticker {ticker}")
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        export_z_fused(model, ticker, days, device,
                       ablate_news=args.ablate_news, ablate_vision=args.ablate_vision)

        from module_c_fusion.fusion.consolidate import build_index, save_index
        index_data = build_index(ticker)
        if index_data:
            save_index(ticker, index_data)

        date_range = [days[0][0], days[-1][0]] if days else None
        log_run("apply_checkpoint", ticker, len(days), 0, args.batch, args.lr, [],
               date_range=date_range,
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
        days_by_ticker = {}
        for t in tickers:
            d = load_dataset(t)
            if not d:
                raise SystemExit(f"找不到 {t} 的 B 向量，先跑 module_b_encoder.generate_vectors --ticker {t}")
            days_by_ticker[t] = d

        eval_cfg = cfg.get("evaluation", {})
        split_overrides = _common_multi_stock_splits(
            days_by_ticker,
            eval_cfg.get("train_ratio", 0.70),
            eval_cfg.get("validation_ratio", 0.15),
        )
        for t, days in days_by_ticker.items():
            days_by_ticker[t] = [
                (date, label, split_overrides[t][date], target_date)
                for date, label, _split, target_date in days
            ]

        batches = []  # [(ticker, h_v, h_t, h_r, y)]，batch 邊界不跨股票
        for t in tickers:
            days = _model_days(days_by_ticker[t])
            if not days:
                raise SystemExit(f"{t} 的 train split 是空的")
            for i in range(0, len(days), args.batch):
                chunk = days[i:i + args.batch]
                arrs = [load_day(t, d, ablate_news=args.ablate_news, ablate_vision=args.ablate_vision)
                       for d, *_ in chunk]
                batches.append((
                    t,
                    np.stack([a[0] for a in arrs]),
                    np.stack([a[1] for a in arrs]),
                    np.stack([a[2] for a in arrs]),
                    np.array([day[1] for day in chunk]),
                ))

        class_weights_by_ticker = None
        if args.weighted:
            class_weights_by_ticker = {}
            for t in tickers:
                days = _model_days(days_by_ticker[t])
                counts = np.bincount([day[1] for day in days], minlength=3).astype(np.float32)
                counts[counts == 0] = 1
                weights = len(days) / (3.0 * counts)
                class_weights_by_ticker[t] = torch.tensor(weights, dtype=torch.float32, device=device)
                print(f"[train] {t} 類別加權啟用，權重（BEARISH/NEUTRAL/BULLISH 順序）={weights.tolist()}")

        _, losses = train_multi(model, batches, args.epochs, args.lr, device, tickers,
                                class_weights_by_ticker=class_weights_by_ticker)

        from module_c_fusion.fusion.consolidate import build_index, save_index
        for t in tickers:
            days = days_by_ticker[t]
            export_z_fused(model, t, days, device,
                           ablate_news=args.ablate_news, ablate_vision=args.ablate_vision)
            index_data = build_index(t, split_overrides=split_overrides[t])
            if index_data:
                save_index(t, index_data)

        total_days = sum(len(_model_days(d)) for d in days_by_ticker.values())
        note = (f"joint training tickers={tickers}, "
               f"news={'off' if args.ablate_news else 'on'}, "
               f"vision={'off' if args.ablate_vision else 'on'}, weighted={args.weighted}")
        log_run("real_multi", "+".join(tickers), total_days, args.epochs, args.batch, args.lr, losses,
               note=note)

    else:
        ticker = args.tickers[0] if args.tickers else args.ticker
        days = load_dataset(ticker)
        if not days:
            raise SystemExit("找不到 B 的向量，先跑 module_b_encoder.generate_vectors")
        train_days = _model_days(days)
        if not train_days:
            raise SystemExit("train split 是空的；請檢查 dataset manifest 與 evaluation 設定")
        batches = []
        for i in range(0, len(train_days), args.batch):
            chunk = train_days[i:i + args.batch]
            arrs = [load_day(ticker, d, ablate_news=args.ablate_news, ablate_vision=args.ablate_vision)
                   for d, *_ in chunk]
            batches.append((
                np.stack([a[0] for a in arrs]),
                np.stack([a[1] for a in arrs]),
                np.stack([a[2] for a in arrs]),
                np.array([day[1] for day in chunk]),
            ))

        class_weights = None
        if args.weighted:
            counts = np.bincount([day[1] for day in train_days], minlength=3).astype(np.float32)
            counts[counts == 0] = 1  # 避免除以 0（理論上三類都該有資料）
            weights = len(train_days) / (3.0 * counts)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            print(f"[train] 類別加權啟用，權重（BEARISH/NEUTRAL/BULLISH 順序）={weights.tolist()}")

        _, losses = train(model, batches, args.epochs, args.lr, device, class_weights=class_weights)
        export_z_fused(model, ticker, days, device,
                       ablate_news=args.ablate_news, ablate_vision=args.ablate_vision)

        # 整段時間範圍都產完了，自動彙整成單一索引檔，供分類驗證/回測/RL 訓練直接讀取
        from module_c_fusion.fusion.consolidate import build_index, save_index
        index_data = build_index(ticker)
        if index_data:
            save_index(ticker, index_data)

        date_range = [train_days[0][0], train_days[-1][0]] if train_days else None
        note = (f"news={'off' if args.ablate_news else 'on'}, "
               f"vision={'off' if args.ablate_vision else 'on'}, weighted={args.weighted}, "
               f"train_only=True, exported_days={len(days)}")
        log_run("real", ticker, len(train_days), args.epochs, args.batch, args.lr, losses,
               date_range=date_range, note=note)

    if args.fake:
        ckpt = paths.OUTPUTS / "checkpoints" / "fusion_fake.pt"
    elif args.tickers and len(args.tickers) > 1:
        ckpt = paths.OUTPUTS / "checkpoints" / f"fusion_{'_'.join(args.tickers)}.pt"
    else:
        ticker = args.tickers[0] if args.tickers else args.ticker
        ckpt = paths.OUTPUTS / "checkpoints" / f"fusion_{ticker}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    trained_tickers = (args.tickers if args.tickers and len(args.tickers) > 1
                       else [args.tickers[0] if args.tickers else args.ticker])
    write_json({
        "checkpoint": str(ckpt.relative_to(paths.ROOT)),
        "trained_tickers": trained_tickers,
        "training_split": "synthetic" if args.fake else "train",
        "seed": cfg["seed"],
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "weighted": args.weighted,
    }, ckpt.with_suffix(".meta.json"))
    print(f"[train] checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
