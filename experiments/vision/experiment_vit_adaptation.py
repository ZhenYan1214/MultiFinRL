"""比較 ViT-only frozen / partial fine-tuning 的嚴格時間序實驗。

每個交易日以兩張圖（預設 K 線 + volume）作為一筆樣本。兩張圖共用同一個 ViT，
取各自 CLS token 串接後交給小型三分類頭。文字、RAG 與 fusion 完全不參與，避免它們
掩蓋 ViT 本身的好壞。

防洩漏設計：
- 依日期切 train / validation / test，不 shuffle split。
- split 之間保留 label horizon 天的 gap。
- 三分類分位數門檻只使用 train period 的 forward returns 計算。
- 所有策略只看 validation；選出最佳策略後，test 只評估勝者與預先定義的 frozen baseline。

快速驗證：
    python experiments/vision/experiment_vit_adaptation.py --strategies frozen --epochs 1 --limit 60

正式第一輪：
    python experiments/vision/experiment_vit_adaptation.py \
        --strategies frozen last1 last2 last4 --epochs 10 --patience 3 --evaluate-test

多股票泛化：
    python experiments/vision/experiment_vit_adaptation.py \
        --train-tickers AAPL NVDA --held-out-ticker MSFT \
        --strategies frozen last4 --epochs 10 --patience 3 --evaluate-test

LoRA（最後四個 ViT blocks 的 query/value）：
    python experiments/vision/experiment_vit_adaptation.py \
        --train-tickers AAPL NVDA MSFT --held-out-ticker JPM \
        --strategies frozen lora_r4 lora_r8 --vit-lr 1e-4 \
        --epochs 10 --patience 3 --evaluate-test
"""
import argparse
import copy
import gc
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.vision.vision_classifier import (
    DualImageViTClassifier,
    build_classification_head,
)
from shared import paths
from shared.utils import load_config, read_json, write_json


LABEL_NAMES = ["BEARISH", "NEUTRAL", "BULLISH"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_rows(ticker: str, expected_inputs: list[str]) -> list[dict]:
    rows = []
    for record_path in sorted((paths.DATASET / ticker).glob("*.json")):
        record = read_json(record_path)
        inputs = record.get("chart", {}).get("inputs", [])
        if [item.get("type") for item in inputs] != expected_inputs:
            continue
        image_paths = [paths.ROOT / item["path"] for item in inputs]
        if len(image_paths) != 2 or any(not path.exists() for path in image_paths):
            continue
        prices = record["prices"]
        forward_return = float(prices["close_t5"]) / float(prices["close_t0"]) - 1.0
        rows.append({
            "ticker": ticker,
            "date": record["date"],
            "image_paths": image_paths,
            "forward_return": forward_return,
        })
    return rows


def assign_train_only_labels(rows: list[dict], train_end: int,
                             low_q: float, high_q: float) -> tuple[float, float]:
    train_returns = np.asarray([row["forward_return"] for row in rows[:train_end]])
    bearish, bullish = np.quantile(train_returns, [low_q, high_q]).tolist()
    for row in rows:
        value = row["forward_return"]
        row["label"] = 0 if value < bearish else 2 if value > bullish else 1
    return float(bearish), float(bullish)


def chronological_split(rows: list[dict], train_ratio: float, val_ratio: float,
                        gap: int) -> tuple[list[dict], list[dict], list[dict], int, int]:
    n = len(rows)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_rows = rows[:train_end]
    val_rows = rows[train_end + gap:val_end]
    test_rows = rows[val_end + gap:]
    if min(len(train_rows), len(val_rows), len(test_rows)) == 0:
        raise ValueError(
            f"切分後有空集合：N={n}, train={len(train_rows)}, "
            f"validation={len(val_rows)}, test={len(test_rows)}, gap={gap}"
        )
    return train_rows, val_rows, test_rows, train_end, val_end


class ChartPairDataset(Dataset):
    def __init__(self, rows: list[dict], processor):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        images = []
        for path in row["image_paths"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return pixels, int(row["label"]), row["ticker"], row["date"]


class FrozenFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 tickers: list[str]):
        self.features = features
        self.labels = labels
        self.tickers = tickers

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index], self.tickers[index]


def make_loader(rows: list[dict], processor, batch_size: int,
                shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ChartPairDataset(rows, processor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def metric_payload(labels: list[int], logits: np.ndarray) -> dict:
    labels_array = np.asarray(labels)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions)),
        "macro_f1": float(f1_score(
            labels_array, predictions, labels=[0, 1, 2], average="macro", zero_division=0
        )),
        "log_loss": float(log_loss(labels_array, probabilities, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(
            labels_array, predictions, labels=[0, 1, 2]
        ).tolist(),
        "detail": classification_report(
            labels_array,
            predictions,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def metrics_with_tickers(labels: list[int], logits: np.ndarray,
                         tickers: list[str]) -> dict:
    result = metric_payload(labels, logits)
    if len(set(tickers)) > 1:
        labels_array = np.asarray(labels)
        tickers_array = np.asarray(tickers)
        result["by_ticker"] = {}
        for ticker in sorted(set(tickers)):
            mask = tickers_array == ticker
            result["by_ticker"][ticker] = metric_payload(
                labels_array[mask].tolist(), logits[mask]
            )
        result["mean_ticker_macro_f1"] = float(np.mean([
            metrics["macro_f1"] for metrics in result["by_ticker"].values()
        ]))
    return result


def evaluate_images(model, loader: DataLoader, device: str) -> tuple[dict, np.ndarray, list[int]]:
    model.eval()
    logits_parts, labels, tickers = [], [], []
    with torch.no_grad():
        for pixels, batch_labels, batch_tickers, _dates in loader:
            logits_parts.append(model(pixels.to(device), batch_tickers).cpu().numpy())
            labels.extend(batch_labels.tolist())
            tickers.extend(batch_tickers)
    logits = np.concatenate(logits_parts)
    return metrics_with_tickers(labels, logits, tickers), logits, labels


def extract_frozen_features(model, loader: DataLoader, device: str) -> FrozenFeatureDataset:
    model.eval()
    features, labels, tickers = [], [], []
    with torch.no_grad():
        for pixels, batch_labels, batch_tickers, _dates in loader:
            features.append(model.encode_cls(pixels.to(device)).cpu())
            labels.append(batch_labels)
            tickers.extend(batch_tickers)
    return FrozenFeatureDataset(torch.cat(features), torch.cat(labels), tickers)


def evaluate_features(model, loader: DataLoader, device: str) -> dict:
    model.eval()
    logits_parts, labels, tickers = [], [], []
    with torch.no_grad():
        for features, batch_labels, batch_tickers in loader:
            logits_parts.append(
                model.classify(features.to(device), batch_tickers).cpu().numpy()
            )
            labels.extend(batch_labels.tolist())
            tickers.extend(batch_tickers)
    return metrics_with_tickers(labels, np.concatenate(logits_parts), tickers)


def train_strategy(model, strategy: str, train_loader: DataLoader, val_loader: DataLoader,
                   device: str, epochs: int, patience: int, head_lr: float, vit_lr: float,
                   weight_decay: float, class_weights: torch.Tensor,
                   batch_size: int) -> tuple[dict, dict[str, torch.Tensor]]:
    info = model.configure_adaptation(strategy)
    model.to(device)
    head_params = list(model.head_parameters())
    head_ids = {id(parameter) for parameter in head_params}
    vit_params = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    groups = [{"params": head_params, "lr": head_lr}]
    if vit_params:
        groups.append({"params": vit_params, "lr": vit_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    started = time.perf_counter()
    frozen_train = frozen_val = None
    if strategy == "frozen":
        frozen_train = extract_frozen_features(model, train_loader, device)
        frozen_val = extract_frozen_features(model, val_loader, device)
        train_source = DataLoader(frozen_train, batch_size=batch_size, shuffle=True)
        val_source = DataLoader(frozen_val, batch_size=batch_size, shuffle=False)
    else:
        train_source, val_source = train_loader, val_loader

    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_source:
            if strategy == "frozen":
                features, labels, batch_tickers = batch
                logits = model.classify(features.to(device), batch_tickers)
            else:
                pixels, labels, batch_tickers, _dates = batch
                logits = model(pixels.to(device), batch_tickers)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)

        if strategy == "frozen":
            validation = evaluate_features(model, val_source, device)
        else:
            validation, _logits, _labels = evaluate_images(model, val_source, device)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total_items,
            "validation": validation,
        })
        print(
            f"[{strategy}] epoch={epoch}/{epochs} loss={total_loss / total_items:.4f} "
            f"val_macro_f1={validation['macro_f1']:.4f}",
            flush=True,
        )

        selection_score = validation.get(
            "mean_ticker_macro_f1", validation["macro_f1"]
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.trainable_state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"[{strategy}] early stopping：{patience} epochs 未改善", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"{strategy} 沒有產生 checkpoint")
    model.load_state_dict(best_state, strict=False)
    return ({
        "parameter_counts": info,
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }, best_state)


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(LABEL_NAMES[row["label"]] for row in rows)
    return {name: counts.get(name, 0) for name in LABEL_NAMES}


def state_path(out_dir: Path, strategy: str) -> Path:
    return out_dir / "checkpoints" / f"{strategy}.pt"


def rows_on_common_dates(rows_by_ticker: dict[str, list[dict]],
                         limit: int | None) -> tuple[dict[str, list[dict]], list[str]]:
    common = None
    for rows in rows_by_ticker.values():
        dates = {row["date"] for row in rows}
        common = dates if common is None else common & dates
    common_dates = sorted(common or [])
    if limit:
        common_dates = common_dates[:limit]
    common_set = set(common_dates)
    aligned = {
        ticker: [row for row in rows if row["date"] in common_set]
        for ticker, rows in rows_by_ticker.items()
    }
    return aligned, common_dates


def evaluate_checkpoint(checkpoint: Path, loader: DataLoader, device: str) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DualImageViTClassifier(
        payload["model_id"], dropout=float(payload["dropout"]),
        head_names=payload.get("head_names"),
    )
    model.configure_adaptation(payload["strategy"])
    model.load_state_dict(payload["state_dict"], strict=False)
    model.to(device)
    result, _logits, _labels = evaluate_images(model, loader, device)
    del model
    gc.collect()
    return result


def evaluate_probe_head(head, loader: DataLoader, device: str) -> dict:
    head.eval()
    logits_parts, labels = [], []
    with torch.no_grad():
        for features, batch_labels, _tickers in loader:
            logits_parts.append(head(features.to(device)).cpu().numpy())
            labels.extend(batch_labels.tolist())
    return metric_payload(labels, np.concatenate(logits_parts))


def held_out_probe(checkpoint: Path, train_loader: DataLoader, val_loader: DataLoader,
                   test_loader: DataLoader, device: str, epochs: int, patience: int,
                   head_lr: float, weight_decay: float, dropout: float,
                   seed: int, batch_size: int) -> dict:
    """凍結 checkpoint 的 ViT，僅為未見股票訓練一個新的診斷分類頭。"""
    set_seed(seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DualImageViTClassifier(
        payload["model_id"], dropout=float(payload["dropout"]),
        head_names=payload.get("head_names"),
    )
    model.configure_adaptation(payload["strategy"])
    model.load_state_dict(payload["state_dict"], strict=False)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.to(device).eval()

    train_features = extract_frozen_features(model, train_loader, device)
    val_features = extract_frozen_features(model, val_loader, device)
    test_features = extract_frozen_features(model, test_loader, device)
    train_source = DataLoader(train_features, batch_size=batch_size, shuffle=True)
    val_source = DataLoader(val_features, batch_size=batch_size, shuffle=False)
    test_source = DataLoader(test_features, batch_size=batch_size, shuffle=False)

    head = build_classification_head(
        model.hidden_size, n_classes=3, dropout=dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=head_lr, weight_decay=weight_decay
    )
    counts = np.bincount(train_features.labels.numpy(), minlength=3)
    weights = torch.tensor(
        len(train_features) / (3.0 * counts), dtype=torch.float32, device=device
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_items = 0
        for features, labels, _tickers in train_source:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features.to(device))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)
        validation = evaluate_probe_head(head, val_source, device)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total_items,
            "validation": validation,
        })
        if validation["macro_f1"] > best_score:
            best_score = validation["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    head.load_state_dict(best_state)
    test_result = evaluate_probe_head(head, test_source, device)
    del model, head, train_features, val_features, test_features
    gc.collect()
    return {
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "test": test_result,
        "history": history,
        "backbone_was_frozen": True,
    }


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=cfg["tickers"][0])
    parser.add_argument(
        "--train-tickers", nargs="+", default=None,
        help="多股票模式：共同訓練與 validation 的股票，例如 AAPL NVDA",
    )
    parser.add_argument(
        "--held-out-ticker", default=None,
        help="多股票模式：完全不參與權重更新，只做跨股票 test 的股票",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=["frozen", "last1", "last2", "last4"],
        help="可用 frozen、lastN、lora_rN（rank=N，掛最後四個 blocks）或 full",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--vit-lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--gap", type=int, default=cfg["label"]["horizon_trading_days"])
    parser.add_argument("--seed", type=int, default=cfg["seed"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="只供 smoke test 使用")
    parser.add_argument(
        "--run-name", default=None,
        help="自訂輸出子目錄；多 seed 實驗用來避免覆蓋既有 run",
    )
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="validation 選出最佳策略後，開啟 held-out test 一次",
    )
    parser.add_argument(
        "--evaluate-existing", action="store_true",
        help="不重訓；從既有 report/checkpoints 補算勝者與 frozen 的 test 指標",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs 與 patience 必須 >= 1")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("train_ratio、val_ratio 必須 > 0，且兩者合計 < 1")
    if args.limit is not None and args.limit < 40:
        parser.error("--limit 至少 40，避免加上 gap 後出現空的 validation/test")
    if args.run_name is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name
    ):
        parser.error("--run-name 只能包含英數字、底線、句點與連字號")
    if bool(args.train_tickers) != bool(args.held_out_ticker):
        parser.error("--train-tickers 與 --held-out-ticker 必須一起提供")

    set_seed(args.seed)
    device = choose_device(args.device)
    expected_inputs = cfg["chart"].get("vision_inputs", ["candlestick", "volume"])
    if len(expected_inputs) != 2:
        raise SystemExit(f"本實驗要求恰好兩張圖，目前為 {expected_inputs}")

    low_q = cfg["label"].get("quantile_low", 1 / 3)
    high_q = cfg["label"].get("quantile_high", 2 / 3)
    multi_stock = args.train_tickers is not None
    thresholds_by_ticker = {}
    split_counts_by_ticker = {}

    if multi_stock:
        train_tickers = [ticker.upper() for ticker in args.train_tickers]
        held_out_ticker = args.held_out_ticker.upper()
        if len(set(train_tickers)) != len(train_tickers):
            parser.error("--train-tickers 不可重複")
        if held_out_ticker in train_tickers:
            parser.error("held-out ticker 不可同時出現在 train tickers")
        requested_tickers = train_tickers + [held_out_ticker]
        rows_by_ticker = {
            ticker: load_rows(ticker, expected_inputs) for ticker in requested_tickers
        }
        rows_by_ticker, common_dates = rows_on_common_dates(rows_by_ticker, args.limit)
        if len(common_dates) < 30:
            raise SystemExit(f"多股票共同日期不足：{len(common_dates)}")

        reference_rows = rows_by_ticker[train_tickers[0]]
        ref_train, ref_val, ref_test, train_end, val_end = chronological_split(
            reference_rows, args.train_ratio, args.val_ratio, args.gap
        )
        for ticker, ticker_rows in rows_by_ticker.items():
            bearish, bullish = assign_train_only_labels(
                ticker_rows, train_end, low_q, high_q
            )
            thresholds_by_ticker[ticker] = {
                "bearish_threshold": bearish,
                "bullish_threshold": bullish,
            }
            split_counts_by_ticker[ticker] = {
                "train": label_counts(ticker_rows[:train_end]),
                "validation": label_counts(ticker_rows[train_end + args.gap:val_end]),
                "test": label_counts(ticker_rows[val_end + args.gap:]),
            }

        train_rows = [
            row
            for ticker in train_tickers
            for row in rows_by_ticker[ticker][:train_end]
        ]
        val_rows = [
            row
            for ticker in train_tickers
            for row in rows_by_ticker[ticker][train_end + args.gap:val_end]
        ]
        test_rows = rows_by_ticker[held_out_ticker][val_end + args.gap:]
        n_dates = len(common_dates)
        scope = f"{'_'.join(train_tickers)}_holdout_{held_out_ticker}"
        train_period = [ref_train[0]["date"], ref_train[-1]["date"]]
        validation_period = [ref_val[0]["date"], ref_val[-1]["date"]]
        test_period = [ref_test[0]["date"], ref_test[-1]["date"]]
    else:
        ticker = args.ticker.upper()
        train_tickers = [ticker]
        held_out_ticker = None
        rows = load_rows(ticker, expected_inputs)
        if args.limit:
            rows = rows[:args.limit]
        if len(rows) < 30:
            raise SystemExit(f"可用圖像資料不足：{len(rows)}")
        train_rows, val_rows, test_rows, train_end, _val_end = chronological_split(
            rows, args.train_ratio, args.val_ratio, args.gap
        )
        bearish, bullish = assign_train_only_labels(rows, train_end, low_q, high_q)
        thresholds_by_ticker[ticker] = {
            "bearish_threshold": bearish,
            "bullish_threshold": bullish,
        }
        split_counts_by_ticker[ticker] = {
            "train": label_counts(train_rows),
            "validation": label_counts(val_rows),
            "test": label_counts(test_rows),
        }
        n_dates = len(rows)
        scope = ticker
        train_period = [train_rows[0]["date"], train_rows[-1]["date"]]
        validation_period = [val_rows[0]["date"], val_rows[-1]["date"]]
        test_period = [test_rows[0]["date"], test_rows[-1]["date"]]

    try:
        processor = AutoImageProcessor.from_pretrained(
            cfg["encoders"]["vision"], local_files_only=True
        )
    except OSError:
        processor = AutoImageProcessor.from_pretrained(cfg["encoders"]["vision"])
    train_loader = make_loader(
        train_rows, processor, args.batch, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_loader(
        val_rows, processor, args.batch, shuffle=False, num_workers=args.num_workers
    )
    test_loader = make_loader(
        test_rows, processor, args.batch, shuffle=False, num_workers=args.num_workers
    )
    heldout_train_loader = heldout_val_loader = None
    if multi_stock:
        heldout_rows = rows_by_ticker[held_out_ticker]
        heldout_train_loader = make_loader(
            heldout_rows[:train_end], processor, args.batch,
            shuffle=False, num_workers=args.num_workers,
        )
        heldout_val_loader = make_loader(
            heldout_rows[train_end + args.gap:val_end], processor, args.batch,
            shuffle=False, num_workers=args.num_workers,
        )
    train_counts = np.bincount([row["label"] for row in train_rows], minlength=3)
    class_weights = torch.tensor(
        len(train_rows) / (3.0 * train_counts), dtype=torch.float32
    )

    suffix = args.run_name or ("smoke" if args.limit else "run")
    out_dir = paths.OUTPUTS / "experiments" / "vit_adaptation" / scope / suffix
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(
        f"[vit_adaptation] device={device}, dates={n_dates}, "
        f"train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)}, gap={args.gap}",
        flush=True,
    )

    report_path = out_dir / "report.json"
    if args.evaluate_existing:
        if not report_path.exists():
            raise SystemExit(f"找不到既有報告：{report_path}")
        report = read_json(report_path)
        selected = report["selected"]
        probe_results = {}
        if multi_stock:
            for strategy in dict.fromkeys([selected, "frozen"]):
                checkpoint = state_path(out_dir, strategy)
                if checkpoint.exists():
                    probe_results[strategy] = held_out_probe(
                        checkpoint, heldout_train_loader, heldout_val_loader,
                        test_loader, device, args.epochs, args.patience,
                        args.head_lr, args.weight_decay, args.dropout,
                        args.seed, args.batch,
                    )
            selected_test = probe_results[selected]["test"]
            frozen_test = (
                probe_results["frozen"]["test"] if "frozen" in probe_results else None
            )
        else:
            selected_test = evaluate_checkpoint(
                state_path(out_dir, selected), test_loader, device
            )
            frozen_test = None
            if state_path(out_dir, "frozen").exists():
                frozen_test = evaluate_checkpoint(
                    state_path(out_dir, "frozen"), test_loader, device
                )
        report["test_was_opened"] = True
        report["selected_test"] = selected_test
        report["frozen_baseline_test"] = frozen_test
        report["held_out_probe_results"] = probe_results if multi_stock else None
        report["selected_vs_frozen_test_delta"] = (
            {
                "accuracy": selected_test["accuracy"] - frozen_test["accuracy"],
                "balanced_accuracy": (
                    selected_test["balanced_accuracy"] - frozen_test["balanced_accuracy"]
                ),
                "macro_f1": selected_test["macro_f1"] - frozen_test["macro_f1"],
                "log_loss": selected_test["log_loss"] - frozen_test["log_loss"],
            }
            if frozen_test is not None else None
        )
        write_json(report, report_path)
        frozen_score = (
            f"{frozen_test['macro_f1']:.4f}" if frozen_test is not None else "N/A"
        )
        print(
            f"[vit_adaptation] existing selected={selected}, "
            f"test_macro_f1={selected_test['macro_f1']:.4f}, "
            f"frozen_test_macro_f1={frozen_score}",
            flush=True,
        )
        print(f"[vit_adaptation] report -> {report_path}", flush=True)
        return

    results = {}
    for strategy in args.strategies:
        set_seed(args.seed)
        model = DualImageViTClassifier(
            cfg["encoders"]["vision"], dropout=args.dropout,
            head_names=train_tickers if multi_stock else None,
        )
        result, best_state = train_strategy(
            model=model,
            strategy=strategy,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            head_lr=args.head_lr,
            vit_lr=args.vit_lr,
            weight_decay=args.weight_decay,
            class_weights=class_weights,
            batch_size=args.batch,
        )
        results[strategy] = result
        torch.save({
            "model_id": cfg["encoders"]["vision"],
            "strategy": strategy,
            "dropout": args.dropout,
            "head_names": train_tickers if multi_stock else None,
            "state_dict": best_state,
        }, state_path(out_dir, strategy))
        del model, best_state
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    selected = max(
        results,
        key=lambda name: results[name]["best_validation"].get(
            "mean_ticker_macro_f1",
            results[name]["best_validation"]["macro_f1"],
        ),
    )
    selected_test = None
    frozen_test = None
    probe_results = {}
    if args.evaluate_test:
        if multi_stock:
            for strategy in dict.fromkeys([selected, "frozen"]):
                checkpoint = state_path(out_dir, strategy)
                if checkpoint.exists():
                    probe_results[strategy] = held_out_probe(
                        checkpoint, heldout_train_loader, heldout_val_loader,
                        test_loader, device, args.epochs, args.patience,
                        args.head_lr, args.weight_decay, args.dropout,
                        args.seed, args.batch,
                    )
            selected_test = probe_results[selected]["test"]
            frozen_test = (
                probe_results["frozen"]["test"] if "frozen" in probe_results else None
            )
        else:
            selected_test = evaluate_checkpoint(
                state_path(out_dir, selected), test_loader, device
            )
            if state_path(out_dir, "frozen").exists():
                frozen_test = evaluate_checkpoint(
                    state_path(out_dir, "frozen"), test_loader, device
                )

    report = {
        "mode": "multi_stock_holdout" if multi_stock else "single_stock",
        "ticker": None if multi_stock else train_tickers[0],
        "train_tickers": train_tickers,
        "held_out_ticker": held_out_ticker,
        "n_common_dates": n_dates,
        "model_id": cfg["encoders"]["vision"],
        "vision_inputs": expected_inputs,
        "strategies": args.strategies,
        "seed": args.seed,
        "run_name": suffix,
        "device": device,
        "hyperparameters": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch": args.batch,
            "head_lr": args.head_lr,
            "vit_lr": args.vit_lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
        },
        "labeling": {
            "horizon_trading_days": cfg["label"]["horizon_trading_days"],
            "threshold_source": "train_only",
            "thresholds_by_ticker": thresholds_by_ticker,
            "held_out_ticker_past_returns_used_only_for_label_threshold": multi_stock,
        },
        "split": {
            "gap": args.gap,
            "n_train": len(train_rows),
            "n_validation": len(val_rows),
            "n_test": len(test_rows),
            "train_period": train_period,
            "validation_period": validation_period,
            "test_period": test_period,
            "class_counts": {
                "train": label_counts(train_rows),
                "validation": label_counts(val_rows),
                "test": label_counts(test_rows),
            },
            "class_counts_by_ticker": split_counts_by_ticker,
        },
        "selection_metric": (
            "validation.mean_ticker_macro_f1"
            if multi_stock else "validation.macro_f1"
        ),
        "results": results,
        "selected": selected,
        "test_was_opened": bool(args.evaluate_test),
        "selected_test": selected_test,
        "frozen_baseline_test": frozen_test,
        "held_out_probe_results": probe_results if multi_stock else None,
        "selected_vs_frozen_test_delta": (
            {
                "accuracy": selected_test["accuracy"] - frozen_test["accuracy"],
                "balanced_accuracy": (
                    selected_test["balanced_accuracy"] - frozen_test["balanced_accuracy"]
                ),
                "macro_f1": selected_test["macro_f1"] - frozen_test["macro_f1"],
                "log_loss": selected_test["log_loss"] - frozen_test["log_loss"],
            }
            if selected_test is not None and frozen_test is not None else None
        ),
        "limitations": [
            "單一 seed 結果只供第一輪篩選，正式結論需對最佳候選跑 3-5 seeds",
            "目前只量測 ViT-only 分類；尚未重新產生 H_v 或放回 fusion",
            (
                "held-out 股票只訓練新的分類 probe，ViT backbone 不接收 held-out 股票梯度"
                if multi_stock else "單一股票模式未測跨股票泛化"
            ),
        ],
    }
    write_json(report, report_path)
    print(
        f"[vit_adaptation] selected={selected}, "
        f"val_macro_f1={results[selected]['best_validation']['macro_f1']:.4f}, "
        f"test_macro_f1={selected_test['macro_f1']:.4f} "
        if selected_test else
        f"[vit_adaptation] selected={selected}, "
        f"val_macro_f1={results[selected]['best_validation']['macro_f1']:.4f}, test=LOCKED",
        flush=True,
    )
    print(f"[vit_adaptation] report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
