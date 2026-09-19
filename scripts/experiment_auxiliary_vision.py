"""比較雙圖 ViT 的第二張圖：volume / RSI / SMA / MACD。

設計重點：
- 第一張 K 線、H_t、H_r 全部固定，只替換 H_v 的第二張圖。
- 所有候選使用共同日期、相同 seed / hyperparameters。
- fusion 只用前 70% 日期的 label 訓練；中間 15% validation 選候選；最後 15%
  held-out test 只評估 validation 勝出者，避免現行正式 pipeline 的 test-label leakage。
- 技術圖的 ViT token 快取放在 data/outputs/experiments/auxiliary_vision/，可續跑。

前置：先用 chart_generator 產生 technical/rsi、technical/sma、technical/macd 圖。

用法：
    python scripts/experiment_auxiliary_vision.py --ticker AAPL
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import paths
from shared.utils import load_config, read_json, write_json
from module_b_encoder.encoders.vision_encoder import VisionEncoder
from module_c_fusion.fusion.model import build_model
from module_c_fusion.fusion.train import LABEL_TO_ID, load_day, train


LABEL_NAMES = ["BEARISH", "NEUTRAL", "BULLISH"]


class LazyBatches:
    """每個 epoch 才按日期載入向量，避免把數 GB 的 H_t/H_r 全塞進 RAM。"""

    def __init__(self, rows, batch_size, auxiliary_loader):
        self.rows = rows
        self.batch_size = batch_size
        self.auxiliary_loader = auxiliary_loader

    def __len__(self):
        return (len(self.rows) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, len(self.rows), self.batch_size):
            chunk = self.rows[start:start + self.batch_size]
            h_vs, h_ts, h_rs, labels = [], [], [], []
            for date, label in chunk:
                h_v, h_t, h_r = load_day(self.ticker, date)
                auxiliary = self.auxiliary_loader(date, h_v)
                h_vs.append(np.stack([h_v[0], auxiliary]))
                h_ts.append(h_t)
                h_rs.append(h_r)
                labels.append(label)
            yield np.stack(h_vs), np.stack(h_ts), np.stack(h_rs), np.asarray(labels)

    @property
    def ticker(self):
        return self._ticker

    @ticker.setter
    def ticker(self, value):
        self._ticker = value


def available_rows(ticker: str, candidates: list[str]) -> list[tuple[str, int]]:
    """只保留 dataset、production vectors 與所有候選圖都存在的共同日期。"""
    rows = []
    for record_path in sorted((paths.DATASET / ticker).glob("*.json")):
        date = record_path.stem
        vector_dir = paths.vector_dir(ticker, date)
        if not (vector_dir / "index.json").exists():
            continue
        if any(
            not (paths.RAW_CHARTS / ticker / "technical" / name / f"{date}.png").exists()
            for name in candidates if name != "volume"
        ):
            continue
        label = LABEL_TO_ID[read_json(record_path)["label"]]
        rows.append((date, label))
    return rows


def encode_candidate_cache(vision: VisionEncoder, ticker: str, candidate: str,
                           rows: list[tuple[str, int]], out_dir: Path,
                           encode_batch_size: int) -> np.ndarray:
    """將一種 technical 圖編碼為 [N,197,768] memmap；已有完整快取就直接讀。"""
    candidate_dir = out_dir / "embeddings" / candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    array_path = candidate_dir / "H_aux.npy"
    dates_path = candidate_dir / "dates.json"
    dates = [date for date, _ in rows]

    if array_path.exists() and dates_path.exists() and read_json(dates_path) == dates:
        cached = np.load(array_path, mmap_mode="r")
        if cached.shape == (len(rows), 197, 768):
            print(f"[auxiliary_vision] {candidate}: 使用既有 ViT 快取 {cached.shape}")
            return cached

    output = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.float32, shape=(len(rows), 197, 768)
    )
    for start in range(0, len(rows), encode_batch_size):
        chunk = rows[start:start + encode_batch_size]
        image_paths = [
            paths.RAW_CHARTS / ticker / "technical" / candidate / f"{date}.png"
            for date, _ in chunk
        ]
        output[start:start + len(chunk)] = vision.encode_batch(image_paths)
        print(
            f"[auxiliary_vision] {candidate} encoding "
            f"{min(start + len(chunk), len(rows))}/{len(rows)}",
            end="\r", flush=True,
        )
    print()
    output.flush()
    write_json(dates, dates_path)
    return np.load(array_path, mmap_mode="r")


def metrics(y_true, pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(
            y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0
        )),
        "detail": classification_report(
            y_true, pred, labels=[0, 1, 2], target_names=LABEL_NAMES,
            output_dict=True, zero_division=0,
        ),
    }


def encode_z(model, rows, batches: LazyBatches, device: str) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for h_v, h_t, h_r, _labels in batches:
            z = model(
                torch.from_numpy(h_v).to(device),
                torch.from_numpy(h_t).to(device),
                torch.from_numpy(h_r).to(device),
            ).cpu().numpy()
            outputs.append(z)
    result = np.concatenate(outputs)
    if result.shape[0] != len(rows):
        raise RuntimeError(f"Z_fused 筆數異常: {result.shape[0]} != {len(rows)}")
    return result


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--candidates", nargs="+", default=["volume", "rsi", "sma", "macd"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--encode_batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    unsupported = set(args.candidates) - {"volume", "rsi", "sma", "macd"}
    if unsupported:
        ap.error(f"不支援的候選: {sorted(unsupported)}")

    rows = available_rows(args.ticker, args.candidates)
    if len(rows) < 100:
        raise SystemExit("共同日期不足；請先產生各候選 technical charts")
    train_end = int(len(rows) * 0.70)
    val_end = int(len(rows) * 0.85)
    train_rows, val_rows, test_rows = rows[:train_end], rows[train_end:val_end], rows[val_end:]
    print(
        f"[auxiliary_vision] common={len(rows)}, train/val/test="
        f"{len(train_rows)}/{len(val_rows)}/{len(test_rows)}, "
        f"test={test_rows[0][0]}~{test_rows[-1][0]}"
    )

    out_dir = paths.OUTPUTS / "experiments" / "auxiliary_vision" / args.ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    technical = [name for name in args.candidates if name != "volume"]
    caches = {}
    if technical:
        vision = VisionEncoder(cfg["encoders"]["vision"], n_images=2)
        for candidate in technical:
            caches[candidate] = encode_candidate_cache(
                vision, args.ticker, candidate, rows, out_dir, args.encode_batch
            )
        del vision

    date_to_index = {date: i for i, (date, _label) in enumerate(rows)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_labels = np.asarray([label for _, label in train_rows])
    counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    weights = len(train_rows) / (3.0 * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    results = {}
    selected_payload = None
    baseline_z = None

    for candidate in args.candidates:
        def auxiliary_loader(date, production_h_v, name=candidate):
            if name == "volume":
                return production_h_v[1]
            return np.asarray(caches[name][date_to_index[date]])

        torch.manual_seed(cfg["seed"])
        model = build_model(cfg)
        train_batches = LazyBatches(train_rows, args.batch, auxiliary_loader)
        train_batches.ticker = args.ticker
        _head, losses = train(
            model, train_batches, args.epochs, args.lr, device,
            class_weights=class_weights,
        )

        all_batches = LazyBatches(rows, args.batch, auxiliary_loader)
        all_batches.ticker = args.ticker
        z = encode_z(model, rows, all_batches, device)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(z[:train_end], train_labels)
        val_labels = [label for _, label in val_rows]
        val_result = metrics(val_labels, clf.predict(z[train_end:val_end]))
        results[candidate] = {"losses": losses, "validation": val_result}
        print(
            f"[auxiliary_vision] {candidate}: val accuracy={val_result['accuracy']:.4f}, "
            f"macro_f1={val_result['macro_f1']:.4f}"
        )
        if selected_payload is None or val_result["macro_f1"] > selected_payload[0]:
            selected_payload = (val_result["macro_f1"], candidate, z)
        if candidate == "volume":
            baseline_z = z.copy()

        del model, z
        if device == "cuda":
            torch.cuda.empty_cache()

    _best_score, selected, selected_z = selected_payload
    # 候選已由 validation 選定後，分類 probe 才用 train+validation refit 並看一次 test。
    train_val_labels = [label for _, label in rows[:val_end]]
    test_labels = [label for _, label in test_rows]
    final_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    final_clf.fit(selected_z[:val_end], train_val_labels)
    test_result = metrics(test_labels, final_clf.predict(selected_z[val_end:]))
    baseline_test = None
    if baseline_z is not None:
        baseline_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        baseline_clf.fit(baseline_z[:val_end], train_val_labels)
        baseline_test = metrics(test_labels, baseline_clf.predict(baseline_z[val_end:]))

    report = {
        "ticker": args.ticker,
        "candidates": args.candidates,
        "n_common": len(rows),
        "split": {
            "n_train": len(train_rows), "n_validation": len(val_rows), "n_test": len(test_rows),
            "train_period": [train_rows[0][0], train_rows[-1][0]],
            "validation_period": [val_rows[0][0], val_rows[-1][0]],
            "test_period": [test_rows[0][0], test_rows[-1][0]],
        },
        "seed": cfg["seed"], "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "selection_metric": "validation.macro_f1",
        "validation_results": results,
        "selected": selected,
        "selected_test": test_result,
        "volume_baseline_test": baseline_test,
        "limitations": [
            "H_t/H_r 固定為導入技術圖前的 production vectors，只隔離第二張圖直接進 fusion 的效果",
            "只跑單一 seed=42，未做多 seed 平均或顯著性檢驗",
        ],
    }
    write_json(report, out_dir / "report.json")
    print(
        f"[auxiliary_vision] selected={selected}; test accuracy={test_result['accuracy']:.4f}, "
        f"macro_f1={test_result['macro_f1']:.4f} -> {out_dir / 'report.json'}"
    )


if __name__ == "__main__":
    main()
