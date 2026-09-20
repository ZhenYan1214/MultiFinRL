"""彙整 C 的逐日 Z_fused，產出單一索引檔，供分類驗證/回測/RL 訓練直接讀取。

逐日檔案（data/outputs/z_fused/{TICKER}/{date}.npy）保留，供除錯與可解釋性分析使用；
本檔額外產出一份彙整檔：

    data/outputs/z_fused/{TICKER}_index.npz
        dates:        [N] 字串陣列，YYYY-MM-DD，已依日期排序
        z:            [N, z_dim] float32
        label:        [N] int，0=BEARISH 1=NEUTRAL 2=BULLISH（對照 A 的 label）
        return_next:  [N] float，t -> t+1 的實際報酬（RL / 回測用，來自 A 的 future_closes[0]）

    data/outputs/z_fused/{TICKER}_index.meta.json   # 人類可讀的摘要

train.py 跑完整段時間範圍、產完所有逐日 Z_fused 後，會自動呼叫本檔重建索引；
也可單獨執行（例如手動補產某支股票的索引）：

    python -m module_c_fusion.fusion.consolidate --ticker AAPL
"""
import argparse
import hashlib

import numpy as np

from shared import paths
from shared.utils import read_json, write_json
from shared.temporal_split import split_summary

LABEL_TO_ID = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def build_index(ticker: str, split_overrides: dict[str, str] | None = None) -> dict:
    """掃 data/outputs/z_fused/{ticker}/*.npy，比對 A 的每日 JSON 取 label/報酬，依日期排序彙整。

    回傳空 dict 代表找不到任何可彙整的資料（該股票尚未產出 Z_fused）。
    """
    z_dir = paths.OUTPUTS / "z_fused" / ticker
    files = sorted(z_dir.glob("*.npy"))  # 檔名即日期，字串排序 = 時間排序（YYYY-MM-DD）
    manifest_path = paths.dataset_manifest(ticker)
    if not manifest_path.exists():
        raise ValueError(f"缺少 {manifest_path}；請先重跑 build_dataset")
    manifest = read_json(manifest_path)
    if manifest.get("protocol_version") != 2 or not manifest.get("records_sha256"):
        raise ValueError(f"{manifest_path} 是舊版格式；請先重跑 build_dataset")
    allowed = set(manifest["dates"])
    files = [file for file in files if file.stem in allowed]
    run_path = z_dir / "run.json"
    if not run_path.exists():
        raise ValueError(f"缺少 {run_path}；無法確認逐日 Z_fused 版本，請重跑 fusion.train")
    run = read_json(run_path)
    if (run.get("ticker") != ticker
            or run.get("dataset_records_sha256") != manifest["records_sha256"]
            or set(run.get("dates", [])) != {file.stem for file in files}):
        raise ValueError(f"{run_path} 與目前 dataset/Z_fused 不一致；請重跑 fusion.train")

    dates, zs, labels, next_returns, splits, target_dates = [], [], [], [], [], []
    for f in files:
        date = f.stem
        record_file = paths.daily_json(ticker, date)
        if not record_file.exists():
            raise ValueError(f"manifest/Z_fused 指向不存在的 {record_file}；請重跑上游")
        record = read_json(record_file)
        prices = record["prices"]
        if "split" not in record or "target_date" not in prices:
            raise ValueError(
                f"{record_file} 缺少 split/target_date；請先重跑 module_a_data.build_dataset"
            )
        dates.append(date)
        zs.append(np.load(f))
        labels.append(LABEL_TO_ID[record["label"]])
        next_returns.append(prices["future_closes"][0] / prices["close_t0"] - 1)
        splits.append((split_overrides or {}).get(date, record["split"]))
        target_dates.append(prices["target_date"])

    if not dates:
        return {}
    z_array = np.stack(zs).astype(np.float32)
    return {
        "dates": np.array(dates),
        "z": z_array,
        "label": np.array(labels, dtype=np.int64),
        "return_next": np.array(next_returns, dtype=np.float32),
        "split": np.array(splits),
        "label_target_date": np.array(target_dates),
        "dataset_records_sha256": np.array(manifest["records_sha256"]),
        "z_fused_sha256": np.array(hashlib.sha256(z_array.tobytes()).hexdigest()),
    }


def save_index(ticker: str, data: dict) -> None:
    out_dir = paths.OUTPUTS / "z_fused"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{ticker}_index.npz"
    np.savez(npz_path, **data)

    meta = {
        "ticker": ticker,
        "n_days": int(len(data["dates"])),
        "date_range": [str(data["dates"][0]), str(data["dates"][-1])],
        "z_dim": int(data["z"].shape[1]),
        "dataset_records_sha256": str(data["dataset_records_sha256"]),
        "z_fused_sha256": str(data["z_fused_sha256"]),
        "label_map": ID_TO_LABEL,
        "split": split_summary(data["dates"].tolist(), data["split"].tolist()),
        "protocol": {
            "fusion_fit_split": "train",
            "downstream_selection_split": "validation",
            "final_evaluation_split": "test",
        },
    }
    write_json(meta, out_dir / f"{ticker}_index.meta.json")
    print(f"[consolidate] {ticker}: {meta['n_days']} days ({meta['date_range'][0]} ~ "
          f"{meta['date_range'][1]}) -> {npz_path}")


def load_index(ticker: str) -> dict | None:
    """給下游使用；不存在時回傳 None，存在時強制驗證 dataset 指紋。"""
    path = paths.OUTPUTS / "z_fused" / f"{ticker}_index.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=False)
    data = {k: npz[k] for k in npz.files}
    manifest_path = paths.dataset_manifest(ticker)
    if not manifest_path.exists():
        raise ValueError(f"缺少 {manifest_path}；無法驗證 Z_fused 是否對應目前 dataset")
    expected = read_json(manifest_path).get("records_sha256")
    actual = str(data.get("dataset_records_sha256", ""))
    if not expected or actual != expected:
        raise ValueError(
            f"{path} 與目前 dataset 版本不一致；請重跑 fusion.train --ticker {ticker}"
        )
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    args = ap.parse_args()
    data = build_index(args.ticker)
    if not data:
        raise SystemExit(
            f"找不到可彙整的 Z_fused: data/outputs/z_fused/{args.ticker}/，"
            "先跑 module_c_fusion.fusion.train"
        )
    save_index(args.ticker, data)


if __name__ == "__main__":
    main()
