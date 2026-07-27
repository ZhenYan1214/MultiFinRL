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

import numpy as np

from shared import paths
from shared.utils import read_json, write_json

LABEL_TO_ID = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def build_index(ticker: str) -> dict:
    """掃 data/outputs/z_fused/{ticker}/*.npy，比對 A 的每日 JSON 取 label/報酬，依日期排序彙整。

    回傳空 dict 代表找不到任何可彙整的資料（該股票尚未產出 Z_fused）。
    """
    z_dir = paths.OUTPUTS / "z_fused" / ticker
    files = sorted(z_dir.glob("*.npy"))  # 檔名即日期，字串排序 = 時間排序（YYYY-MM-DD）

    dates, zs, labels, next_returns = [], [], [], []
    for f in files:
        date = f.stem
        record_file = paths.daily_json(ticker, date)
        if not record_file.exists():
            continue  # 找不到對應 A 的標籤就跳過，避免索引裡出現無 label 的天
        record = read_json(record_file)
        prices = record["prices"]
        dates.append(date)
        zs.append(np.load(f))
        labels.append(LABEL_TO_ID[record["label"]])
        next_returns.append(prices["future_closes"][0] / prices["close_t0"] - 1)

    if not dates:
        return {}
    return {
        "dates": np.array(dates),
        "z": np.stack(zs).astype(np.float32),
        "label": np.array(labels, dtype=np.int64),
        "return_next": np.array(next_returns, dtype=np.float32),
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
        "label_map": ID_TO_LABEL,
    }
    write_json(meta, out_dir / f"{ticker}_index.meta.json")
    print(f"[consolidate] {ticker}: {meta['n_days']} days ({meta['date_range'][0]} ~ "
          f"{meta['date_range'][1]}) -> {npz_path}")


def load_index(ticker: str) -> dict | None:
    """給 classifier.py / backtest.py / train_ppo.py 用。

    索引檔不存在時回傳 None（呼叫端應 fallback 成逐日掃描，見各檔案的 load_* 函式）。
    """
    path = paths.OUTPUTS / "z_fused" / f"{ticker}_index.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=False)
    return {k: npz[k] for k in npz.files}


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
