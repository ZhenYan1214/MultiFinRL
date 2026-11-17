"""B 模組主程式：對每個交易日產出 H_v（兩張圖）與 H_t（K 線文字敘述）。

輸入：data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json（讀入前先 validate）
      data/labels/chart_caption/{TICKER}.json（A 的 module_a_data.chart_caption 產生）
輸出：data/vectors/{TICKER}/{YYYY-MM-DD}/{H_v,H_t}.npy + index.json
寫出 index.json 前必須通過 shared.schemas.validate_vector_index()。

- H_v [2,197,768]：K 線圖＋成交量圖進凍結 ViT。已經有 H_v.npy 且 encoder／圖別相同時直接沿用，
  不重算（加 --recompute_vision 才重算）。
- H_t [512,768]：K 線文字敘述進凍結 FinBERT（decisions.md #109）。新聞不再編碼（#110）。
- 舊架構每天的 H_r（財報檢索）拿掉；RAG-1 的型態定義所有日子共用，存在
  data/vectors/_patterns/patterns.npz（module_b_encoder.rag.pattern_kb），檢索在融合層裡做。

用法：
    python -m module_b_encoder.generate_vectors --fake --n 10                 # 假資料測通流程
    python -m module_b_encoder.generate_vectors --ticker AAPL                 # 真實資料
"""
import argparse
import datetime as dt
import time

import numpy as np

from shared import paths, schemas
from shared.utils import load_config, read_json, write_json


def save_vectors(ticker: str, date: str, h_v, h_t, *, vision_id: str, text_id: str,
                 source_json: str, vision_input_types: list[str], caption: dict) -> None:
    """兩個 .npy + index.json 落地（validate 後）。h_v 為 None 代表沿用既有的 H_v.npy。"""
    out_dir = paths.vector_dir(ticker, date)
    out_dir.mkdir(parents=True, exist_ok=True)
    if h_v is not None:
        np.save(out_dir / "H_v.npy", h_v)
    np.save(out_dir / "H_t.npy", h_t)
    n_tokens = int((np.abs(h_t).sum(axis=-1) > 0).sum())
    index = {
        "ticker": ticker,
        "date": date,
        "vectors": {
            "H_v": {"file": "H_v.npy", "shape": [2, 197, 768], "dtype": "float32",
                    "encoder": vision_id, "input_types": vision_input_types},
            "H_t": {"file": "H_t.npy", "shape": list(h_t.shape), "dtype": "float32",
                    "encoder": text_id, "source": "chart_caption", "n_tokens": n_tokens},
        },
        "caption": {"model": caption.get("model"),
                    "pattern": caption["pattern"], "volume_label": caption["volume_label"]},
        "source_json": source_json,
    }
    schemas.validate_vector_index(index)
    write_json(index, out_dir / "index.json")


def _reuse_h_v(out_dir, vision_id: str, input_types: list[str]) -> bool:
    """已有 H_v.npy，且 index.json 記錄的 encoder 與圖別順序相同 -> 可以沿用。"""
    if not (out_dir / "H_v.npy").exists() or not (out_dir / "index.json").exists():
        return False
    old = read_json(out_dir / "index.json")["vectors"].get("H_v", {})
    return old.get("encoder") == vision_id and old.get("input_types") == input_types


def run_fake(cfg, ticker: str, n: int) -> None:
    """不需要 A 的資料、不需要 GPU，隨機向量測通儲存格式。"""
    from module_b_encoder.encoders.vision_encoder import fake_h_v
    from module_b_encoder.encoders.text_encoder import fake_h_t
    from shared.patterns import PATTERN_IDS

    input_types = cfg["chart"]["vision_inputs"]
    d0 = dt.date(2021, 3, 1)
    fake_ticker = f"_{ticker}_FAKE"   # 絕不可寫入正式 ticker namespace
    for i in range(n):
        date = (d0 + dt.timedelta(days=i)).isoformat()
        caption = {"pattern": PATTERN_IDS[i % len(PATTERN_IDS)], "volume_label": "FLAT",
                   "model": "fake"}
        save_vectors(fake_ticker, date, fake_h_v(seed=i), fake_h_t(seed=i),
                     vision_id=cfg["encoders"]["vision"], text_id=cfg["encoders"]["text"],
                     source_json=f"FAKE://{ticker}/{date}", vision_input_types=input_types,
                     caption=caption)
    print(f"[generate_vectors] FAKE {ticker}: {n} days -> {paths.VECTORS / fake_ticker}")


def run_real(cfg, ticker: str, limit: int | None,
             recompute_vision: bool, allow_missing: bool) -> None:
    from module_a_data.chart_caption import compose_text, load_captions

    files = sorted((paths.DATASET / ticker).glob("*.json"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"找不到 A 的資料: {paths.DATASET / ticker}，先跑 module_a_data.build_dataset")
    captions = load_captions(ticker)
    missing = [f.stem for f in files if f.stem not in captions]
    if missing and not allow_missing:
        raise SystemExit(f"{len(missing)} 天沒有 K 線文字敘述（例如 {missing[:3]}），先補跑 "
                         f"module_a_data.chart_caption --ticker {ticker}；或加 --allow_missing 跳過")
    files = [f for f in files if f.stem in captions]

    from module_b_encoder.encoders.text_encoder import TextEncoder
    vision_id, input_types = cfg["encoders"]["vision"], cfg["chart"]["vision_inputs"]
    text = TextEncoder(cfg["encoders"]["text"])
    vision = None
    n_reused = 0
    for i, f in enumerate(files, 1):
        record = read_json(f)
        schemas.validate_daily_record(record)
        date = record["date"]
        chart_inputs = record["chart"]["inputs"]
        types = [item["type"] for item in chart_inputs]
        if types != input_types:
            raise ValueError(f"{f} 的 chart.inputs 順序 {types} 與 config {input_types} 不一致")

        h_v = None
        if recompute_vision or not _reuse_h_v(paths.vector_dir(ticker, date), vision_id, input_types):
            if vision is None:
                from module_b_encoder.encoders.vision_encoder import VisionEncoder
                vision = VisionEncoder(vision_id, n_images=len(input_types))
            h_v = vision.encode([paths.ROOT / item["path"] for item in chart_inputs])
        else:
            n_reused += 1
        caption = captions[date]
        h_t = text.encode(compose_text(caption))
        save_vectors(ticker, date, h_v, h_t, vision_id=vision_id, text_id=text.model_id,
                     source_json=str(f.relative_to(paths.ROOT)).replace("\\", "/"),
                     vision_input_types=types, caption=caption)
        print(f"[generate_vectors] {ticker} {date} done（{i}/{len(files)}）", end="\r", flush=True)
    print()
    print(f"[generate_vectors] {ticker}: {len(files)} days -> {paths.VECTORS / ticker}"
          f"（H_v 沿用 {n_reused} 天、重算 {len(files) - n_reused} 天）")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--fake", action="store_true", help="隨機向量測通流程")
    ap.add_argument("--n", type=int, default=10, help="--fake 時產幾天")
    ap.add_argument("--limit", type=int, default=None, help="真實模式只跑前 N 天")
    ap.add_argument("--recompute_vision", action="store_true", help="已有 H_v.npy 也重算")
    ap.add_argument("--allow_missing", action="store_true", help="沒有敘述的日子跳過，不報錯")
    args = ap.parse_args()

    started_at = time.perf_counter()
    try:
        if args.fake:
            run_fake(cfg, args.ticker, args.n)
        else:
            run_real(cfg, args.ticker, args.limit,
                     args.recompute_vision, args.allow_missing)
    finally:
        elapsed = time.perf_counter() - started_at
        print(f"[generate_vectors] 總執行時間：{elapsed:.1f} 秒（{elapsed / 60:.1f} 分鐘）")


if __name__ == "__main__":
    main()
