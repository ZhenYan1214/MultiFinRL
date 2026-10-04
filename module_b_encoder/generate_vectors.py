"""B 模組主程式：對每個交易日產出 H_v / H_t / H_r 三個向量。

輸入：data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json（讀入前先 validate）
輸出：data/vectors/{TICKER}/{YYYY-MM-DD}/{H_v,H_t,H_r}.npy + index.json
寫出 index.json 前必須通過 shared.schemas.validate_vector_index()。

用法：
    python -m module_b_encoder.generate_vectors --fake --n 10       # 第一階段：假資料測通流程
    python -m module_b_encoder.generate_vectors --ticker AAPL       # 第二階段：真實資料
"""
import argparse
import datetime as dt
import hashlib
import time

import numpy as np

from shared import paths, schemas
from shared.utils import load_config, read_json, write_json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 沒裝 python-dotenv 時仍可使用系統環境變數


def save_vectors(ticker: str, date: str, h_v, h_t, h_r, chunk_ids: list[str],
                 vision_id: str, text_id: str, top_k: int, source_json: str,
                 vision_input_types: list[str] | None = None,
                 context_sources: list[str] | None = None) -> None:
    """三個 .npy + index.json 落地（validate 後）。"""
    out_dir = paths.vector_dir(ticker, date)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "H_v.npy", h_v)
    np.save(out_dir / "H_t.npy", h_t)
    np.save(out_dir / "H_r.npy", h_r)
    index = {
        "ticker": ticker,
        "date": date,
        "vectors": {
            "H_v": {
                "file": "H_v.npy", "shape": list(h_v.shape), "dtype": "float32",
                "encoder": vision_id, "input_types": vision_input_types or [],
            },
            "H_t": {"file": "H_t.npy", "shape": list(h_t.shape), "dtype": "float32", "encoder": text_id},
            "H_r": {"file": "H_r.npy", "shape": list(h_r.shape), "dtype": "float32", "encoder": text_id, "top_k": top_k},
        },
        "retrieved_chunk_ids": chunk_ids,
        "source_json": source_json,
        "context_sources": context_sources or [],
    }
    schemas.validate_vector_index(index)
    write_json(index, out_dir / "index.json")


def run_fake(cfg, ticker: str, n: int) -> None:
    """第一階段：不需要 A 的資料、不需要 GPU，隨機向量測通儲存格式。"""
    from module_b_encoder.encoders.vision_encoder import fake_h_v
    from module_b_encoder.encoders.text_encoder import fake_h_t
    from module_b_encoder.rag.retriever import fake_h_r

    k = cfg["rag"]["top_k"]
    input_types = cfg["chart"].get("vision_inputs", ["candlestick", "volume"])
    h_v_shape = (len(input_types), 197, 768)
    d0 = dt.date(2021, 3, 1)
    # 絕不可寫入正式 ticker namespace，否則會覆蓋同日期的真實向量。
    fake_ticker = f"_{ticker}_FAKE"
    for i in range(n):
        date = (d0 + dt.timedelta(days=i)).isoformat()
        h_r, chunk_ids = fake_h_r(k, seed=i)
        save_vectors(fake_ticker, date, fake_h_v(shape=h_v_shape, seed=i), fake_h_t(seed=i), h_r, chunk_ids,
                     cfg["encoders"]["vision"], cfg["encoders"]["text"], k,
                     f"FAKE://{ticker}/{date}", input_types, ["synthetic"])
    print(f"[generate_vectors] FAKE {ticker}: {n} days -> {paths.VECTORS / fake_ticker}")


def run_real(cfg, ticker: str, limit: int | None, balance_sources: bool = False) -> None:
    """第二階段：讀 A 的每日 JSON，跑真模型。

    balance_sources=True 時，H_r 的 top-K 檢索改用按來源分配名額的模式（財報、逐字稿
    各至少保留 1 個名額，其餘名額才自由競爭），取代純相似度排序（`docs/decisions.md #70`：
    法說會逐字稿只佔向量庫 25% 的 chunk，卻拿下 75% 的 top-3 結果，懷疑是查詢向量有一半
    來自新聞、跟逐字稿文體較接近所致，純相似度排序會讓同一種文體壟斷 top-K）。
    預設關閉，行為與加入這個選項之前完全一樣，方便直接跟不開的版本做 A/B 對照。
    """
    dataset_dir = paths.DATASET / ticker
    files = sorted(dataset_dir.glob("*.json"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"找不到 A 的資料: {dataset_dir}，先跑 module_a_data.build_dataset")

    first_record = read_json(files[0])
    schemas.validate_daily_record(first_record)
    macro_mode = "macro" in first_record
    expected_inputs = (
        cfg.get("macro", {}).get("vision_inputs", ["candlestick", "volume"])
        if macro_mode else cfg["chart"].get("vision_inputs", ["candlestick", "volume"])
    )

    from module_b_encoder.encoders.vision_encoder import VisionEncoder
    from module_b_encoder.encoders.text_encoder import TextEncoder

    # Apple Silicon 上先載入 FAISS、再初始化 PyTorch/ViT，會在部分版本組合中觸發
    # 原生層 SIGSEGV。先完成兩個 PyTorch 模型的初始化，再匯入 FAISS。
    vision = VisionEncoder(cfg["encoders"]["vision"], n_images=len(expected_inputs))
    text = TextEncoder(cfg["encoders"]["text"])

    from module_b_encoder.rag.vector_db import ChunkVectorDB
    from module_b_encoder.rag.retriever import retrieve

    k = cfg["rag"]["top_k"]
    alpha = cfg["rag"].get("query_alpha", 0.5)
    db = ChunkVectorDB()
    quota = {"filing": 1, "transcript": 1} if balance_sources and not macro_mode else None

    for i, f in enumerate(files, 1):
        record = read_json(f)
        schemas.validate_daily_record(record)
        date = record["date"]
        if ("macro" in record) != macro_mode:
            raise ValueError(f"{dataset_dir} 混有 macro 與一般個股 record，請分開重建 dataset")

        # 當日文件 chunk 進向量庫（隨時間累積，天然避免 look-ahead），分開加入才有 source 標籤
        db.add_chunks(record["filing_chunks"], text, source="filing")
        db.add_chunks(record["transcript_chunks"], text, source="transcript")
        macro_chunks = _macro_chunks(record)
        db.add_chunks(macro_chunks["indicators"], text, source="macro_indicator")
        db.add_chunks(macro_chunks["news"], text, source="macro_news")

        chart_inputs = record["chart"].get("inputs")
        if not chart_inputs:
            raise ValueError(
                f"{f} 缺少 chart.inputs；雙圖 ViT 需要先重跑 chart_generator --volume "
                "與 build_dataset"
            )
        input_types = [item["type"] for item in chart_inputs]
        if input_types != expected_inputs:
            raise ValueError(
                f"{f} 的 chart.inputs 順序 {input_types} 與 config {expected_inputs} 不一致"
            )
        h_v = vision.encode([paths.ROOT / item["path"] for item in chart_inputs])
        text_items, context_sources = _text_context(record)
        h_t = text.encode_news(text_items)
        h_r, chunk_ids = retrieve(h_v, h_t, db, text, k, alpha, quota=quota)

        save_vectors(ticker, date, h_v, h_t, h_r, chunk_ids,
                     vision.model_id, text.model_id, k,
                     str(f.relative_to(paths.ROOT)).replace("\\", "/"), input_types,
                     context_sources)
        print(f"[generate_vectors] {ticker} {date} done（{i}/{len(files)}）", end="\r", flush=True)

    print()
    if limit is None:
        db.save(ticker)
    else:
        # 部分重建／smoke 不可覆蓋完整期間的 canonical FAISS index。
        print(f"[generate_vectors] --limit={limit}：不覆蓋 {ticker} 的完整 FAISS index")
    print(f"[generate_vectors] {ticker}: {len(files)} days -> {paths.VECTORS / ticker}")


def _text_context(record: dict) -> tuple[list[dict], list[str]]:
    """組合 H_t 文字；macro 指標置前，避免 512-token 截斷時完全看不到數值快照。"""
    ticker_news = list(record["news"])
    macro = record.get("macro")
    if not macro:
        return ticker_news, ["ticker_news"] if ticker_news else []

    items = []
    sources = []
    indicators = [item for item in macro["indicators"] if item.get("value") is not None]
    if indicators:
        content = "; ".join(
            f"{item['name']}={item['value']} {item['units']} "
            f"(observation {item['observation_date']}, available {item['available_date']})"
            for item in indicators
        )
        items.append({
            "headline": f"US macro indicators available as of {record['date']}",
            "content": content,
            "source": "fred_alfred_point_in_time",
            "published_at": f"{record['date']}T00:00:00-04:00",
            "days_ago": 0,
        })
        sources.append("macro_indicators")
    if macro["news"]:
        items.extend(macro["news"])
        sources.append("macro_news")
    if ticker_news:
        items.extend(ticker_news)
        sources.append("ticker_news")
    return items, sources


def _macro_chunks(record: dict) -> dict[str, list[dict]]:
    """將結構化指標與 FOMC 新聞轉成可追溯、可去重的 RAG chunks。"""
    macro = record.get("macro")
    if not macro:
        return {"indicators": [], "news": []}

    indicators = []
    for item in macro["indicators"]:
        if item.get("value") is None:
            continue
        chunk_id = (
            f"MACRO_{item['series_id']}_{item['observation_date']}_{item['available_date']}"
        ).replace("-", "")
        indicators.append({
            "chunk_id": chunk_id,
            "text": (
                f"{item['name']} was {item['value']} {item['units']} for "
                f"{item['observation_date']}, first available on {item['available_date']}."
            ),
        })

    news = []
    for item in macro["news"]:
        identity = item.get("url") or (
            f"{item['published_at']}|{item['headline']}|{item.get('category', '')}"
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
        news.append({
            "chunk_id": f"MACRO_NEWS_{item['published_at'][:10].replace('-', '')}_{digest}",
            "text": f"{item['headline']}. {item['content']}",
        })
    return {"indicators": indicators, "news": news}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--fake", action="store_true", help="第一階段：隨機向量測通流程")
    ap.add_argument("--n", type=int, default=10, help="--fake 時產幾天")
    ap.add_argument("--limit", type=int, default=None, help="真實模式只跑前 N 天")
    ap.add_argument("--balance_sources", action="store_true",
                    help="RAG top-K 按來源（財報/逐字稿）各保留至少 1 個名額，取代純相似度排序，"
                         "見 docs/decisions.md #70；預設關閉")
    args = ap.parse_args()

    started_at = time.perf_counter()
    try:
        if args.fake:
            run_fake(cfg, args.ticker, args.n)
        else:
            run_real(cfg, args.ticker, args.limit, balance_sources=args.balance_sources)
    finally:
        elapsed = time.perf_counter() - started_at
        print(f"[generate_vectors] 總執行時間：{elapsed:.1f} 秒（{elapsed / 60:.1f} 分鐘）")


if __name__ == "__main__":
    main()
