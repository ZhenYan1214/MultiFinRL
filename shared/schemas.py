"""資料格式驗證。

契約定義見 docs/data_format.md。
A 寫出 JSON 前、B 讀入前，都必須通過 validate_daily_record()。
B 寫出 index.json 前、C 讀入前，都必須通過 validate_vector_index()。

任何欄位變更：先改 docs/data_format.md -> 改本檔 -> 通知全員 -> 才改各自程式。
"""

VALID_LABELS = {"BULLISH", "BEARISH", "NEUTRAL"}

DAILY_RECORD_REQUIRED_KEYS = {
    "ticker", "date", "chart", "news",
    "filing_chunks", "transcript_chunks", "prices", "label",
}

VECTOR_INDEX_REQUIRED_KEYS = {
    "ticker", "date", "vectors", "retrieved_chunk_ids", "source_json",
}


def validate_daily_record(record: dict) -> None:
    """驗證 A 的每日 JSON。不合法時 raise ValueError。"""
    missing = DAILY_RECORD_REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"daily record missing keys: {missing}")
    if record["label"] not in VALID_LABELS:
        raise ValueError(f"invalid label: {record['label']}")
    # chart
    chart = record["chart"]
    for k in ("path", "window_days", "size", "channels"):
        if k not in chart:
            raise ValueError(f"chart missing key: {k}")
    if list(chart["size"]) != [224, 224] or chart["channels"] != 3:
        raise ValueError(f"chart must be 224x224x3, got {chart['size']}x{chart['channels']}")
    inputs = chart.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("chart.inputs must contain exactly two vision inputs")
    for i, item in enumerate(inputs):
        for k in ("type", "path"):
            if not item.get(k):
                raise ValueError(f"chart.inputs[{i}] missing key: {k}")
    # news
    for i, n in enumerate(record["news"]):
        for k in ("headline", "content", "source", "published_at", "days_ago"):
            if k not in n:
                raise ValueError(f"news[{i}] missing key: {k}")
        if not isinstance(n["days_ago"], int) or n["days_ago"] < 0:
            raise ValueError(f"news[{i}].days_ago must be int >= 0")
    # chunks
    for field, keys in (
        ("filing_chunks", ("chunk_id", "doc_type", "filing_date", "text")),
        ("transcript_chunks", ("chunk_id", "doc_type", "event_date", "text")),
    ):
        for i, c in enumerate(record[field]):
            for k in keys:
                if k not in c:
                    raise ValueError(f"{field}[{i}] missing key: {k}")
    # ETF／指數才會有 macro；個股舊資料沒有此欄位仍保持相容。
    macro = record.get("macro")
    if macro is not None:
        for key in ("as_of_date", "indicators", "news", "dot_plot"):
            if key not in macro:
                raise ValueError(f"macro missing key: {key}")
        if macro["as_of_date"] != record["date"]:
            raise ValueError("macro.as_of_date must equal record.date")
        for i, item in enumerate(macro["indicators"]):
            for key in (
                "series_id", "name", "frequency", "units", "observation_date",
                "available_date", "value", "source",
            ):
                if key not in item:
                    raise ValueError(f"macro.indicators[{i}] missing key: {key}")
            if item["available_date"] > record["date"]:
                raise ValueError(f"macro.indicators[{i}] uses future available_date")
        for i, item in enumerate(macro["news"]):
            for key in ("headline", "content", "source", "published_at", "days_ago", "category"):
                if key not in item:
                    raise ValueError(f"macro.news[{i}] missing key: {key}")
            if not isinstance(item["days_ago"], int) or item["days_ago"] < 0:
                raise ValueError(f"macro.news[{i}].days_ago must be int >= 0")
            if item["published_at"][:10] > record["date"]:
                raise ValueError(f"macro.news[{i}] is from the future")
        dot_plot = macro["dot_plot"]
        for key in ("available_date", "meeting_date", "path", "source_json", "source"):
            if not dot_plot.get(key):
                raise ValueError(f"macro.dot_plot missing key: {key}")
        if dot_plot["available_date"] > record["date"]:
            raise ValueError("macro.dot_plot uses future available_date")
        input_types = [item["type"] for item in inputs]
        valid_macro_inputs = (
            ["candlestick", "volume"],
            ["candlestick", "macro_dot_plot"],
        )
        if input_types not in valid_macro_inputs:
            raise ValueError(
                "macro record chart.inputs must be ['candlestick', 'volume'] or "
                "['candlestick', 'macro_dot_plot']"
            )
        if input_types[1] == "macro_dot_plot" and inputs[1]["path"] != dot_plot["path"]:
            raise ValueError("macro.dot_plot.path must match chart.inputs[1].path")
    # prices
    prices = record["prices"]
    for k in ("close_t0", "close_t5", "future_closes"):
        if k not in prices:
            raise ValueError(f"prices missing key: {k}")
    if len(prices["future_closes"]) != 5:
        raise ValueError("prices.future_closes must have exactly 5 values")


def validate_vector_index(index: dict) -> None:
    """驗證 B 的 index.json。不合法時 raise ValueError。"""
    missing = VECTOR_INDEX_REQUIRED_KEYS - index.keys()
    if missing:
        raise ValueError(f"vector index missing keys: {missing}")
    for name in ("H_v", "H_t", "H_r"):
        if name not in index["vectors"]:
            raise ValueError(f"vectors missing: {name}")
        for k in ("file", "shape", "dtype", "encoder"):
            if k not in index["vectors"][name]:
                raise ValueError(f"vectors.{name} missing key: {k}")

    h_v = index["vectors"]["H_v"]
    if h_v["shape"] != [2, 197, 768]:
        raise ValueError(f"vectors.H_v shape must be [2, 197, 768], got {h_v['shape']}")
    if h_v.get("input_types") is None or len(h_v["input_types"]) != 2:
        raise ValueError("vectors.H_v.input_types must contain exactly two ordered input types")
