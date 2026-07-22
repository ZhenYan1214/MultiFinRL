"""漲跌標籤產生。

定義（docs/decisions.md #16）：
    r = (未來第 5 個交易日收盤價 / 當日收盤價) - 1
    r > +2%  -> BULLISH
    r < -2%  -> BEARISH
    其餘     -> NEUTRAL
交易日以 OHLCV 檔案的實際列為準（已自然跳過休市日）。
"""
import pandas as pd


def make_label(close_t0: float, close_t5: float,
               bullish_threshold: float = 0.02, bearish_threshold: float = -0.02) -> str:
    r = close_t5 / close_t0 - 1
    if r > bullish_threshold:
        return "BULLISH"
    if r < bearish_threshold:
        return "BEARISH"
    return "NEUTRAL"


def build_price_and_label(df: pd.DataFrame, date: str, horizon: int = 5,
                          bullish_threshold: float = 0.02,
                          bearish_threshold: float = -0.02) -> dict | None:
    """回傳 {"prices": {...}, "label": ...}（data_format.md 第 1 節格式）。

    df: fetch_ohlcv.load_ohlcv() 的 DataFrame（index=Date）。
    未來交易日不足 horizon 天時回傳 None（資料尾端無法產標籤）。
    """
    idx = df.index.get_indexer([date])[0]
    if idx < 0 or idx + horizon >= len(df):
        return None
    closes = df["Close"]
    close_t0 = float(closes.iloc[idx])
    future = [round(float(closes.iloc[idx + i]), 4) for i in range(1, horizon + 1)]
    close_t5 = future[-1]
    return {
        "prices": {"close_t0": round(close_t0, 4), "close_t5": close_t5,
                   "future_closes": future},
        "label": make_label(close_t0, close_t5, bullish_threshold, bearish_threshold),
    }
