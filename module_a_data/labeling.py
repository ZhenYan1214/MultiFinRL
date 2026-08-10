"""漲跌標籤產生。

定義（2026-08 改為分位數門檻，取代 docs/decisions.md #16 的固定 ±2%，見 #30）：
    r = (未來第 5 個交易日收盤價 / 當日收盤價) - 1
    先算出整段資料裡「每一天的 r」這個分布，取其 1/3、2/3 分位數當門檻：
    r > 分位數門檻(高)  -> BULLISH
    r < 分位數門檻(低)  -> BEARISH
    其餘                -> NEUTRAL
    這樣三類天數會自然接近平衡（各約 1/3），門檻也會隨資料實際波動幅度調整，
    不像固定 2% 可能剛好切在報酬分布最密集的地方，製造大量邊界模糊天。
    門檻數值本身是用整段期間的 r 分布算出來的固定值，算完之後每一天都用同一組
    固定門檻判斷，不是每天重算——不是模型訓練當下才看到未來，是「定義規則」
    這一步驟用了整段期間的統計特性，跟原本用 2% 這個數字的性質相同（人為選定
    的固定切點，只是這次改用資料自己的分布來選，不是憑感覺選 2%）。
    舊版固定 ±2% 邏輯保留在 `make_label_fixed_threshold()`，供之後想做門檻定義的
    ablation 實驗時對照用。
交易日以 OHLCV 檔案的實際列為準（已自然跳過休市日）。
"""
import pandas as pd


def compute_return_series(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """算出每一天『未來 horizon 個交易日』的報酬率 r，跟 build_price_and_label 用同一套算法。"""
    closes = df["Close"]
    r = closes.shift(-horizon) / closes - 1
    return r.dropna()


def compute_quantile_thresholds(df: pd.DataFrame, horizon: int = 5,
                                low_q: float = 1 / 3, high_q: float = 2 / 3) -> tuple[float, float]:
    """回傳 (bearish_threshold, bullish_threshold)，依報酬分布的分位數決定。

    low_q/high_q 預設抓 1/3、2/3，讓 BEARISH/NEUTRAL/BULLISH 三類天數約各佔 1/3。
    """
    r = compute_return_series(df, horizon)
    bearish_threshold = float(r.quantile(low_q))
    bullish_threshold = float(r.quantile(high_q))
    return bearish_threshold, bullish_threshold


def make_label(close_t0: float, close_t5: float,
               bullish_threshold: float, bearish_threshold: float) -> str:
    """依給定門檻判斷單一天的標籤（門檻可以是分位數算出來的，也可以是固定值）。"""
    r = close_t5 / close_t0 - 1
    if r > bullish_threshold:
        return "BULLISH"
    if r < bearish_threshold:
        return "BEARISH"
    return "NEUTRAL"


def make_label_fixed_threshold(close_t0: float, close_t5: float,
                               bullish_threshold: float = 0.02,
                               bearish_threshold: float = -0.02) -> str:
    """舊版固定 ±2% 門檻（docs/decisions.md #16，已被 #30 取代為預設做法），保留供對照實驗用。"""
    return make_label(close_t0, close_t5, bullish_threshold, bearish_threshold)


def build_price_and_label(df: pd.DataFrame, date: str, horizon: int,
                          bullish_threshold: float, bearish_threshold: float) -> dict | None:
    """回傳 {"prices": {...}, "label": ...}（data_format.md 第 1 節格式）。

    df: fetch_ohlcv.load_ohlcv() 的 DataFrame（index=Date）。
    bullish_threshold/bearish_threshold：呼叫端先用 compute_quantile_thresholds()
    對整段 df 算好、傳進來的固定值（見 build_dataset.py），每一天共用同一組門檻。
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
