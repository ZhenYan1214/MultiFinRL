"""產生供 ViT 使用的價格圖、技術指標圖與純成交量圖。

輸入：data/raw/ohlcv/{TICKER}.csv

輸出：
    K 線圖（維持既有路徑，供 build_dataset.py 使用）
        data/raw/charts/{TICKER}/{YYYY-MM-DD}.png
    技術指標圖（一張圖可包含多個上下排列的 panel）
        data/raw/charts/{TICKER}/technical/{INDICATOR_SET}/{YYYY-MM-DD}.png
    純成交量圖
        data/raw/charts/{TICKER}/volume/{YYYY-MM-DD}.png

三種圖皆為 224x224 RGB PNG，使用相同交易日窗口。預設產生 config 中指定的雙圖輸入
（目前為 K 線 + 成交量）；CLI 可用 --no-volume 等旗標覆寫。技術指標全部只使用當日及
之前的資料計算，不會使用未來資料。

用法：
    # 依 config 產生正式雙圖輸入（目前為 K 線 + 成交量）
    python -m module_a_data.preprocess.chart_generator --ticker AAPL --limit 100

    # K 線 + 一張同時包含 RSI、MACD 的技術指標圖
    python -m module_a_data.preprocess.chart_generator --ticker AAPL \
        --technical --indicators rsi macd

    # 只畫技術指標圖
    python -m module_a_data.preprocess.chart_generator --ticker AAPL \
        --no-candlestick --technical --indicators sma ema rsi macd bollinger

    # K 線 + 純成交量圖；若只要成交量，再加 --no-candlestick
    python -m module_a_data.preprocess.chart_generator --ticker AAPL --volume
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 無視窗環境
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from PIL import Image

from shared import paths
from shared.utils import load_config
from module_a_data.crawler.fetch_ohlcv import load_ohlcv


SUPPORTED_INDICATORS = ("sma", "ema", "rsi", "macd", "bollinger")
INDICATOR_COLUMNS = {
    "sma": ["sma_5", "sma_10", "sma_20"],
    "ema": ["ema_12", "ema_26"],
    "rsi": ["rsi_14"],
    "macd": ["macd", "macd_signal", "macd_hist"],
    "bollinger": ["bb_mid", "bb_upper", "bb_lower"],
}

# 統一樣式：綠漲紅跌、無成交量、無座標軸。
_STYLE = mpf.make_mpf_style(
    marketcolors=mpf.make_marketcolors(up="green", down="red", edge="inherit", wick="inherit"),
    gridstyle="",
)


def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV 缺少欄位: {sorted(missing)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV 日期索引必須由舊到新排序")


def _window_for_date(df: pd.DataFrame, date: str, window_days: int) -> pd.DataFrame | None:
    """回傳包含 date 當天的 trailing window；資料不足時回傳 None。"""
    idx = int(df.index.get_indexer([pd.Timestamp(date)])[0])
    if idx < 0:
        raise ValueError(f"OHLCV 找不到日期: {date}")
    if idx < window_days - 1:
        return None
    return df.iloc[idx - window_days + 1: idx + 1]


def _save_figure(fig, out: Path, size: tuple[int, int]) -> str:
    """將 matplotlib figure 正規化成固定尺寸 RGB PNG，回傳 repo-relative 路徑。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)

    # mplfinance/matplotlib 搭配 tight bbox 時像素可能不是正方形；最後統一輸出契約。
    with Image.open(out) as image:
        normalized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        normalized.save(out)
    return str(out.relative_to(paths.ROOT)).replace("\\", "/")


def generate_candlestick_chart(
    df: pd.DataFrame, ticker: str, date: str, window_days: int = 20, size: tuple[int, int] = (224, 224)
) -> str | None:
    """畫 date 當天往前 window_days 個交易日的 K 線圖。"""
    _validate_ohlcv(df)
    window = _window_for_date(df, date, window_days)
    if window is None:
        return None

    out = paths.RAW_CHARTS / ticker / f"{date}.png"
    fig, _ = mpf.plot(
        window, type="candle", style=_STYLE, volume=False, axisoff=True,
        returnfig=True, figsize=(3, 3), scale_padding=0,
    )
    return _save_figure(fig, out, size)


def _column_with_prefix(frame: pd.DataFrame, prefix: str) -> pd.Series:
    """安全取得 pandas-ta-classic 的具名輸出，避免依賴欄位順序。"""
    matches = [column for column in frame.columns if column.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"pandas-ta-classic 輸出異常：prefix={prefix!r}, columns={list(frame.columns)}")
    return frame[matches[0]]


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """用 pandas-ta-classic 一次計算全部支援的指標。"""
    try:
        import pandas_ta_classic as ta
    except ImportError as exc:  # K 線／成交量模式不需要載入此額外依賴
        raise RuntimeError(
            "產生技術指標圖需要 pandas-ta-classic；請先執行 pip install -r requirements.txt"
        ) from exc

    close = df["Close"].astype(float)
    result = pd.DataFrame(index=df.index)
    result["close"] = close

    result["sma_5"] = ta.sma(close, length=5)
    result["sma_10"] = ta.sma(close, length=10)
    result["sma_20"] = ta.sma(close, length=20)
    result["ema_12"] = ta.ema(close, length=12)
    result["ema_26"] = ta.ema(close, length=26)
    result["rsi_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is None:
        raise RuntimeError("pandas-ta-classic 無法計算 MACD；請確認 OHLCV 資料長度")
    result["macd"] = _column_with_prefix(macd, "MACD_")
    result["macd_signal"] = _column_with_prefix(macd, "MACDs_")
    result["macd_hist"] = _column_with_prefix(macd, "MACDh_")

    bbands = ta.bbands(close, length=20, std=2.0)
    if bbands is None:
        raise RuntimeError("pandas-ta-classic 無法計算 Bollinger Bands；請確認 OHLCV 資料長度")
    result["bb_lower"] = _column_with_prefix(bbands, "BBL_")
    result["bb_mid"] = _column_with_prefix(bbands, "BBM_")
    result["bb_upper"] = _column_with_prefix(bbands, "BBU_")
    return result


def _indicator_window_complete(
    values: pd.DataFrame, date: str, indicators: list[str] | tuple[str, ...], window_days: int
) -> bool:
    """由套件實際輸出的 NaN 判斷該日是否已有完整指標窗口。"""
    idx = int(values.index.get_indexer([pd.Timestamp(date)])[0])
    if idx < window_days - 1:
        return False
    window = values.iloc[idx - window_days + 1: idx + 1]
    return all(window[INDICATOR_COLUMNS[name]].notna().all().all() for name in indicators)


def _plot_indicator(ax, name: str, values: pd.DataFrame) -> None:
    """在單一 panel 畫一種指標。圖中不放文字，避免 ViT 學到標籤捷徑。"""
    x = np.arange(len(values))
    close = values["close"].to_numpy(dtype=float)

    if name == "sma":
        ax.plot(x, close, color="#555555", linewidth=0.8)
        ax.plot(x, values["sma_5"], color="#f39c12", linewidth=1.0)
        ax.plot(x, values["sma_10"], color="#2980b9", linewidth=1.0)
        ax.plot(x, values["sma_20"], color="#8e44ad", linewidth=1.0)
    elif name == "ema":
        ax.plot(x, close, color="#555555", linewidth=0.8)
        ax.plot(x, values["ema_12"], color="#16a085", linewidth=1.0)
        ax.plot(x, values["ema_26"], color="#c0392b", linewidth=1.0)
    elif name == "rsi":
        ax.plot(x, values["rsi_14"], color="#6c3483", linewidth=1.2)
        ax.axhline(70, color="#c0392b", linewidth=0.6, linestyle="--")
        ax.axhline(30, color="#27ae60", linewidth=0.6, linestyle="--")
        ax.set_ylim(0, 100)
    elif name == "macd":
        hist = values["macd_hist"].to_numpy(dtype=float)
        colors = np.where(np.nan_to_num(hist) >= 0, "#27ae60", "#c0392b")
        ax.bar(x, hist, color=colors, width=0.75, alpha=0.65)
        ax.plot(x, values["macd"], color="#2980b9", linewidth=1.0)
        ax.plot(x, values["macd_signal"], color="#f39c12", linewidth=1.0)
        ax.axhline(0, color="#777777", linewidth=0.5)
    elif name == "bollinger":
        upper = values["bb_upper"].to_numpy(dtype=float)
        lower = values["bb_lower"].to_numpy(dtype=float)
        ax.plot(x, close, color="#333333", linewidth=0.9)
        ax.plot(x, values["bb_mid"], color="#2980b9", linewidth=0.9)
        ax.plot(x, upper, color="#7f8c8d", linewidth=0.7)
        ax.plot(x, lower, color="#7f8c8d", linewidth=0.7)
        ax.fill_between(x, lower, upper, color="#bdc3c7", alpha=0.25)
    else:  # pragma: no cover - CLI 與 generate_technical_chart 會先驗證
        raise ValueError(f"不支援的技術指標: {name}")

    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.axis("off")


def generate_technical_chart(
    df: pd.DataFrame, ticker: str, date: str, indicators: list[str] | tuple[str, ...],
    window_days: int = 20, size: tuple[int, int] = (224, 224),
    computed: pd.DataFrame | None = None,
) -> str | None:
    """將多個技術指標畫成一張上下排列的 RGB 圖。"""
    _validate_ohlcv(df)
    normalized = tuple(dict.fromkeys(name.lower() for name in indicators))
    if not normalized:
        raise ValueError("technical chart 至少要指定一個 indicator")
    unknown = set(normalized) - set(SUPPORTED_INDICATORS)
    if unknown:
        raise ValueError(f"不支援的技術指標: {sorted(unknown)}")

    window = _window_for_date(df, date, window_days)
    if window is None:
        return None
    indicator_values = computed if computed is not None else _compute_indicators(df)
    values = indicator_values.loc[window.index]

    # 每個 panel 的整個顯示窗口都必須有完整數值，避免早期日期產出半張空白圖。
    if any(not values[INDICATOR_COLUMNS[name]].notna().all().all() for name in normalized):
        return None

    n_panels = len(normalized)
    fig, axes = plt.subplots(n_panels, 1, figsize=(3, 3), squeeze=False, gridspec_kw={"hspace": 0.04})
    for ax, name in zip(axes[:, 0], normalized):
        _plot_indicator(ax, name, values)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.04)

    indicator_set = "_".join(normalized)
    out = paths.RAW_CHARTS / ticker / "technical" / indicator_set / f"{date}.png"
    return _save_figure(fig, out, size)


def generate_volume_chart(
    df: pd.DataFrame, ticker: str, date: str, window_days: int = 20, size: tuple[int, int] = (224, 224)
) -> str | None:
    """畫純成交量圖；綠色為收盤不低於開盤，紅色為收盤低於開盤。"""
    _validate_ohlcv(df)
    window = _window_for_date(df, date, window_days)
    if window is None:
        return None

    colors = np.where(window["Close"].to_numpy() >= window["Open"].to_numpy(), "green", "red")
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(np.arange(len(window)), window["Volume"].to_numpy(dtype=float), color=colors, width=0.75)
    ax.set_xlim(-0.5, len(window) - 0.5)
    ax.set_ylim(bottom=0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out = paths.RAW_CHARTS / ticker / "volume" / f"{date}.png"
    return _save_figure(fig, out, size)


def generate_chart(
    df: pd.DataFrame, ticker: str, date: str, window_days: int = 20, size: tuple[int, int] = (224, 224)
) -> str | None:
    """向下相容的舊函式名稱；等同 generate_candlestick_chart。"""
    return generate_candlestick_chart(df, ticker, date, window_days, size)


def main() -> None:
    cfg = load_config()
    configured_inputs = cfg["chart"].get("vision_inputs", ["candlestick", "volume"])
    technical_inputs = [name for name in configured_inputs if name.startswith("technical/")]
    if len(technical_inputs) > 1:
        raise ValueError(f"chart.vision_inputs 最多只能有一張 technical 圖: {technical_inputs}")
    configured_indicators = (
        technical_inputs[0].removeprefix("technical/").split("_")
        if technical_inputs else ["rsi", "macd"]
    )
    unknown = set(configured_indicators) - set(SUPPORTED_INDICATORS)
    if unknown:
        raise ValueError(f"chart.vision_inputs 包含不支援的技術指標: {sorted(unknown)}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--limit", type=int, default=None, help="只產前 N 個可用交易日（測試用）")
    ap.add_argument(
        "--candlestick", action=argparse.BooleanOptionalAction, default=True,
        help="是否產生 K 線圖（預設開啟；用 --no-candlestick 關閉）",
    )
    technical_default = bool(technical_inputs)
    ap.add_argument("--technical", action=argparse.BooleanOptionalAction, default=technical_default,
                    help=f"是否產生技術指標圖（目前預設 {'開啟' if technical_default else '關閉'}）")
    volume_default = "volume" in configured_inputs
    ap.add_argument("--volume", action=argparse.BooleanOptionalAction, default=volume_default,
                    help=f"是否產生純成交量圖（目前預設 {'開啟' if volume_default else '關閉'}）")
    ap.add_argument(
        "--indicators", nargs="+", choices=SUPPORTED_INDICATORS, default=configured_indicators,
        help=f"技術圖內要包含的指標，可複選（目前預設：{' '.join(configured_indicators)}）",
    )
    args = ap.parse_args()

    if not (args.candlestick or args.technical or args.volume):
        ap.error("至少要開啟 candlestick、technical、volume 其中一種圖")

    window_days = cfg["chart"]["window_days"]
    size = tuple(cfg["chart"]["image_size"])
    df = load_ohlcv(args.ticker)
    _validate_ohlcv(df)

    computed = _compute_indicators(df) if args.technical else None
    dates = [d.strftime("%Y-%m-%d") for d in df.index[window_days - 1:]]
    if computed is not None:
        # 不自己猜 warm-up 天數，直接以 pandas-ta-classic 的有效輸出為準。
        dates = [
            date for date in dates if _indicator_window_complete(computed, date, args.indicators, window_days)
        ]
    if args.limit is not None:
        if args.limit <= 0:
            ap.error("--limit 必須是正整數")
        dates = dates[:args.limit]

    counts = {"candlestick": 0, "technical": 0, "volume": 0}
    last_paths: dict[str, str] = {}

    for date in dates:
        if args.candlestick:
            out = generate_candlestick_chart(df, args.ticker, date, window_days, size)
            if out:
                counts["candlestick"] += 1
                last_paths["candlestick"] = out
        if args.technical:
            out = generate_technical_chart(
                df, args.ticker, date, args.indicators, window_days, size, computed=computed
            )
            if out:
                counts["technical"] += 1
                last_paths["technical"] = out
        if args.volume:
            out = generate_volume_chart(df, args.ticker, date, window_days, size)
            if out:
                counts["volume"] += 1
                last_paths["volume"] = out

    chart_modes = (
        ("candlestick", args.candlestick),
        ("technical", args.technical),
        ("volume", args.volume),
    )
    enabled = [name for name, on in chart_modes if on]
    for name in enabled:
        print(f"[chart_generator] {args.ticker} {name}: {counts[name]} charts "
              f"(last={last_paths.get(name, 'none')})")


if __name__ == "__main__":
    main()
