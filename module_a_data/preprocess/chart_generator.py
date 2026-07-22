"""用 mplfinance 畫每天往前 window_days（預設 20）個交易日的 K 線圖。

輸出：data/raw/charts/{TICKER}/{YYYY-MM-DD}.png
規格：PNG、224x224、RGB（configs/config.yaml 的 chart 區塊）。
K 線樣式（顏色、是否含成交量）全檔案統一，一旦定案不可中途變更。

用法：
    python -m module_a_data.preprocess.chart_generator --ticker AAPL
    python -m module_a_data.preprocess.chart_generator --ticker AAPL --limit 100
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # 無視窗環境
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image

from shared import paths
from shared.utils import load_config
from module_a_data.crawler.fetch_ohlcv import load_ohlcv

# 統一樣式：綠漲紅跌（美股慣例）、無成交量、無座標軸
_STYLE = mpf.make_mpf_style(
    marketcolors=mpf.make_marketcolors(up="green", down="red", edge="inherit", wick="inherit"),
    gridstyle="",
)


def generate_chart(df, ticker: str, date: str, window_days: int = 20,
                   size=(224, 224)) -> str | None:
    """畫 date 當天往前 window_days 個交易日的 K 線圖。

    回傳相對 repo 根目錄的路徑字串；資料不足（前面交易日不夠）回傳 None。
    """
    idx = df.index.get_indexer([date])[0]
    if idx < window_days - 1:
        return None
    window = df.iloc[idx - window_days + 1: idx + 1]

    out = paths.RAW_CHARTS / ticker / f"{date}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = mpf.plot(
        window, type="candle", style=_STYLE, volume=False,
        axisoff=True, returnfig=True, figsize=(3, 3), scale_padding=0,
    )
    fig.savefig(out, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 統一縮放到指定尺寸、RGB 三通道
    Image.open(out).convert("RGB").resize(size).save(out)
    return str(out.relative_to(paths.ROOT)).replace("\\", "/")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["tickers"][0])
    ap.add_argument("--limit", type=int, default=None, help="只產前 N 張（測試用）")
    args = ap.parse_args()

    window = cfg["chart"]["window_days"]
    size = tuple(cfg["chart"]["image_size"])
    df = load_ohlcv(args.ticker)

    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    if args.limit:
        dates = dates[: window - 1 + args.limit]  # 前 window-1 天畫不出來

    n = 0
    for date in dates:
        if generate_chart(df, args.ticker, date, window, size):
            n += 1
    print(f"[chart_generator] {args.ticker}: {n} charts -> {paths.RAW_CHARTS / args.ticker}")


if __name__ == "__main__":
    main()
