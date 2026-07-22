"""用 yfinance 下載指定股票的歷史 OHLCV 數據，存到 data/raw/ohlcv/{TICKER}.csv。

欄位：Date,Open,High,Low,Close,Volume（Date 為索引，僅交易日）。
注意：yfinance 是否為最終數據源尚待與老師確認（docs/decisions.md #9）。

用法：
    python -m module_a_data.crawler.fetch_ohlcv            # 用 config.yaml 的設定
    python -m module_a_data.crawler.fetch_ohlcv --ticker AAPL --start 2021-01-01 --end 2025-12-31
"""
import argparse

import pandas as pd
import yfinance as yf

from shared import paths
from shared.utils import load_config


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """下載單一股票 OHLCV，回傳 DataFrame（index=Date）。"""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance 回傳空資料: {ticker} {start}~{end}")
    # yfinance 新版欄位是 MultiIndex，統一攤平
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    return df


def save_ohlcv(ticker: str, df: pd.DataFrame) -> None:
    out = paths.RAW_OHLCV / f"{ticker}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"[fetch_ohlcv] {ticker}: {len(df)} rows -> {out}")


def load_ohlcv(ticker: str) -> pd.DataFrame:
    """讀回已下載的 OHLCV（給 chart_generator / labeling / build_dataset 用）。"""
    return pd.read_csv(paths.RAW_OHLCV / f"{ticker}.csv", index_col="Date", parse_dates=True)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None, help="預設跑 config.yaml 的全部 tickers")
    ap.add_argument("--start", default=cfg["date_range"]["start"])
    ap.add_argument("--end", default=cfg["date_range"]["end"])
    args = ap.parse_args()

    tickers = [args.ticker] if args.ticker else cfg["tickers"]
    for t in tickers:
        save_ohlcv(t, fetch_ohlcv(t, args.start, args.end))


if __name__ == "__main__":
    main()
