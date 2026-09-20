"""用 yfinance 下載指定股票的歷史 OHLCV 數據，存到 data/raw/ohlcv/{TICKER}.csv。

欄位：Date,Open,High,Low,Close,Volume（Date 為索引，僅交易日）。預設用 yfinance
auto_adjust 調整拆股與股利，避免公司行動扭曲標籤與回測；可用 --unadjusted 做對照。
注意：yfinance 是否為最終數據源尚待與老師確認（docs/decisions.md #9）。

用法：
    python -m module_a_data.crawler.fetch_ohlcv            # 用 config.yaml 的設定
    python -m module_a_data.crawler.fetch_ohlcv --ticker AAPL --start 2021-01-01 --end 2025-12-31
"""
import argparse

import pandas as pd
import yfinance as yf

from shared import paths
from shared.utils import load_config, read_json, write_json


def fetch_ohlcv(ticker: str, start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
    """下載單一股票 OHLCV，回傳 DataFrame（index=Date）。"""
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance 回傳空資料: {ticker} {start}~{end}")
    # yfinance 新版欄位是 MultiIndex，統一攤平
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    return df


def save_ohlcv(ticker: str, df: pd.DataFrame, start: str, end: str,
               auto_adjust: bool) -> None:
    out = paths.RAW_OHLCV / f"{ticker}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    write_json({
        "ticker": ticker,
        "source": "yfinance",
        "start": start,
        "end_exclusive": end,
        "auto_adjust": auto_adjust,
        "n_rows": len(df),
    }, paths.RAW_OHLCV / f"{ticker}.meta.json")
    print(f"[fetch_ohlcv] {ticker}: {len(df)} rows -> {out}")


def load_ohlcv(ticker: str) -> pd.DataFrame:
    """讀回已下載的 OHLCV（給 chart_generator / labeling / build_dataset 用）。"""
    meta_path = paths.RAW_OHLCV / f"{ticker}.meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"缺少 {meta_path}；舊 OHLCV 無法確認 adjustment，請重跑 fetch_ohlcv")
    meta = read_json(meta_path)
    expected_adjustment = load_config().get("market_data", {}).get("auto_adjust", True)
    if meta.get("auto_adjust") != expected_adjustment:
        raise RuntimeError(
            f"{meta_path} 的 auto_adjust={meta.get('auto_adjust')} 與 config "
            f"{expected_adjustment} 不一致；請依目前 config 重跑 fetch_ohlcv"
        )
    return pd.read_csv(paths.RAW_OHLCV / f"{ticker}.csv", index_col="Date", parse_dates=True)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None, help="預設跑 config.yaml 的全部 tickers")
    ap.add_argument("--start", default=cfg["date_range"]["start"])
    ap.add_argument("--end", default=cfg["date_range"]["end"])
    ap.add_argument("--unadjusted", action="store_true",
                    help="對照用：停用拆股/股利調整；下游重跑前也須將 config 的 auto_adjust=false")
    args = ap.parse_args()

    tickers = [args.ticker] if args.ticker else cfg["tickers"]
    auto_adjust = cfg.get("market_data", {}).get("auto_adjust", True) and not args.unadjusted
    for t in tickers:
        save_ohlcv(t, fetch_ohlcv(t, args.start, args.end, auto_adjust),
                   args.start, args.end, auto_adjust)


if __name__ == "__main__":
    main()
