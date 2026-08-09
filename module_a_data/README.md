# Module A: Data Engineering

**Owner:** (fill in name)
**Deliverable:** `data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json`, one record per day. Format defined in `docs/data_format.md`, section 1.

## Responsibilities

1. Download historical OHLCV data via yfinance (`crawler/fetch_ohlcv.py`).
2. Render a 20-trading-day candlestick chart per day with mplfinance, saved as 224×224 RGB PNG (`preprocess/chart_generator.py`).
3. Fetch daily financial news and clean HTML/noise (`crawler/fetch_news.py` for recent news, `crawler/fetch_news_alpaca.py` for historical backfill via the Alpaca News API, `preprocess/text_cleaner.py` for cleanup).
4. Download SEC EDGAR filings and earnings-call transcripts, chunked to ≤512 tokens (`crawler/fetch_filings.py`, `crawler/fetch_transcripts.py`, `preprocess/chunker.py`).
5. Generate BULLISH / BEARISH / NEUTRAL labels from the 5-trading-day forward return vs. same-day close, ±2% thresholds (`labeling.py`).
6. Assemble everything into one JSON record per day (`build_dataset.py`).
7. **Deliver an initial 50–100 sample days early** so B and C can start development against real formats sooner.

## Suggested Order

```
fetch_ohlcv -> chart_generator -> labeling      # fastest path to a usable sample
fetch_news / fetch_news_alpaca -> text_cleaner  # news
fetch_filings / fetch_transcripts -> chunker    # documents
build_dataset last, to assemble all three lines
```

## Commands (run from the repo root)

```bash
python -m module_a_data.crawler.fetch_ohlcv                          # download OHLCV
python -m module_a_data.preprocess.chart_generator --ticker AAPL     # generate charts (--limit 100 for a quick test)
python -m module_a_data.crawler.fetch_news --ticker AAPL             # recent news
python -m module_a_data.crawler.fetch_news_alpaca --ticker AAPL --start 2021-01-01 --end 2026-08-08   # historical news backfill
python -m module_a_data.crawler.fetch_filings --ticker AAPL          # SEC filings
python -m module_a_data.build_dataset --ticker AAPL --limit 100      # assemble output (start with a 50-100 sample)
```

`build_dataset` runs fine before news/filings are fetched (empty arrays are valid under the schema), so the fastest path to a deliverable sample is: `fetch_ohlcv -> chart_generator -> build_dataset`.

## Notes

- Every record must pass `validate_daily_record()` in `shared/schemas.py` before being written.
- News entries must keep `published_at` (Eastern time) and `days_ago`.
- `future_closes` exists only to generate labels and for backtesting — it must never reach the model as an input feature. Keep it isolated from other fields.
- The ticker universe is currently fixed to AAPL. The code is written to support multiple tickers, but only one is run for now.
