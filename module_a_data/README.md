# Module A: Data Engineering

**Owner:** (fill in name)
**Deliverable:** `data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json`, one record per day. Format defined in `docs/data_format.md`, section 1.

## Responsibilities

1. Download historical OHLCV data via yfinance (`crawler/fetch_ohlcv.py`). The formal pipeline uses
   split/dividend-adjusted OHLC so labels, rewards, and backtests approximate investor total return;
   `--unadjusted` is retained only for controlled comparison.
2. Render two 20-trading-day 224×224 RGB inputs per day: a candlestick chart and a separate
   volume chart (`preprocess/chart_generator.py`). They are kept as separate images for the shared
   dual-image ViT rather than composited into one bitmap.
3. Fetch daily financial news and clean HTML/noise (`crawler/fetch_news.py` for recent news, `crawler/fetch_news_alpaca.py` for historical backfill via the Alpaca News API, `preprocess/text_cleaner.py` for cleanup).
4. Download SEC EDGAR filings (10-K/10-Q as background, 8-K as timestamped supplementary events that don't overwrite the background — see `docs/decisions.md` #41/#44/#45) and earnings-call transcripts via the official Alpha Vantage API (`crawler/fetch_transcripts.py`, replacing the earlier third-party `foolcalls` scraper), chunked to ≤512 tokens (`crawler/fetch_filings.py`, `preprocess/chunker.py`). ETF/index macro data (`crawler/fetch_macro.py`) is scaffolded but not implemented (`docs/decisions.md` #38).
5. Generate BULLISH / BEARISH / NEUTRAL labels from the 5-trading-day forward return vs. same-day close. Quantile thresholds are fitted on the train split only and then frozen for validation/test. Samples whose five-day target crosses a split boundary are marked `purged`. The old fixed ±2% version is kept for ablation comparisons.
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
python -m module_a_data.preprocess.chart_generator --ticker AAPL     # generate configured ViT inputs (--limit 100 for a quick test)
python -m module_a_data.preprocess.chart_generator --ticker AAPL --technical --indicators rsi macd  # candlestick + multi-indicator chart
python -m module_a_data.preprocess.chart_generator --ticker AAPL --volume  # candlestick + pure-volume chart
python -m module_a_data.crawler.fetch_news --ticker AAPL             # recent news
python -m module_a_data.crawler.fetch_news_alpaca --ticker AAPL --start 2021-01-01 --end 2026-08-08   # historical news backfill
python -m module_a_data.crawler.fetch_filings --ticker AAPL          # SEC filings
python -m module_a_data.crawler.fetch_transcripts --ticker AAPL --start 2021-01-01 --end 2026-08-08   # earnings-call transcripts (Alpha Vantage, needs ALPHA_VANTAGE_API_KEY)
python -m module_a_data.build_dataset --ticker AAPL --limit 100      # assemble output (start with a 50-100 sample)
```

`build_dataset` runs fine before news/filings are fetched (empty arrays are valid under the schema),
but both configured vision images must exist. The fastest path to a deliverable sample is:
`fetch_ohlcv -> chart_generator -> build_dataset`.

## Notes

- Every record must pass `validate_daily_record()` in `shared/schemas.py` before being written.
- News entries must keep `published_at` (Eastern time) and `days_ago`.
- SEC filings use their acceptance time to determine availability. Sources without a reliable time are conservatively delayed to the next weekday.
- News published after 16:00 ET is assigned to the next effective trading day by both news crawlers.
- `build_dataset` writes a manifest with a stable record-content fingerprint; downstream stages reject stale vectors built from another dataset version.
- `future_closes` exists only to generate labels and for backtesting — it must never reach the model as an input feature. Keep it isolated from other fields.
- The ticker universe is currently fixed to AAPL. The code is written to support multiple tickers, but only one is run for now.
