# Module A：資料工程

**負責人：**（填入姓名）
**交付物：** `data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json`，一天一筆，格式見 `docs/data_format.md` 第 1 節。

## 職責

1. 用 yfinance 下載指定股票的歷史 OHLCV 數據（`crawler/fetch_ohlcv.py`）
2. 用 mplfinance 畫每天往前 20 個交易日的 K 線圖，存成 PNG、224×224、RGB（`preprocess/chart_generator.py`）
3. 爬取每日相關財經新聞，清洗 HTML 與雜訊（`crawler/fetch_news.py` + `preprocess/text_cleaner.py`）
4. 下載 SEC EDGAR 財報與法說會逐字稿，切成 ≤512 token 的 Chunk（`crawler/fetch_filings.py`、`crawler/fetch_transcripts.py` + `preprocess/chunker.py`）
5. 依「未來第 5 個交易日收盤 vs 當日收盤 ±2%」產生 BULLISH / BEARISH / NEUTRAL 標籤（`labeling.py`）
6. 彙整為統一 JSON，一天一筆（`build_dataset.py`）
7. **優先交付前 50–100 筆樣本**給 B 和 C 提早開發

## 執行順序建議

```
fetch_ohlcv → chart_generator → labeling      # 這條線最快能產出樣本
fetch_news → text_cleaner                      # 新聞線
fetch_filings / fetch_transcripts → chunker    # 文件線
最後 build_dataset 彙整三條線
```

## 指令（在 repo 根目錄執行）

```bash
python -m module_a_data.crawler.fetch_ohlcv                        # 下載 OHLCV
python -m module_a_data.preprocess.chart_generator --ticker AAPL   # 產 K 線圖（--limit 100 先試跑）
python -m module_a_data.crawler.fetch_news --ticker AAPL           # 近期新聞（歷史新聞來源待補）
python -m module_a_data.crawler.fetch_filings --ticker AAPL        # SEC 財報
python -m module_a_data.build_dataset --ticker AAPL --limit 100    # 彙整輸出（先交 50-100 筆樣本）
```

新聞和財報還沒抓時也能跑 build_dataset（news / chunks 為空陣列，格式允許），
所以最快的樣本交付路徑是：fetch_ohlcv → chart_generator → build_dataset。

## 注意事項

- 寫出每筆 JSON 前必須通過 `shared/schemas.py` 的 `validate_daily_record()`。
- 新聞需保留 `published_at`（含美東時區）與 `days_ago`。
- `future_closes` 只用來產標籤與回測，下游模型不會拿到——不要混進其他欄位。
- 股票池目前固定 AAPL；程式設計時以「多 ticker」為前提，但先只跑一檔。
