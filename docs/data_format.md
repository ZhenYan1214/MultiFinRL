# 資料格式契約（Data Format Contract）

本文件是 A、B、C 三個模組之間銜接的**唯一標準**。任何欄位的新增、刪除、改名都必須先改這份文件與 `shared/schemas.py`，並通知全員。

> 負責人提供的雲端範例檔（DataStruct.json / vectors.json）為原始依據，本文件是其正式化版本；若兩者衝突，開會確認後以更新過的本文件為準。

---

## 1. A 的交付物：每日資料 JSON

**路徑規則**：`data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json`
**單位**：一個「股票 + 交易日」一個檔案。

```json
{
  "ticker": "AAPL",
  "date": "2021-03-15",
  "chart": {
    "path": "data/raw/charts/AAPL/2021-03-15.png",
    "window_days": 20,
    "size": [224, 224],
    "channels": 3
  },
  "news": [
    {
      "headline": "Apple unveils new product line",
      "content": "（已清洗、無 HTML 的全文）",
      "source": "yahoo_finance",
      "published_at": "2021-03-15T09:30:00-04:00",
      "days_ago": 0
    },
    {
      "headline": "Supply chain update",
      "content": "...",
      "source": "reuters",
      "published_at": "2021-03-13T14:00:00-04:00",
      "days_ago": 2
    }
  ],
  "filing_chunks": [
    {
      "chunk_id": "AAPL_10-Q_2021-02-01_003",
      "doc_type": "10-Q",
      "filing_date": "2021-02-01",
      "text": "（≤512 token 的段落）"
    }
  ],
  "transcript_chunks": [
    {
      "chunk_id": "AAPL_EC_2021Q1_012",
      "doc_type": "earnings_call",
      "event_date": "2021-01-27",
      "text": "（≤512 token 的段落）"
    }
  ],
  "prices": {
    "close_t0": 123.99,
    "close_t5": 120.53,
    "future_closes": [124.1, 122.8, 121.9, 121.2, 120.53]
  },
  "label": "BEARISH"
}
```

### 欄位規則

| 欄位 | 規則 |
|---|---|
| `date` | ISO 8601（YYYY-MM-DD），僅交易日 |
| `chart.path` | 相對 repo 根目錄的路徑；PNG、224×224、RGB |
| `news[].days_ago` | 該則新聞發布日距 `date` 的天數；當日新聞為 0；當日無新聞時以近日新聞回補 |
| `news[].published_at` | 含時區（美東），供後續切齊時間、避免 look-ahead |
| `*_chunks[].text` | 每段 ≤512 token（以 FinBERT tokenizer 計） |
| `filing_chunks` / `transcript_chunks` | 「最新一份沿用到下一份發布為止」；當日無有效文件時為空陣列 `[]` |
| `prices.future_closes` | 未來第 1~5 個交易日收盤價（標籤依據，**僅供產生標籤與回測，不可作為模型輸入**） |
| `label` | 依 close_t5 vs close_t0 報酬：> +2% → `BULLISH`；< −2% → `BEARISH`；其餘 → `NEUTRAL` |

---

## 2. B 的交付物：每日向量

**路徑規則**：`data/vectors/{TICKER}/{YYYY-MM-DD}/`
向量本體用 `.npy` 儲存（float32），另附一份 `index.json` 描述中繼資料。JSON 內不直接放大型數值陣列。

```
data/vectors/AAPL/2021-03-15/
├── H_v.npy        # 視覺向量
├── H_t.npy        # 文字向量
├── H_r.npy        # 檢索向量
└── index.json
```

`index.json`：

```json
{
  "ticker": "AAPL",
  "date": "2021-03-15",
  "vectors": {
    "H_v": { "file": "H_v.npy", "shape": [197, 768], "dtype": "float32", "encoder": "google/vit-base-patch16-224" },
    "H_t": { "file": "H_t.npy", "shape": [512, 768], "dtype": "float32", "encoder": "ProsusAI/finbert" },
    "H_r": { "file": "H_r.npy", "shape": [3, 512, 768], "dtype": "float32", "encoder": "ProsusAI/finbert", "top_k": 3 }
  },
  "retrieved_chunk_ids": [
    "AAPL_10-Q_2021-02-01_003",
    "AAPL_EC_2021Q1_012",
    "AAPL_10-K_2020-10-30_047"
  ],
  "source_json": "data/processed/dataset/AAPL/2021-03-15.json"
}
```

### 欄位規則

| 欄位 | 規則 |
|---|---|
| `shape` | 實際維度以最終選定的 encoder 為準，但**一旦定案不可再變**；換 encoder 需全員同意並重跑 |
| `encoder` | HuggingFace model id，供實驗比較與論文記錄 |
| `retrieved_chunk_ids` | 對應 A 資料中的 `chunk_id`，供事後追溯與可解釋性分析 |
| `H_r` 第一維 | = K（目前 K=3），順序為相似度由高至低 |

---

## 3. C 的產出

**路徑規則**：`data/outputs/`

```
data/outputs/
├── z_fused/{TICKER}/{YYYY-MM-DD}.npy   # 每日 Z_fused 向量
├── checkpoints/                         # Fusion 模型與 RL agent 權重
├── metrics/
│   ├── classification_report.json      # 情緒分類準確率（對照 A 的 label）
│   └── event_extraction_report.json    # 事件抽取 P/R/F1（B 提供）
└── backtest/
    └── report_{run_id}.json            # 累積報酬、Sharpe、MDD、有無 RL 對照
```

---

## 4. 通用約定

1. **編碼**：所有 JSON 一律 UTF-8、無 BOM。
2. **路徑**：一律使用相對 repo 根目錄的正斜線路徑，程式中透過 `shared/paths.py` 取得，不硬編。
3. **時區**：所有時間戳記以美東時間（America/New_York）記錄並含時區偏移；「當日」的界定為美股交易日曆。
4. **驗證**：A 寫出 JSON 前、B 讀入前，都必須通過 `shared/schemas.py` 的驗證；B 寫出 index.json 前、C 讀入前同理。
5. **缺值**：允許空陣列，不允許缺欄位；數值缺失以 `null` 明示。
