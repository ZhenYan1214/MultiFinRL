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
  "split": "train",
  "chart": {
    "path": "data/raw/charts/AAPL/2021-03-15.png",
    "inputs": [
      { "type": "candlestick", "path": "data/raw/charts/AAPL/2021-03-15.png" },
      { "type": "volume", "path": "data/raw/charts/AAPL/volume/2021-03-15.png" }
    ],
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
    "future_closes": [124.1, 122.8, 121.9, 121.2, 120.53],
    "target_date": "2021-03-22"
  },
  "label": "BEARISH"
}
```

### 欄位規則

| 欄位 | 規則 |
|---|---|
| `date` | ISO 8601（YYYY-MM-DD），僅交易日 |
| `split` | `train` / `validation` / `test` / `purged`。全流程沿用此欄位；五日標籤跨越下一區段邊界的日期標為 `purged`，不進模型訓練或評估 |
| `chart.path` | 向下相容欄位，等於 `chart.inputs[0].path` |
| `chart.inputs` | ViT 輸入的固定順序清單，目前恰好兩張：K 線、成交量；每張皆為相對 repo 根目錄的 224×224 RGB PNG。未來可將第二張換成技術指標圖，但須同步更新 config 並重跑 H_v |
| `news[].days_ago` | 該則新聞發布日距 `date` 的天數；當日新聞為 0；當日無新聞時以近日新聞回補 |
| `news[].published_at` | 含時區（美東），供後續切齊時間、避免 look-ahead |
| `*_chunks[].text` | 每段 ≤512 token（以 FinBERT tokenizer 計） |
| `filing_chunks` / `transcript_chunks` | 「最新一份沿用到下一份發布為止」，且只納入該決策日已可取得的文件。SEC 以 acceptance time 判斷；時間未知的舊 SEC/法說資料保守延到下一平日 |
| `prices.future_closes` | 未來第 1~5 個交易日收盤價（標籤依據，**僅供產生標籤與回測，不可作為模型輸入**） |
| `prices.target_date` | `close_t5` 所在交易日，用來檢查標籤是否跨越 split 邊界 |
| `label` | 依 close_t5 vs close_t0 報酬；1/3、2/3 分位數門檻**只用 train split**估計，再固定套用至 validation/test，避免測試期分布進入標籤規則 |

`data/processed/dataset/{TICKER}_manifest.json` 保存本次有效日期、OHLCV 來源與 adjustment、
train-only 標籤門檻、split 筆數與日期範圍，以及 `protocol_version` / `records_sha256`。B 模組依 manifest 讀取，
避免舊版殘留 JSON 混入新實驗。

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
    "H_v": { "file": "H_v.npy", "shape": [2, 197, 768], "dtype": "float32", "encoder": "google/vit-base-patch16-224", "input_types": ["candlestick", "volume"] },
    "H_t": { "file": "H_t.npy", "shape": [512, 768], "dtype": "float32", "encoder": "ProsusAI/finbert" },
    "H_r": { "file": "H_r.npy", "shape": [3, 512, 768], "dtype": "float32", "encoder": "ProsusAI/finbert", "top_k": 3 }
  "retrieved_chunk_ids": [
    "AAPL_10-Q_2021-02-01_003",
    "AAPL_EC_2021Q1_012",
    "AAPL_10-K_2020-10-30_047"
  ],
  "source_json": "data/processed/dataset/AAPL/2021-03-15.json",
  "source_record_sha256": "...",
  "dataset_records_sha256": "..."
}
```

### 欄位規則

| 欄位 | 規則 |
|---|---|
| `shape` | 實際維度以最終選定的 encoder 為準，但**一旦定案不可再變**；換 encoder 需全員同意並重跑 |
| `encoder` | HuggingFace model id，供實驗比較與論文記錄 |
| `H_v` 第一維 | = 視覺輸入張數（目前 2），順序必須與 `chart.inputs`／`input_types` 一致；第二維為每張 ViT 的 CLS + patch tokens |
| `retrieved_chunk_ids` | 對應 A 資料中的 `chunk_id`，供事後追溯與可解釋性分析 |
| `H_r` 第一維 | = K（目前 K=3），順序為相似度由高至低 |
| `source_record_sha256` | 產生本日向量時所讀 daily JSON 的內容指紋；C 讀取時必須一致，否則拒絕混用舊向量 |
| `dataset_records_sha256` | 產生向量時所用整批 dataset 指紋；因 H_r 會累積先前文件，任一日資料改變即要求整批重建 |

---

## 3. C 的產出

**路徑規則**：`data/outputs/`

```
data/outputs/
├── z_fused/
│   ├── {TICKER}/{YYYY-MM-DD}.npy       # 每日 Z_fused 向量（除錯/可解釋性分析用）
│   ├── {TICKER}/run.json               # 本批 Z 的 dataset 指紋與日期清單
│   ├── {TICKER}_index.npz              # ★ 彙整索引：整段時間範圍一次讀取用
│   └── {TICKER}_index.meta.json        # 索引摘要（天數、日期範圍、z_dim）
├── checkpoints/                         # ticker 專屬 Fusion / decoder / PPO 權重
├── metrics/
│   ├── classification_report_{TICKER}.json       # 情緒分類準確率
│   ├── classification_report_{TICKER}_{run_id}.json # 每次執行的不可覆寫歷史報告
│   ├── event_validation_head_report_{TICKER}.json # Z_fused 事件 probe
│   └── event_extraction_report_{TICKER}.json     # 事件抽取 P/R/F1（B 提供）
└── backtest/
    └── report_{TICKER}_{strategy}_{run_id}.json  # 累積報酬、Sharpe、MDD
```

`{TICKER}_index.npz` 內容（`module_c_fusion/fusion/consolidate.py` 產出）：

| 陣列 | shape | 說明 |
|---|---|---|
| `dates` | [N] | YYYY-MM-DD，依日期排序 |
| `z` | [N, z_dim] | 每日 Z_fused，float32 |
| `label` | [N] | 0=BEARISH 1=NEUTRAL 2=BULLISH，對照 A 的 label |
| `return_next` | [N] | t → t+1 實際報酬，來自 A 的 `future_closes[0]` / `close_t0` |
| `split` | [N] | 每日固定的 `train` / `validation` / `test` / `purged` 標記 |
| `label_target_date` | [N] | 五日標籤的目標交易日，供切分稽核 |
| `dataset_records_sha256` | scalar | A dataset manifest 的內容指紋，防止舊 Z 與新 dataset 混用 |
| `z_fused_sha256` | scalar | 本次整批 Z_fused 的內容指紋，PPO/decoder checkpoint 會綁定此版本 |

`train.py` 只用 `train` 更新 fusion，凍結後對全部日期輸出 Z_fused，再自動產生此索引。
Classifier/decoder/PPO/backtest/IG 一律讀取索引中的 split，不對各自的資料交集重新切分。
真實流程不再 fallback 掃描零散舊檔；dataset、B 向量、Z index、PPO/decoder 任一層指紋
不一致時會直接停止並提示重跑上游。
LLM 產生的 `data/labels/y_belief/{TICKER}.json` 另有同名 `.meta.json` 綁定 dataset 指紋；
文字來源或 train-only TREND 標籤改變後，不會續用舊的生成目標。

---

## 4. 通用約定

1. **編碼**：所有 JSON 一律 UTF-8、無 BOM。
2. **路徑**：一律使用相對 repo 根目錄的正斜線路徑，程式中透過 `shared/paths.py` 取得，不硬編。
3. **時區**：所有時間戳記以美東時間（America/New_York）記錄並含時區偏移；「當日」的界定為美股交易日曆。
4. **驗證**：A 寫出 JSON 前、B 讀入前，都必須通過 `shared/schemas.py` 的驗證；B 寫出 index.json 前、C 讀入前同理。
5. **缺值**：允許空陣列，不允許缺欄位；數值缺失以 `null` 明示。
6. **報酬口徑**：正式 OHLCV 使用拆股／股利調整價格（`market_data.auto_adjust=true`）；下載設定記錄於 `data/raw/ohlcv/{TICKER}.meta.json`。
