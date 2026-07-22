# MultiFinRL

多模態檢索增強金融決策框架（Multimodal Retrieval-Augmented Financial Decision Framework with Reinforcement Learning）

本專案將每支股票每個交易日的 **K 線圖（視覺）**、**財經新聞（文字）**、**財報／法說會文件（外部知識）** 三種資料，分別經過視覺編碼器（ViT）、文字編碼器（FinBERT / LLaMA）與 RAG 檢索模組處理後，用 Cross-Modal Transformer 融合成一個代表當日市場狀態的向量 **Z_fused**，並以此訓練強化學習（PPO）投資組合配置 agent。

---

## 兩年期目標

| 年度 | 目標 |
|---|---|
| 第一年 | 建立完整資料與模型 pipeline，產出 Z_fused，以市場情緒分類與事件抽取驗證其品質，並完成簡單的投資組合回測 |
| 第二年 | 將 Z_fused 接上 RL（PPO），進行完整投資組合配置與回測，評估 Sharpe Ratio、最大回撤等績效指標 |

## Pipeline 總覽

```
輸入：股票代號 + 時間範圍
        │
        ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌─────────────┐
│  A 抓取資料   │──▶│  B 編碼 + RAG 檢索 │──▶│ C 融合 Z_fused │──▶│ C 分類/RL/回測 │
│ 圖片/新聞/財報 │    │  H_v, H_t, H_r    │    │ Cross-Modal   │    │  驗證與績效    │
└─────────────┘    └──────────────────┘    └──────────────┘    └─────────────┘
   module_a_data      module_b_encoder       module_c_fusion     module_c_fusion
```

對時間範圍內每個交易日重複執行，最終產出數千筆 Z_fused。

---

## 分工與目錄對應

三人平行開發，**只透過約定好的資料格式銜接**（見 `docs/data_format.md`）。每個人只需要在自己的模組目錄內工作，不修改他人模組。

| 負責人 | 模組目錄 | 職責 | 交付物 |
|---|---|---|---|
| A | `module_a_data/` | 資料工程：爬蟲、K 線圖產生、文字清洗、Chunk 切分、漲跌標籤 | 統一 JSON 資料集（`data/processed/dataset/`），一天一筆 |
| B | `module_b_encoder/` | ViT / FinBERT 編碼、RAG 向量庫與檢索、事件抽取 | 每日 H_v、H_t、H_r 向量（`data/vectors/`），格式固定 |
| C | `module_c_fusion/` | Cross-Modal Transformer + QLoRA、分類驗證、PPO、回測 | 可依市場狀態輸出投組建議的完整系統與回測績效報告 |

共用程式（資料格式驗證、路徑常數、共用工具）放在 `shared/`，由三人共同維護；改動 `shared/` 前先在群組告知。

## 專案結構

```
MultiFinRL/
├── README.md                     # 本文件
├── requirements.txt              # 統一開發環境（唯一依據）
├── .gitignore
├── configs/
│   └── config.yaml               # 全域參數：股票池、日期範圍、K 值、圖片尺寸等
├── docs/
│   ├── data_format.md            # ★ 資料格式契約（A/B/C 銜接的唯一標準）
│   └── decisions.md              # 已確認的規格決策紀錄（含負責人 Q&A）
├── samples/
│   ├── DataStruct.example.json   # A 產出格式範例
│   └── vectors_index.example.json# B 產出格式範例
├── shared/
│   ├── schemas.py                # 資料格式驗證（讀寫雙方都要通過）
│   ├── paths.py                  # 路徑常數，避免硬編路徑
│   └── utils.py
├── module_a_data/                # ═══ A：資料工程 ═══
│   ├── README.md
│   ├── crawler/
│   │   ├── fetch_ohlcv.py        # yfinance 下載 OHLCV
│   │   ├── fetch_news.py         # 財經新聞爬蟲
│   │   ├── fetch_filings.py      # SEC EDGAR 財報
│   │   └── fetch_transcripts.py  # 法說會逐字稿
│   ├── preprocess/
│   │   ├── chart_generator.py    # mplfinance 產生 20 日 K 線圖 PNG
│   │   ├── text_cleaner.py       # HTML 清洗
│   │   └── chunker.py            # 文件切 Chunk（≤512 token）
│   ├── labeling.py               # 漲跌標籤產生
│   └── build_dataset.py          # 彙整輸出每日 JSON
├── module_b_encoder/             # ═══ B：Encoder + RAG + 事件抽取 ═══
│   ├── README.md
│   ├── encoders/
│   │   ├── vision_encoder.py     # ViT → H_v
│   │   └── text_encoder.py       # FinBERT/LLaMA → H_t
│   ├── rag/
│   │   ├── vector_db.py          # FAISS/Milvus 建庫
│   │   └── retriever.py          # top-K 檢索 → H_r
│   ├── event_extraction.py       # 財經事件抽取 + P/R/F1
│   └── generate_vectors.py       # 主程式：每日產出三個向量
├── module_c_fusion/              # ═══ C：Fusion + 驗證 + RL + 回測 ═══
│   ├── README.md
│   ├── fusion/
│   │   ├── model.py              # Cross-Modal Transformer
│   │   └── train.py              # QLoRA 微調
│   ├── validation/
│   │   └── classifier.py         # Z_fused 分類任務驗證
│   ├── rl/
│   │   ├── env.py                # PPO 環境與獎勵函數
│   │   └── train_ppo.py
│   └── backtest/
│       └── backtest.py           # Sharpe、MDD、累積報酬
├── scripts/
│   └── run_pipeline.py           # 串接 A→B→C 的完整 pipeline 入口
└── data/                         # 不進 git（見 .gitignore），本機/雲端存放
    ├── raw/                      # A 的原始資料
    │   ├── ohlcv/  charts/  news/  filings/  transcripts/
    ├── processed/dataset/        # A 的交付物：每日 JSON
    ├── vectors/                  # B 的交付物：.npy 向量 + index JSON
    └── outputs/                  # C 的產出：模型權重、Z_fused、回測報告
```

## 資料流與格式（摘要）

完整定義見 **`docs/data_format.md`**，這裡只列銜接點：

1. **A → B**：`data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json`，一天一筆，含 K 線圖路徑、新聞列表（含 `days_ago` 欄位）、財報/法說會 Chunk、漲跌標籤。
2. **B → C**：`data/vectors/{TICKER}/{YYYY-MM-DD}/` 內存 `H_v.npy`、`H_t.npy`、`H_r.npy` 三個檔案，外加一份 `index.json` 記錄 shape 與來源。
3. **C 產出**：`data/outputs/` 內存 Z_fused、模型 checkpoint、回測報告。

## 已確定的全域規格

| 項目 | 規格 |
|---|---|
| 市場 / 初始股票池 | 美股，先以 AAPL 為主，後續擴充（NVDA 等大型個股；指數/ETF 因無財報暫不納入） |
| 資料時間範圍 | 5 年 |
| K 線圖 | 往前 20 個交易日，PNG、224×224、RGB 三通道 |
| 漲跌標籤 | 未來第 5 個交易日收盤價 vs 當日收盤價之報酬：> +2% → BULLISH；< −2% → BEARISH；其餘 → NEUTRAL |
| 文字 Chunk | 每段最長 512 token（FinBERT 輸入上限） |
| RAG 檢索 | K = 3 |
| 新聞缺漏處理 | 以前幾日新聞補上，並附 `days_ago` 欄位供模型判斷相關性 |
| 微調方式 | QLoRA（單機 RTX 5090 可訓練） |
| 開發環境 | 以本 repo 的 `requirements.txt` 為唯一依據 |

## 執行順序

**第一階段（平行進行）**
- A：依格式開始抓資料，先交付 50–100 筆樣本
- B：用假資料把 ViT / FinBERT / RAG 架構跑通（一張圖進去、一個向量出來、存成正確格式）
- C：用假向量把 Fusion Transformer 跑通（三個向量進去、Z_fused 出來）

**第二階段（A 資料量足夠後）**
- A：持續完善資料
- B：換成真實資料，產出真實 H_v / H_t / H_r
- C：換成真實向量，正式訓練 Fusion、接 RL、做回測

## 快速開始

```bash
git clone <repo-url>
cd MultiFinRL
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

各模組的執行方式見各自目錄下的 `README.md`。

## 協作規則

- `main` 分支保持可執行；各自在 `feat/a-*`、`feat/b-*`、`feat/c-*` 分支開發，PR 合併。
- 資料檔（`data/`）不進 git，透過雲端硬碟同步；repo 只放程式與格式範例。
- 任何**資料格式的變更**必須先更新 `docs/data_format.md` 與 `shared/schemas.py`，並通知另外兩人，才能改自己的程式。
