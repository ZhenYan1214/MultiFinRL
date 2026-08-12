# 計畫現況總覽（2026-08）

本文件整理正式計畫書（`docs/reference/115WFAA310699_CM03.pdf`）的完整架構，逐項對照目前程式碼的實作狀態。所有「已解決」「已實作」的判斷，皆有 `docs/decisions.md` 對應編號可查；本文件不是新的決議，是現有決議的整理彙總，之後計畫有變動仍以 `docs/decisions.md` 為準。

## 1. 計畫全貌（PDF 架構）

### 研究定位
題目：「A Multimodal Retrieval-Augmented Financial Decision Framework with Reinforcement Learning」。核心主張：把多模態大語言模型（MLLM）與 RAG 定位為「市場信念構造者」，把異質金融訊號（K 線圖、新聞、財報、法說會、檢索文件）壓縮成一個潛在信念狀態 Z_fused，再用強化學習（RL）在這個信念狀態上訓練投資決策代理人，並加上跨模態可解釋性機制。三大主軸：(1) 多模態信念建構、(2) RL 序列決策優化、(3) 跨模態可解釋性。

### 兩年分期（page 7-8、page 19-21）

**第一年 / Phase 1：感知與信念建構**
數據工程與清洗、RAG 向量資料庫建置、MLLM 信念編碼器訓練（PEFT/QLoRA）、Z_fused 表徵效能驗證（診斷分類）。目標：產出可靠的 Z_fused。

**第二年 / Phase 2：決策優化與解釋性**
RL 環境構建（MDP）、PPO 策略代理人訓練、跨模態歸因解釋模組（Integrated Gradients）、全系統整合與歷史回測。目標：用 Z_fused 訓練出可獲利、風險敏感、可解釋的交易策略。

（`decisions.md #18`：團隊實際執行把兩年壓縮成一年內的「第一階段／第二階段」，內容不變，只是時程重排。）

### 架構逐節對照（3.1 - 3.6）

| 節次 | 內容 | 產出 |
|---|---|---|
| 3.1 Vision encoder | K 線圖／圖表，用財經圖表語料預訓練的視覺編碼器（CLIP 或 ViT） | H_v |
| 3.2 Text encoder | 新聞／財報／法說會文字，用 FinBERT／FinancialBERT／domain-adapted LLaMA/BLOOM | H_t |
| 3.3 RAG | 對 SEC filings、分析師報告、新聞、總經公告做相似度檢索，取 top-K | H_r |
| 3.4 Cross-Modal Fusion and Decoder | Transformer 融合 [H_t;H_v;H_r] 得 Z_fused；decoder（LLaVA 或 domain-adapted LLaMA-2）以 Z_fused 為條件生成結構化敘述 Y | Z_fused、生成敘述 Y |
| 3.5 Training | PEFT/QLoRA 微調；三個 loss 組成：L_align（對比對齊）+ L_ground（RAG 證據 grounding）+ L_belief（生成結構化信念 token） | 訓練好的第一階段模型 |
| 3.6 RL & 可解釋性 | MDP（state=[Z_fused, 前期持倉]）、PPO 訓練、風險敏感 reward（報酬-波動-回撤-交易成本）、Integrated Gradients 歸因 | 交易策略 + 歸因解釋 |

### 第 19 頁正式工作項目清單（逐年，必達）

**第一年**：數據工程與清洗、RAG 模組開發、信念建構模型訓練（PEFT/QLoRA）、**表徵效能驗證**（對 Z_fused 做診斷分類測試，驗證市場情緒分類與事件抽取準確度）。

**第二年**：RL 環境構建、PPO 策略網路訓練、跨模態歸因分析（Integrated Gradients）、系統整合與回測（Sharpe Ratio、MDD）。

## 2. 目前實作對照表

| PDF 元件 | 對應程式碼 | 現況 |
|---|---|---|
| 3.1 Vision encoder (H_v) | `module_b_encoder/encoders/vision_encoder.py` | 已實作，但用通用 ImageNet 預訓練 ViT（`google/vit-base-patch16-224`），**不是**計畫書要求的財經圖表語料預訓練版本，完全凍結、無領域調整。已知落差（`#10`、`#36`），domain gap 對照實驗建議過但尚未執行 |
| 3.2 Text encoder (H_t) | `module_b_encoder/encoders/text_encoder.py` | 已實作，用 FinBERT（`ProsusAI/finbert`），符合計畫書建議選項之一 |
| 3.3 RAG (H_r) | `module_b_encoder/rag/retriever.py`、`vector_db.py` | 已實作，做法（query 加權合併、相似度檢索、top-K 重新編碼）符合計畫書 3.3 節描述；K=3 |
| 3.4 Cross-Modal Fusion (Z_fused) | `module_c_fusion/fusion/model.py` | 已實作，Transformer 融合 H_v/H_t/H_r（結構符合公式），但 H_t/H_r 先各自 mean-pool 成單一 token 再進融合層（簡化版，控制序列長度） |
| 3.4 Decoder（LLaVA/LLaMA-2 生成敘述） | 無 | **未實作**。已知必要（`#29`），屬第一階段，需 A 端先產出結構化敘述標準答案（y_belief）才能訓練，待老師確認優先順序 |
| 3.5 Training（QLoRA + L_align + L_ground + L_belief） | `module_c_fusion/fusion/train.py` | **簡化版**：目前只用市場情緒分類的 cross-entropy loss 端到端訓練融合層，程式檔頭本身已註明「L_align + L_ground + L_belief 為後續強化」。三個 PDF 指定的 loss 皆未實作。因為沒有真正的大型 MLLM backbone，QLoRA/PEFT 也未使用 |
| 3.6 MDP / PPO | `module_c_fusion/rl/env.py`、`train_ppo.py` | 已實作，state=[Z_fused, 前期持倉]、reward=報酬−λ_vol×波動−λ_mdd×回撤−交易成本，跟計畫書公式對得上；目前僅單股+現金二維動作空間，多資產未擴充 |
| Curriculum learning | 無 | **未實作**（`#29`），`train_ppo.py` 直接用全部歷史資料訓練 |
| Integrated Gradients（跨模態歸因） | 無 | **未實作**（`#29`），屬第二階段，待老師確認優先順序 |
| 回測（Sharpe/MDD） | `module_c_fusion/backtest/backtest.py` | 已實作，三種策略對照（buy_and_hold/rule_based/ppo），僅支援單一股票，多資產投組未擴充（`#29`） |
| Z_fused 表徵效能驗證（市場情緒） | `module_c_fusion/validation/classifier.py` | 已實作 |
| Z_fused 表徵效能驗證（事件抽取，page 19 明訂） | `module_c_fusion/validation/event_validation_head.py` | 已實作，5 類 micro f1=0.229，計畫書 page 19 明確要求（`#34`、`#37`） |
| 事件抽取（`event_extraction.py`，讀 A 原始資料） | `module_b_encoder/event_extraction.py` | 已實作（keyword + LLM 兩種方法），但**計畫書無此項目依據**，定位為 Track A 資料品質檢查的附屬分析（`#35`、`#37`），非系統模組 |
| ETF/指數總經資料 | `module_a_data/crawler/fetch_macro.py` | 只有骨架（`NotImplementedError`），教授已確認方向（CPI/PCE/點陣圖，`#38`），目前計畫仍以 AAPL 個股為主，未實際擴充 |

## 3. 目前進行中、尚未定案的討論

負責人提案「三版本 Z_fused 消融實驗」（把事件抽取結果當作第四個輸入 H_e：`Z_fused = F([H_t;H_v;H_r;H_e])`，無/關鍵字/LLM 三版本比較，寫進論文的相關性分析），以及提到 3.4 節的 decoder（LLaVA/LLaMA-2）要接在 Z_fused 之後做最終預測。詳見 `decisions.md #39`：這是負責人自己的論文方法論提案，非計畫書要求；decoder 部分確認屬 PDF 3.4 節、第一階段，但目前完全未實作。三個疑慮已回問負責人，待回覆：(1) 三版本比較的「準確度」指市場情緒還是事件本身；(2) 若為後者，149 天事件 ground truth 是 LLM 標記的，LLM-based 方法對比 LLM 標記答案有循環判斷疑慮；(3) 負責人原本誤以為事件抽取是 RAG 的 H_r 之一，已查證 PDF 3.3 節澄清兩者無關，待負責人核對 PDF/GitHub 後回覆。

## 4. 二次查證記錄（逐項對照程式碼實際內容，非憑印象）

上面第 2 節「已實作」的每一項，這次都重新直接讀了程式碼原文（不是靠記憶）交叉核對，結果：

- **QLoRA/PEFT**：`grep -rin "lora|qlora|peft"` 掃過 A/B/C/shared 全部程式碼，只有 `fusion/train.py` 檔頭註解提到「後續強化」，沒有任何實際實作，確認完全沒做。
- **PPO**：`train_ppo.py` 讀原始碼確認真的呼叫 `from stable_baselines3 import PPO`、`PPO("MlpPolicy", env, ...)`，不是假的或占位程式碼。
- **Reward 公式**：`env.py` 逐項比對計畫書公式 `R_t = μ(Rp) − λ1·σ(Rp) − λ2·MDD_t − η·TC_t`：程式碼的 `reward = r_portfolio − lambda_vol*vol − lambda_mdd*drawdown`，其中 `r_portfolio` 已扣除 `turnover*cost`（對應 TC 項）。**一個精確的落差要指出**：計畫書的 μ(Rp) 是「預期報酬」的統計量，程式碼直接用單步「已實現報酬」代入，不是真的算期望值——這是強化學習常見的合理簡化（單步 reward 本來就是即時訊號，不是要求先算好分布再代入），不算錯誤，但跟公式不是逐字對應，這裡精確講出來。
- **回測 Sharpe/MDD**：讀 `backtest.py` 原始碼確認 `sharpe_ratio()`、`max_drawdown()` 兩個函式的計算邏輯正確（年化 Sharpe 用 252 個交易日、MDD 用累積峰值回撤），對應計畫書 page 19「比較夏普比率與最大回撤」的要求。
- **文字編碼器**：讀 `text_encoder.py` 確認用 `ProsusAI/finbert`，是計畫書 3.2 節列的三個建議選項之一（FinBERT／FinancialBERT／domain-adapted LLaMA/BLOOM）；池化方式用 mean pooling，也是計畫書 3.2 節明講「可以是 [CLS] 或 mean pooling」允許的其中一種，不是隨便選的。
- **市場情緒診斷分類**：讀 `classifier.py` 確認輸入真的是 `Z_fused`（透過 `load_index`/`load_z_and_labels`），不是誤用其他向量，對應計畫書 page 19「表徵效能驗證」市場情緒的部分。
- **Integrated Gradients／curriculum learning**：`grep -rin "captum|integrated.gradient|curriculum"` 掃過全部程式碼，完全沒有找到，確認兩項都是真的零實作，不是我之前漏看。

以上皆為這一輪重新查證的結果，跟第 2 節的判斷一致，沒有發現需要修正的地方。

**補充：PDF 圖片內容查證（`#40`）**。用 `pypdf` 掃過全部 24 頁，確認整份計畫書只有 1 張內嵌圖片（page 8，Figure 1 架構圖），已直接抽出查看，內容跟正文架構描述一致，沒有矛盾之處。新發現一個較輕微的細節：Figure 1 把 Visual Inputs 畫成「Candlestick Charts」與「Technical Indicators」兩個獨立方框，暗示技術指標可能也要單獨視覺化，正文 3.1 節沒有這樣明講；目前 `chart_generator.py` 只產生 K 線圖，沒有另外做技術指標視覺化，是否要補上待團隊評估，非緊急。Table 1、Table 2、page 20-21 甘特圖皆為文字/區塊字元排版，非圖片，先前純文字擷取已完整涵蓋，確認沒有遺漏。

## 5. 一句話總結現況

第一階段（感知與信念建構）的核心管線——A 的資料收集/標籤、B 的三種編碼器（H_v/H_t/H_r）、C 的融合模型（Z_fused）、兩種表徵效能驗證（市場情緒分類 + 事件診斷分類）——全部跑通且有實測數字；第二階段（RL 決策）的 PPO/reward/回測骨架也已建好、隨時可跑。真正的落差集中在三處：(1) 視覺編碼器沒有做領域預訓練（3.1 節要求 vs 現況）；(2) 訓練沒有用計畫書指定的三個 loss（L_align/L_ground/L_belief）和 QLoRA，是簡化版分類訓練；(3) 生成式 decoder、跨模態可解釋性（Integrated Gradients）、curriculum learning、多資產回測，四項都還沒動工，已知必要但待老師確認優先順序後再排入排程。
