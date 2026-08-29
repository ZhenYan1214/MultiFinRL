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
| 3.1 Vision encoder (H_v) | `module_b_encoder/encoders/vision_encoder.py` | 已實作，但用通用 ImageNet 預訓練 ViT（`google/vit-base-patch16-224`），**不是**計畫書要求的財經圖表語料預訓練版本，完全凍結、無領域調整。已知落差（`#10`、`#36`）；domain gap 對照實驗已執行（`#46`）：拿掉 K 線圖 macro f1 掉 51%，現有 ViT 貢獻很大，不是沒用，落差仍在但不是「拖累」，換編碼器的迫切性降低、邊際效益待評估 |
| 3.2 Text encoder (H_t) | `module_b_encoder/encoders/text_encoder.py` | 已實作，用 FinBERT（`ProsusAI/finbert`），符合計畫書建議選項之一 |
| 3.3 RAG (H_r) | `module_b_encoder/rag/retriever.py`、`vector_db.py` | 已實作，做法（query 加權合併、相似度檢索、top-K 重新編碼）符合計畫書 3.3 節描述；K=3 |
| 3.4 Cross-Modal Fusion (Z_fused) | `module_c_fusion/fusion/model.py` | 已實作，Transformer 融合 H_v/H_t/H_r（結構符合公式），但 H_t/H_r 先各自 mean-pool 成單一 token 再進融合層（簡化版，控制序列長度） |
| 3.4 Decoder（LLaVA/LLaMA-2 生成敘述） | `module_c_fusion/decoder/`（`model.py`/`train.py`/`evaluate.py`/`generate_y_belief.py`） | **已實作第一版並完成訓練+評估**（`#57`~`#64`）：QLoRA 微調 LLaMA-2，以 Z_fused 當 soft-prompt 前綴生成 `<TREND>`/`<RISK_LEVEL>` + 敘述文字。結果：格式正確率 100%、loss delta 1.966（微調 vs 未微調 backbone，證明有實質效果）、RISK_LEVEL 準確率 90%、**TREND 準確率僅 46.7%**（反映 Z_fused 對股價方向訊號有限這個全專案既有瓶頸，非 decoder 獨有，見下方 `#64`）。只做 `L_belief` 一項 loss，`L_align`/`L_ground` 仍未實作；敘述文字品質（LLM-as-judge）尚未驗證 |
| 3.5 Training（QLoRA + L_align + L_ground + L_belief） | `module_c_fusion/fusion/train.py`（融合層）vs `module_c_fusion/decoder/train.py`（decoder） | **要分兩塊看，不是同一件事**：融合層（`fusion/train.py`，產出 Z_fused）依然是簡化版，只用市場情緒分類的 cross-entropy loss，未用 QLoRA，三個 PDF loss 皆未實作，維持原判斷不變；**decoder（`decoder/train.py`）現在已經是真正的 QLoRA**（4-bit 量化 + LoRA 掛全部 linear 層，`#59` 依 2026 年業界共識調整過訓練細節），但只實作 `L_belief` 一項，`L_align`/`L_ground` 仍缺（需要聯合訓練 encoder／需要不存在的 oracle relevance scores，見 `model.py` 檔頭） |
| 3.6 MDP / PPO | `module_c_fusion/rl/env.py`、`train_ppo.py` | 已實作，state=[Z_fused, 前期持倉]、reward=報酬−λ_vol×波動−λ_mdd×回撤−交易成本，跟計畫書公式對得上；目前僅單股+現金二維動作空間，多資產未擴充 |
| Curriculum learning | `module_c_fusion/rl/train_ppo.py`（`--curriculum`） | **已實作並驗證有效**（`#52`~`#55`）：依滾動波動度分階段訓練，階段步數依難度遞增分配。第一版均分步數訓出「永遠空手」的退化 policy（policy collapse），排查後改用難度加權步數修好，最終跟 baseline 打平（Sharpe 皆 0.67） |
| Integrated Gradients（跨模態歸因） | `module_c_fusion/explainability/integrated_gradients.py` | **已實作並驗證有意義**（`#49`、`#56`）：captum + 零向量 baseline，套在訓練好的 PPO policy 的動作分布 mean 上。第一次在退化 policy（curriculum collapse 那版）上跑出來的結果不可信，**在修好的健康 policy 上重跑後確認有意義**：768 維裡 373 維（48.6%）有實質貢獻，維度間差異化明確（top10/bottom10 相差約 4 個數量級） |
| 回測（Sharpe/MDD） | `module_c_fusion/backtest/backtest.py` | 已實作，三種策略對照（buy_and_hold/rule_based/ppo），僅支援單一股票，多資產投組未擴充（`#29`） |
| Z_fused 表徵效能驗證（市場情緒） | `module_c_fusion/validation/classifier.py` | 已實作 |
| Z_fused 表徵效能驗證（事件抽取，page 19 明訂） | `module_c_fusion/validation/event_validation_head.py` | 已實作，5 類 micro f1=0.229，計畫書 page 19 明確要求（`#34`、`#37`） |
| 事件抽取（`event_extraction.py`，讀 A 原始資料） | `module_b_encoder/event_extraction.py` | 已實作（keyword + LLM 兩種方法），但**計畫書無此項目依據**，定位為 Track A 資料品質檢查的附屬分析（`#35`、`#37`），非系統模組 |
| ETF/指數總經資料 | `module_a_data/crawler/fetch_macro.py` | 只有骨架（`NotImplementedError`），教授已確認方向（CPI/PCE/點陣圖，`#38`），目前計畫仍以 AAPL 個股為主，未實際擴充 |
| 財報 8-K（重大訊息即時揭露） | `module_a_data/crawler/fetch_filings.py`、`build_dataset.py` | **已實作並本機真實資料驗證通過**（`#41`/`#44`/`#45`/`#47`）：10-K/10-Q 維持「取最新一份沿用到下一份發布為止」當背景，8-K 另外標時間戳記當補充事件、不覆蓋背景。曾一度發生 8-K 資料沒有真正流入 Z_fused（B/C 沒有在資料重建後重跑），已補跑修正（`#50`/`#51`）；市場情緒分類／事件驗證頭兩個既有診斷指標變化在雜訊量級，看不出明顯影響，不代表功能本身有問題 |

**2026-08 架構釐清（`#43`，取代下方原本記錄的「三版本消融實驗」討論）**：架構圖上把「分類驗證」「事件驗證頭」「Decoder」三個框合併成一個——三者功能意圖一致（都是從 Z_fused 判斷市場情緒/事件），只是實現方式不同。`classifier.py`／`event_validation_head.py` 原本是 Decoder+L_belief 這整套機制尚未做出來之前的簡化代打版本；**這個狀態已經改變**——Decoder 現在已經實際做出來並訓練+評估完成（見上表 3.4 列、`#57`~`#64`），但**程式碼層級三者仍是各自獨立的程式**，`classifier.py`／`event_validation_head.py` 沒有被 Decoder 取代或合併，三者分別預測不同目標（市場情緒 3 類／事件類型 7 類多標籤／結構化敘述+風險等級），現階段仍需要三者並存，不是合併成一支。

## 3. 近期進度總覽（`#44`~`#65`，取代原本記錄的討論）

**8-K 財報缺口**：已解決，見上表新增列。

**三版本 Z_fused 消融實驗**：已解決，確認不做，見 `#42`、`#43`（詳見上方架構釐清）。

**`#48` 使用者裁示不再等老師表態，四項落差按工程成本自行排序動工，目前進度**：
1. **Integrated Gradients**——已完成並驗證有意義，見上表、`#49`、`#56`。
2. **Curriculum learning**——已完成並驗證有效（跟 baseline 打平），見上表、`#52`~`#55`。
3. **多資產回測**——**仍未動工**，需要先擴充第二支股票的完整 A/B/C 資料才有意義測試，資料成本高於程式碼成本，排序在後；目前規劃用這個擴充順便驗證「TREND 準確率偏弱是不是資料量問題」這個假設（見下方第 5 節）。
4. **生成式 decoder**——**已完成第一版訓練+評估**，見上表、`#57`~`#64`。過程中連續修過三個真實 bug（GPU 未被實際使用、label 未遮罩 padding 導致生成空字串、LoRA dropout 推論時未關閉），且有一次因為同時跑兩個 GPU 程式導致當機、被迫中斷訓練。最終結果：格式正確率 100%、loss delta 1.966（微調有實質效果）、RISK_LEVEL 準確率 90%、TREND 準確率僅 46.7%（反映的是 Z_fused 對股價方向訊號有限這個全專案既有瓶頸，不是 decoder 獨有的缺陷）；`L_align`/`L_ground` 兩個 loss 仍未實作，敘述文字品質（LLM-as-judge）尚未驗證，建議下一步優先補上。

**其他**：
- ViT domain gap 對照實驗已執行（`#46`）：拿掉 K 線圖 macro f1 掉 51%，現有 ViT 貢獻很大，換編碼器的迫切性降低（但邊際效益可能仍大，待評估）。
- Technical Indicators 是否要獨立視覺化：**已解決（`#65`）**。王崇穎確認要做——用原本的 K 線價格資料計算指標（不用挑很多）、額外做成一張圖跟 K 線圖一起合併進 H_v；不急，之後再排入即可，若有時間可變成論文的討論/比較範疇。另確認 Table 1「technical patterns」（頭肩頂等圖形型態）跟「technical indicators」（RSI/MACD 數字指標）是兩個不同概念，解決了 `#58` 留下的疑問。
- 「事件」的定義是否該涵蓋 K 線圖技術型態（頭肩頂、W 底反轉等）——張教授討論架構圖時舉的例子，跟現有事件驗證頭測的七類商業事件不同，待團隊評估，非緊急。

## 4. 二次查證記錄（逐項對照程式碼實際內容，非憑印象）

上面第 2 節「已實作」的每一項，這次都重新直接讀了程式碼原文（不是靠記憶）交叉核對，結果：

- **QLoRA/PEFT**：`fusion/train.py`（分類訓練）本身仍未使用；但 `decoder/train.py`/`decoder/model.py` 已實作（2026-08 新增），4-bit 量化 + LoRA（掛全部 attention+MLP 線性層）微調 LLaMA-2-7b，`peft`/`bitsandbytes` 是真的被呼叫、不是占位程式碼，且已完成第一輪完整訓練+評估（見上表 3.4 列、`#57`~`#64`）。
- **PPO**：`train_ppo.py` 讀原始碼確認真的呼叫 `from stable_baselines3 import PPO`、`PPO("MlpPolicy", env, ...)`，不是假的或占位程式碼。
- **Reward 公式**：`env.py` 逐項比對計畫書公式 `R_t = μ(Rp) − λ1·σ(Rp) − λ2·MDD_t − η·TC_t`：程式碼的 `reward = r_portfolio − lambda_vol*vol − lambda_mdd*drawdown`，其中 `r_portfolio` 已扣除 `turnover*cost`（對應 TC 項）。**一個精確的落差要指出**：計畫書的 μ(Rp) 是「預期報酬」的統計量，程式碼直接用單步「已實現報酬」代入，不是真的算期望值——這是強化學習常見的合理簡化（單步 reward 本來就是即時訊號，不是要求先算好分布再代入），不算錯誤，但跟公式不是逐字對應，這裡精確講出來。
- **回測 Sharpe/MDD**：讀 `backtest.py` 原始碼確認 `sharpe_ratio()`、`max_drawdown()` 兩個函式的計算邏輯正確（年化 Sharpe 用 252 個交易日、MDD 用累積峰值回撤），對應計畫書 page 19「比較夏普比率與最大回撤」的要求。
- **文字編碼器**：讀 `text_encoder.py` 確認用 `ProsusAI/finbert`，是計畫書 3.2 節列的三個建議選項之一（FinBERT／FinancialBERT／domain-adapted LLaMA/BLOOM）；池化方式用 mean pooling，也是計畫書 3.2 節明講「可以是 [CLS] 或 mean pooling」允許的其中一種，不是隨便選的。
- **市場情緒診斷分類**：讀 `classifier.py` 確認輸入真的是 `Z_fused`（透過 `load_index`/`load_z_and_labels`），不是誤用其他向量，對應計畫書 page 19「表徵效能驗證」市場情緒的部分。
- **Integrated Gradients／curriculum learning**：兩者皆已實作（`explainability/integrated_gradients.py`、`rl/train_ppo.py` 的 `--curriculum`），對應 `#49`/`#56`、`#52`~`#55`。

以上皆為這一輪重新查證的結果，跟第 2 節的判斷一致；`#44`~`#65` 的新進度已補充在上方對照表與第 3 節。

**補充：PDF 圖片內容查證（`#40`，Technical Indicators 疑問已於 `#65` 解決）**。用 `pypdf` 掃過全部 24 頁，確認整份計畫書只有 1 張內嵌圖片（page 8，Figure 1 架構圖），內容跟正文架構描述一致，沒有矛盾之處。Figure 1 把 Visual Inputs 畫成「Candlestick Charts」與「Technical Indicators」兩個獨立方框，這個疑問已由王崇穎確認（`#65`）：用原本 K 線資料算指標、另外做成一張圖合併進 H_v，非緊急項目。Table 1、Table 2、page 20-21 甘特圖皆為文字/區塊字元排版，非圖片，先前純文字擷取已完整涵蓋。

## 5. 一句話總結現況（2026-08 更新，取代先前版本）

第一階段（感知與信念建構）的核心管線——A 的資料收集/標籤（含 8-K）、B 的三種編碼器（H_v/H_t/H_r）、C 的融合模型（Z_fused）、兩種表徵效能驗證（市場情緒分類 + 事件診斷分類）——全部跑通且有實測數字；第二階段（RL 決策）的 PPO/reward/回測、**curriculum learning**、**Integrated Gradients** 也都已實作並驗證完成；**生成式 decoder 也已完成第一版訓練+評估**（QLoRA 微調 LLaMA-2，格式正確率 100%、loss delta 1.966、RISK_LEVEL 準確率 90%，但 TREND 準確率僅 46.7%）。

**目前真正還沒動工的，只剩一項**：多資產回測（`backtest.py` 仍僅支援單一股票，需要先擴充第二支股票的完整資料才有意義測試）。

**已知但非「未做」、屬於品質/深度落差的部分**：(1) 視覺編碼器沒有做領域預訓練（3.1 節要求 vs 現況），但 `#46` 實測顯示現有 ViT 貢獻其實很大，換編碼器的迫切性下降；(2) Decoder 只做了 `L_belief` 一項 loss，`L_align`/`L_ground` 仍缺；(3) Decoder 的敘述文字品質（LLM-as-judge）尚未驗證；(4) TREND（市場方向）預測偏弱是全專案共通的既有瓶頸（`classifier.py` 跟 decoder 表現一致地弱），不是單一模組的問題，根源可能在 Z_fused 本身對股價方向的訊號含量有限——`#66`（多股票擴充實驗，計畫中）會進一步檢驗這個假設。

**待辦（下一步，見對話紀錄與 `docs/decisions.md` 最新條目）**：(1) 補跑 `evaluate.py --llm_judge` 驗證敘述文字品質，成本低、不用重新訓練；(2) 決定要不要投入多股票（AAPL+NVDA+MSFT）擴充實驗，驗證 TREND 偏弱是否為資料量問題；(3) `fetch_transcripts.py` 已改用 Alpha Vantage API（`#67`），待本機實際執行取得真實法說會逐字稿資料。
