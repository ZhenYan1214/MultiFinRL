# A 資料來源紀錄 + B/C 實驗結果紀錄

這份文件持續更新，不是一次性報告。新增資料來源、重跑一次分類驗證，都回來這裡補一行。

## 一、資料來源（截至 2026-08-08）

| 資料類型 | 來源 | 抓取腳本 | 目前涵蓋範圍 | 狀態 |
|---|---|---|---|---|
| 股價 OHLCV | yfinance | `fetch_ohlcv.py` | AAPL 2021-01-04 ~ 2026-08-07（1405 筆） | 2026-08-09 延伸完成，對齊新聞涵蓋範圍 |
| K 線圖 | 用 OHLCV 自畫（mplfinance，20 日窗口） | `chart_generator.py` | 同上，1386 張（頭尾幾天因窗口不足缺圖） | 2026-08-09 延伸完成 |
| 成交量圖 | 用 OHLCV Volume 自畫（20 日窗口） | `chart_generator.py` | 同上，1386 張 | 2026-09-19 完成；作為雙圖 ViT 的第二張輸入 |
| 新聞（近期） | yfinance | `fetch_news.py` | 僅最新 8~10 則，歷史價值低 | 可用，範圍小 |
| 新聞（歷史回補） | Alpaca News API（來源 Benzinga，官方 API 非爬蟲） | `fetch_news_alpaca.py` | AAPL 2021-01-01 ~ 2026-08-08，15,989 則 / 1,459 天 | 2026-08 新增，已抽樣驗證為真實全文 |
| 財報 10-K/10-Q | SEC EDGAR（官方 API，原生 requests） | `fetch_filings.py` | 22 份，2021 ~ 2026 | 可用；唯一版本，`fetch_filings_edgartools.py` 候選方案已刪除（decisions.md #41，取代 #31） |
| 法說會逐字稿 | Alpha Vantage API（官方，取代原本從未成功端到端跑過的 foolcalls） | `fetch_transcripts.py` | 待本機執行後補上實際涵蓋範圍 | 2026-08 改用 Alpha Vantage，`foolcalls/` 已移除，見 decisions.md #67 |

**已放棄的路徑**（保留記錄，程式碼已刪除）：FNSPID 爬蟲工具與 FNSPID 現成 HuggingFace dataset，兩者都驗證不可行（自動化偵測擋爬蟲、頁面改版、dataset 全文欄位是空的）。詳見 `docs/decisions.md` #26。

**已解決的落差**：OHLCV/K 線圖原本只到 2025-12-30、新聞已回補到 2026-08-08 的範圍不對齊問題，已於 2026-08-09 延伸 OHLCV/圖表到 2026-08-07 解決（`configs/config.yaml` 的 `date_range.end` 同步更新）。整條 pipeline（`build_dataset.py` → `generate_vectors.py` → `fusion.train` → `classifier.py`）已針對延伸後的新範圍重跑過（decoder 訓練用的 Z_fused/y_belief 交集樣本數 1381 天即為此次重跑後的結果，見 `docs/decisions.md` decoder 相關條目）。

## 二、B/C 實驗結果紀錄（分類驗證準確率）

每次重跑 `classifier.py` 都在這裡加一行。程式會自動保留
`classification_report_{TICKER}_{run_id}.json`，把該檔填入 `detail_file` 方便回頭比對。

| 日期 | 資料版本/這次改了什麼 | n_train/val/test | accuracy | BEARISH f1 | NEUTRAL f1 | BULLISH f1 | 備註 / detail 檔案 |
|---|---|---|---|---|---|---|---|
| 2026-08-08（之前） | 新聞僅 yfinance 近期，歷史期間新聞幾乎全空（7 天回補窗口內大多抓不到） | 861 / 184 / 185 | 0.459 | 0.000 | 0.621 | 0.090 | 模型幾乎塌縮成只猜 NEUTRAL，準確率只比「每天都猜 NEUTRAL」（44%）高一點；`data/outputs/metrics/classification_report_baseline_no_news.json` |
| 2026-08-08（之後） | + Alpaca 歷史新聞回補（重跑 build_dataset → generate_vectors → fusion.train → classifier，OHLCV/圖表範圍不變，仍是 2021~2025） | 861 / 184 / 185 | 0.465 | 0.000 | 0.626 | 0.116 | 準確率只多 0.9 個百分點，在 185 筆測試集裡等於大約多猜對 1 天，不算有意義的差異；BEARISH 完全沒動（還是 0）；BULLISH f1 從 0.090 到 0.116，support 只有 64，等於大概多對 1 題，也在雜訊範圍內；NEUTRAL 幾乎沒變、recall 仍是 1.0，模型還是幾乎每天都猜 NEUTRAL。結論：新聞資料補齊本身沒有明顯改善這個下游分類結果，瓶頸應該在別的地方（label 定義、類別不平衡、訓練集只有 861 筆、ViT 沒微調等），不是新聞量不夠。`data/outputs/metrics/classification_report.json` |
| 2026-08-08（診斷實驗，見 spec_c_accuracy_diagnostics.md） | 新聞有無 × 類別加權有無，四種組合對照（詳見下方獨立小節） | 861 / 184 / 185 | 0.465 / 0.465 / 0.314 / 0.286 | 0.000 / 0.000 / 0.196 / 0.165 | 0.626 / 0.626 / 0.406 / 0.377 | 0.116 / 0.116 / 0.308 / 0.281 | 四個數字依序對應：有新聞不加權／沒新聞不加權／有新聞加權／沒新聞加權。結論：不加權時新聞完全沒影響（結果逐位數字相同）；加權後模型不再塌縮成只猜 NEUTRAL，且加權狀態下「有新聞」四個指標全面優於「沒新聞」——新聞資料是有真實貢獻的，只是先前被類別不平衡蓋住看不出來。整體 accuracy 加權後變低是預期中的正常現象（模型不再靠猜多數類別灌水），不是變差，判讀要看 macro f1 與少數類別表現 |
| 2026-08-09（延伸日期範圍 + RAG query 修正，加權預設開啟） | OHLCV/K線圖延伸到 2026-08-07（資料集實際到 2026-07-31，最後幾天缺 label）；RAG query 從單純平均改成正規化+alpha 加權（alpha=0.5） | 966 / 207 / 208 | 0.3029 | 0.1875 | 0.3497 | 0.3212 | macro f1≈0.286，跟前一輪「有新聞+加權」（macro f1=0.303）相比持平、略降，沒有明顯進步。**重要限制**：這次同時改了兩件事（延伸範圍 + RAG 修正），且測試集因為時間序切分跟著往後移動（這次測試期間 2025-10-02~2026-07-31，跟前一輪 2025-03-31~2025-12-22 不同），無法從這次比較單獨歸因是哪個改動造成差異，也可能只是新測試期間本身難度不同。要乾淨驗證 RAG 修正的效果，需要同一個日期範圍、只切換 RAG 新舊版本的對照實驗，目前尚未做。`classification_report_2026-08_extended_range_rag_fix.json` |
| 2026-08-10（漲跌標籤改為分位數門檻，見 decisions.md #30） | 只改標籤定義（固定±2% → 分位數1/3門檻），日期範圍、RAG、加權都跟上一輪相同 | 966 / 207 / 208（跟上一輪完全相同的切分與測試期間，可乾淨對照） | 0.3269 | 0.2689 | 0.3313 | 0.3731 | macro f1 從 0.286 提升到 0.324（+0.038），是這一路診斷下來第一次有乾淨、無混雜因素的正向結果——這次 n_train/val/test 筆數與測試期間跟上一輪完全一樣，只有標籤定義變了，可以放心把差異歸因到標籤改動本身。BEARISH f1 進步最多（0.188→0.269），BULLISH 也進步（0.321→0.373），NEUTRAL 略降（0.350→0.331）。測試集類別分布也從原本 BEARISH 明顯偏少（原本 support 39~48）變成三類接近平均（support 65/67/76）。誠實記錄：macro f1=0.324 仍不算「表現良好」，只是目前為止最好的一次。過程中曾因為 sandbox 執行 `build_dataset.py` 中途被 timeout 打斷，導致新舊標籤混雜跑出一次不可信的結果（accuracy=0.3221），已作廢重跑並逐日核對 208 天全部一致才採信這次結果。`classification_report_2026-08_quantile_labels.json` |
| 2026-09-19（雙圖 ViT：K 線 + 成交量） | 同一個 frozen ViT 以 batch 一次編碼兩張獨立圖片，H_v `[197,768]` → `[2,197,768]`；fusion 加入圖別 slot embedding。兩組皆使用 1,381 天、seed=42、weighted、3 epochs | 966 / 207 / 208 | 0.3413（固定相同 H_t/H_r 的單圖對照 0.3173，+0.0240） | 0.3036（0.2810） | 0.3432（0.3353） | 0.3704（0.3281） | macro f1 0.3390，受控單圖對照 0.3148，+0.0242；三類 f1 全數上升。另有導入前完整 pipeline 基準 0.3060，但因 H_v 也參與 RAG query，主要結論採固定 H_t/H_r 的受控結果。只跑一個 seed，尚未做顯著性檢驗。報告：`data/outputs/experiments/dual_image_vit/AAPL/` |

### 視覺輸入實驗：K 線疊加 Bollinger Bands（2026-09-18）

比較 20 根純 K 線圖與「同一張圖疊加 20 日 Bollinger 上／中／下軌」。兩組固定使用相同的
1,362 天（2021-03-01~2026-07-31），依時間切成 953 train／204 validation／205 test；凍結
`google/vit-base-patch16-224`，取 CLS token 訓練相同的 balanced logistic linear probe，以
validation macro F1 決定是否採用疊圖。純 K 線 validation accuracy=0.3971、macro F1=0.3880；
K+BBands accuracy=0.3578、macro F1=0.3537，疊圖的 macro F1 下降 0.0343。勝出的純 K 線在
held-out test accuracy=0.3756、macro F1=0.3692。

結論：不採用 K+BBands 疊圖，正式 pipeline 維持純 K=20。合理推測是三條高對比曲線增加強烈
視覺邊緣、稀釋蠟燭形態，而 BBands 又是收盤價的確定性衍生資訊，沒有增加新的原始資料。
此輪只做視覺模態快速篩選，數字不可直接和完整 Z_fused classifier 比較；實驗程式、圖片與
H_v 快取已依要求刪除，只保留本紀錄。

### 視覺輸入架構：雙圖 ViT（K 線 + 成交量，2026-09-19）

兩張 224×224 圖以一個 batch 通過同一個 frozen `google/vit-base-patch16-224`，保留每張圖各自的
CLS + 196 patch tokens，輸出 `H_v=[2,197,768]`。module_c 對兩個圖別加入可學習的 slot
embedding，再展平成 394 個 vision tokens 與 H_t/H_r 融合；`Z_fused` 仍固定 768 維，因此
classifier、decoder、RL 與 backtest 的 shape 不需更改。第二張圖由 `config.yaml` 的
`chart.vision_inputs` 控制，目前為 volume，日後可換成 `technical/<indicator_set>`。

比較使用完全相同的 1,381 天、966/207/208 時序切分、seed=42、weighted loss、batch=4、
learning rate=1e-4、3 epochs。導入前完整 pipeline 基準 test accuracy=0.3077、macro F1=0.3060；
雙圖為 accuracy=0.3413、macro F1=0.3390。為排除重算 RAG 的混雜因素，另固定雙圖版本的
H_t/H_r，只讓 fusion 看到第一張 K 線重新訓練；這個受控單圖對照為 accuracy=0.3173、macro
F1=0.3148，雙圖仍分別增加 0.0240 與 0.0242。受控對照的 BEARISH/NEUTRAL/BULLISH F1
由 0.2810/0.3353/0.3281 上升到 0.3036/0.3432/0.3704，三類同方向改善。採用雙圖架構。

限制：RAG query 原本就由 H_v 與 H_t 組成，因此換成雙圖 H_v 後 H_r 也會跟著重算；這次量到
的是「雙圖架構導入完整 pipeline」的端到端效果，不是固定 H_r 後只量成交量圖的純視覺
ablation。固定 H_t/H_r 的受控對照已證明第二張圖直接進 fusion 時仍有正向結果，但其 H_r
本身已由雙圖 query 產生；若論文需要把成交量從 ViT 到 RAG 的每條路徑完全拆開，仍需更細的
factorial ablation。以上目前只跑 seed=42 一次，尚未做多 seed 平均或顯著性檢驗。

### 第二張視覺圖比較：Volume / RSI / SMA / MACD（2026-09-19）

固定第一張 20 日 K 線、既有 H_t/H_r 與雙圖 fusion 架構，只替換第二張圖。四組取共同的
1,348 天，嚴格依時間切成 943 train／202 validation／203 test；fusion 僅使用 train label
訓練，validation macro F1 選候選，held-out test 在選定 SMA 後才開啟。設定固定 seed=42、
weighted loss、batch=4、learning rate=1e-4、3 epochs。

| 第二張圖 | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| Volume | 0.2822 | 0.2819 |
| RSI(14) | 0.3069 | 0.3067 |
| SMA(5/10/20) | **0.3960** | **0.3802** |
| MACD(12/26/9) | 0.3020 | 0.2996 |

SMA 由 validation 選出後，在 203 天 held-out test 得到 accuracy=0.3399、macro F1=0.3415；
預先定義的 Volume baseline 在同一 test 為 accuracy=0.3251、macro F1=0.3253，SMA 分別提升
0.0148 與 0.0162（約多答對 3 天）。類別 F1：BEARISH 0.3433→0.4320、NEUTRAL
0.3066→0.2677、BULLISH 0.3259→0.3247；改善幾乎全由 BEARISH 帶來，並非三類全面進步。

結論：SMA 是這輪最值得繼續驗證的候選，但 test 增幅小、只有單一 seed，暫不取代正式 Volume
設定。下一步若要定案，應只針對 Volume vs SMA 跑 3～5 seeds，報告 mean±std；若仍穩定勝出，
再把 `chart.vision_inputs` 改成 `[candlestick, technical/sma]` 並重建正式 H_v/H_r。完整報告在
`data/outputs/experiments/auxiliary_vision/AAPL/report.json`；技術圖與 ViT 快取暫時保留以便續跑。

另外，本輪發現當時正式 `fusion.train` 會先用全日期 label 訓練 fusion，classifier 才做
70/15/15 切分。這個問題已由 `decisions.md #78` 的共用時間協議修正；本節數字是在修正前或
獨立實驗 protocol 下得到，不能直接當成新版正式 pipeline 的基準。

### 實驗有效性協議修正（2026-09-19）

依正式企劃書的「離線 belief construction → 凍結 Z_fused → PPO policy optimization」主線，
全流程改用單一時間序 split。AAPL 現有 1,381 個可用日預檢結果為 train 961、validation 202、
test 208、purged 10；purged 是切分邊界前五日標籤的 target date 已跨入下一區段。分位數門檻
只在 961 個 train 樣本估計。Fusion/class weight、decoder、PPO 均只使用指定訓練區段；三種
回測策略與 IG 使用相同 test 日期。事件 probe 改為只向前看的 expanding-window CV。

同時補上兩類非模型本身、但會污染結果的工程問題：(1) curriculum 不再把彼此不相鄰的
低波動日期串成同一個持倉 episode，而是從連續低波動區間向外擴張；(2) dataset、每日向量、
Z index、y_belief、PPO 與 decoder checkpoint 以 SHA-256 指紋串接，並採 ticker 專屬檔名。只要上游
重建或股票不同，下游會直接停止，不再靜默使用舊產物；fake fusion checkpoint 也與正式權重分開。
B 的 `--fake` 同樣改寫入 `{TICKER}_FAKE` namespace，不再覆蓋真實日期的向量。
Classifier 與 event-validation 每次執行都另存含 `run_id` 的歷史報告，同時更新 ticker
專屬 latest 檔，避免消融或重跑再次把正式結果無痕覆蓋。
LLM event extraction cache 也綁定 provider、model 與 dataset 指紋，切換模型或資料版本時
不會把舊模型答案誤當成新模型輸出續跑。
`fusion.train --apply_checkpoint` 會檢查 checkpoint metadata，拒絕 fake/舊協議權重，且若
目標 ticker 已在 `trained_tickers` 中會明確拒絕將它宣稱成 held-out 泛化測試。

OHLCV 正式預設亦改為 yfinance `auto_adjust=true`，讓五日標籤、PPO reward 與 backtest
納入拆股／現金股利調整，避免把除權息造成的機械性跳空當成預測錯誤。每次下載另寫
`data/raw/ohlcv/{TICKER}.meta.json` 保存來源、區間與 adjustment 設定；舊 raw CSV 必須重抓。

這是 protocol 變更，不是新模型結果。目前尚未重跑 B/C 的昂貴步驟，因此本文件前面所有
舊分類、decoder、PPO 與回測數字保留作歷史工程比較，不能與新版重訓結果直接比較。新版第一筆
正式結果必須重新建立 dataset/manifest，重產受 availability date 影響的 H_t/H_r，重訓 fusion、
decoder、PPO，再跑 test-only classifier/backtest/IG。

**怎麼判斷有沒有進步**：不是只看 accuracy 這一個數字，因為之前的模型可能只是學會「都猜 NEUTRAL」就拿到 0.459。更要看 BEARISH/BULLISH 的 f1 有沒有從 0 附近的塌縮狀態動起來，那才代表模型真的開始從新聞（或其他輸入）學到區分漲跌的訊號，不是準確率數字好看但其實沒學到東西。

## 三、診斷實驗細節：新聞有無 × 類別加權有無（2026-08-08）

對應 `docs/spec_c_accuracy_diagnostics.md`、`docs/tickets_c_accuracy_diagnostics.md`。動機：補齊五年新聞後準確率幾乎沒變，懷疑類別不平衡（NEUTRAL/BULLISH/BEARISH 比例約 44%/35%/21%，訓練/分類都沒加權）才是真正卡住結果的原因，所以設計這組 2×2 對照實驗，把「新聞有無」「加權有無」分開測。

實作方式：沒有實際刪改或備份任何向量檔案——直接在 `fusion/train.py` 加入 `--ablate_news`（讀進向量後在記憶體把新聞表徵歸零，圖表與檢索表徵不動，不寫回磁碟）跟 `--weighted`（訓練 loss 與最終分類器依類別頻率加權）兩個開關，預設都關閉。四種組合跑完後，用「不加任何開關重跑一次」當回歸測試，確認結果跟改動前的基準逐位數字一致（`accuracy=0.4648648648648649`，通過）。

| 組合 | accuracy | BEARISH（precision/recall/f1） | NEUTRAL（precision/recall/f1） | BULLISH（precision/recall/f1） | macro f1 |
|---|---|---|---|---|---|
| 有新聞 + 不加權（原本基準） | 0.4649 | 0.000 / 0.000 / 0.000 | 0.456 / 1.000 / 0.626 | 1.000 / 0.047 / 0.090 | 0.247 |
| 沒新聞 + 不加權 | 0.4649（跟上面逐位數字相同） | 0.000 / 0.000 / 0.000 | 0.456 / 1.000 / 0.626 | 0.800 / 0.063 / 0.116 | 0.247 |
| 有新聞 + 加權 | 0.3135 | 0.159 / 0.256 / 0.196 | 0.500 / 0.341 / 0.406 | 0.303 / 0.313 / 0.308 | 0.303 |
| 沒新聞 + 加權 | 0.2865 | 0.138 / 0.205 / 0.165 | 0.464 / 0.317 / 0.377 | 0.268 / 0.297 / 0.281 | 0.274 |

（原始檔案：`classification_report.json` 為有新聞不加權基準與回歸測試共用；`classification_report_noNews_noWeight.json`、`classification_report_news_weighted.json`、`classification_report_noNews_weighted.json` 分別對應另外三組。）

**三個問題的結論**：

1. **新聞有沒有幫助？** 有，但要在類別加權打開之後才看得出來。不加權時，有新聞跟沒新聞的結果逐位數字完全相同——不是差異很小，是完全沒有可量測的差異，代表在沒加權的訓練方式下，模型實質上沒有學到怎麼用新聞這個輸入。加權之後，有新聞在 accuracy、BEARISH f1、NEUTRAL f1、BULLISH f1 四個指標上全面贏過沒新聞（0.3135 vs 0.2865、0.196 vs 0.165、0.406 vs 0.377、0.308 vs 0.281），代表新聞本身是有真實貢獻的，只是先前被類別不平衡這個更大的問題完全蓋住。

2. **類別加權有沒有幫助？** 有，而且是目前為止影響最大的單一改動。不加權時模型幾乎塌縮成只猜 NEUTRAL（BEARISH 完全抓不到、BULLISH recall 只有個位數百分比）；加權之後，三個類別都有實質的 precision/recall，BEARISH 從完全學不到（f1=0）進步到 f1=0.196，macro f1 從 0.247 提升到 0.303。整體 accuracy 加權後從 0.46 掉到 0.31~0.29，這是預期中的正常現象，不是變差——不加權時的高 accuracy 是靠「安全牌全猜 NEUTRAL」灌水出來的假象。

3. **兩者有沒有交互作用？** 有：新聞的效果被類別不平衡完全遮蔽，只有先解決類別不平衡，新聞的貢獻才顯現得出來。這代表這兩個問題不是各自獨立、可以分開處理就好，之前只補新聞資料看不到效果，不是資料沒用，是被另一個更根本的訓練方式問題擋住了。

**下一步建議**：

- 類別加權應該直接採用為之後訓練的預設做法，不再只是診斷用的關閉選項——不加權時的準確率是假象，繼續用不加權的結果判斷好壞會誤導後續所有決策。
- 新聞資料證實有真實貢獻，照 `docs/spec_c_accuracy_diagnostics.md` 原本排好的優先順序，下一步是延伸 AAPL 的時間範圍到 2026-08（跟新聞歷史涵蓋範圍對齊），這是原本清單裡最便宜的資料擴充選項，現在有實測證據支持值得做。
- 誠實看目前的數字：即使加權 + 有新聞，macro f1 也只有 0.303，BEARISH 的 precision 僅 0.159，離「表現良好」還有很大差距，這一輪只是修掉了一個明顯的方法論問題（類別不平衡沒處理），不是宣告問題解決了。標籤定義、法說會逐字稿、更多股票這幾個先前決定暫緩的項目，目前還沒有必要提前處理，但也不代表現在的結果已經夠好，之後（例如訓練 epoch 數、learning rate 這類訓練細節，屬於使用者先前明確排除在這輪之外的「模型」範疇）大機率還需要投入才能真正達到堪用的準確率。

## 四、回測結果解讀（buy_and_hold / rule_based / ppo，2026-08-09 補記）

三組回測（`data/outputs/backtest/report_*.json`）之前只有算出數字、從未寫進文件解讀。回測區間 2021-02-01 ~ 2025-12-22（1230 天，舊範圍，尚未涵蓋延伸出來的 2026 年資料），交易成本 0.1%（`cost=0.001`）。

| 策略 | 累積報酬 | Sharpe Ratio | 最大回撤 |
|---|---|---|---|
| buy_and_hold（單純持有 AAPL，基準） | 102.8% | 0.660 | 33.4% |
| rule_based（用分類訊號進出，無 RL 對照組） | 51.1% | 0.669 | 18.0% |
| ppo（RL agent） | 14.6% | 0.660 | 6.0% |

**看到的現象**：從 buy_and_hold 到 rule_based 到 ppo，策略越「聰明」，最大回撤越小（33.4% → 18.0% → 6.0%），但累積報酬也跟著大幅下降（102.8% → 51.1% → 14.6%）；三者的 Sharpe Ratio 幾乎相等（都在 0.66 左右）。也就是說，PPO 不是在風險調整後的報酬上贏過單純持有 AAPL，而是把整條策略往「更保守」的方向移動——用大幅犧牲報酬去換取大幅降低的最大回撤，Sharpe 幾乎沒變代表這只是在風險-報酬曲線上換了一個位置，不是真的找到更好的位置。

**這不是意外，是獎勵函數設計出來的結果**：`rl/env.py` 目前的獎勵函數是 `reward = 當期報酬 - λ_vol×波動度 - λ_mdd×回撤`，預設 `lambda_vol=0.1`、`lambda_mdd=0.1`。PPO agent 訓練時本來就是在被懲罰波動與回撤，會學出「保守」的策略是獎勵函數設計的直接後果，不是模型自己發現了什麼特別的市場規律。如果想要 PPO 更積極追求報酬，需要調低 `lambda_vol`/`lambda_mdd` 重新訓練，這本身是一組還沒做過的實驗。

**限制與 caveat（誠實記錄，不要誤讀這組結果）**：

- 只有一支股票（AAPL）、只有一段固定的 5 年區間（剛好涵蓋 2022 熊市與之後的多頭），結果可能是這段特定期間的產物，換一段時間或換一支股票不保證同樣的現象會重現。
- 目前的回測是單一次結果，沒有做多組隨機種子或多時間窗口的重複實驗，不能排除是雜訊。
- `rule_based` 策略用的分類訊號，來自準確率還不高的分類器（見上方第二節，macro f1 僅 0.3 左右），這組對照組本身品質有限，不是一個「已經很準」的基準。
- 回測範圍是舊的 1230 天，還沒涵蓋這次延伸出來到 2026-08 的資料，之後重跑 pipeline 後這三個數字都需要更新。

**結論**：目前不能說「加了 RL 有比較好」，只能說「加了 RL 讓策略變得比較保守、風險比較低」，兩者是不同的主張。是否要往更積極的方向調整獎勵函數、要不要多做幾組重複實驗，是這組結果之後需要進一步決定的問題，暫不在這輪處理。

## 五、事件抽取實驗紀錄（2026-08-10）

對應 `docs/decisions.md` #32。動機：事件抽取（`event_extraction.py`）之前完全沒有 ground truth，不知道現有純關鍵字方法準不準；補上 150 天（實際 149 天）分層抽樣的 ground truth（`data/labels/event_ground_truth/AAPL.json`，LLM 輔助標記，這一輪由 Claude 在對話中直接標記）後，第一次量出真實 P/R/F1。

| 版本 | precision | recall | f1 | tp | fp | fn | 改了什麼 |
|---|---|---|---|---|---|---|---|
| baseline（改進前） | 0.073 | 0.987 | 0.135 | 75 | 959 | 1 | 無，就是原本的 `EVENT_KEYWORDS` + 純關鍵字比對 |
| + 公司相關性檢查 | 0.078 | 0.987 | 0.145 | 75 | 886 | 1 | 新聞標題要出現公司名稱/ticker 才進關鍵字比對 |
| + 過濾陳舊 filing/transcript | 0.135 | 0.868 | 0.233 | 66 | 424 | 10 | 只有 chunk 的 `filing_date`/`event_date` 等於當天才算，跳過被沿用當背景脈絡的舊資料 |
| + 關鍵字調整（最終版） | 0.162 | 0.868 | 0.273 | 66 | 341 | 10 | 拿掉過寬鬆的單字（`PRODUCT_LAUNCH` 的 announce/release、`GUIDANCE` 的單字 raise/lower/cut、`MANAGEMENT_CHANGE` 的單獨 ceo/cfo），改用更完整的片語 |

**根因排查，跟原本猜測不同**：一開始以為主因是「文章跟公司無關但被誤判」（例如迪士尼文章被標成 AAPL 事件），但實際加上公司相關性檢查後 precision 幾乎沒動（0.073→0.078）。真正的主因是 `build_dataset.py` 的 `collect_filing_chunks()`/`collect_transcript_chunks()`：設計成「沿用最新一份財報/法說會到下一份發布為止」，同一份文件的標準樣板文字（股利政策、訴訟揭露、併購風險因素等段落，任何 10-Q/10-K 都會固定寫）被連續很多天重複讀到，每次都被誤判成「當天發生」的新事件。加上日期比對（只有 chunk 真正對應當天才算）之後，fp 從 886 大幅降到 424，才是這次改善的主要來源。

**誠實記錄目前限制**：precision 絕對值仍偏低（0.162），代表關鍵字方法還是抓到不少雜訊，`EARNINGS`/`GUIDANCE`/`PRODUCT_LAUNCH` 仍有較多誤判，純關鍵字比對難以分辨「討論、回顧某件事」跟「當天真的宣布」的語意差別；ground truth 裡唯一一筆 `MANAGEMENT_CHANGE`（新聞用「Apple Hires Former BMW Executive To Lead Electric Car Efforts」這種不含 "appoint"/"resign" 字眼的報導方式）目前仍抓不到，反映純關鍵字方法的天花板。這是 `event_extraction.py` 檔頭本來就寫的「後續可改成 LLM-based 抽取」的實際動機來源，不在這一輪範圍內處理。

**事件驗證頭（2026-08-10，decisions.md #34）**：拿 149 天 ground truth 對應的 Z_fused（768 維）當輸入，訓練 multi-label `LogisticRegression`（`class_weight="balanced"`），5-fold cross-validation 評估。`MA`、`MANAGEMENT_CHANGE` 各只有 1 個正樣本，標注「資料量不足，不評估」，其餘 5 類結果：

| 事件類別 | 正樣本天數 | precision | recall | f1 |
|---|---|---|---|---|
| EARNINGS | 10 | 0.135 | 0.700 | 0.226 |
| PRODUCT_LAUNCH | 15 | 0.111 | 0.400 | 0.174 |
| LAWSUIT | 36 | 0.280 | 0.389 | 0.326 |
| GUIDANCE | 7 | 0.115 | 0.857 | 0.203 |
| DIVIDEND | 6 | 0.100 | 0.833 | 0.179 |

micro-avg（5 類合計）：precision=0.147、recall=0.514、f1=0.229，跟事件抽取直接讀文字的 f1=0.273 同一個量級，代表 Z_fused 確實吸收了一定程度的事件相關資訊，沒有在 B/C 的編碼融合過程中把這類資訊完全丟失。**誠實記錄**：樣本數小（149 天）、輸入維度高（768），屬於高維度低樣本的困難設定，數字噪音大，是初步結果不是定論；`class_weight="balanced"` 讓 recall 明顯偏高、precision 偏低，跟未用平衡權重的事件抽取數字不是完全同條件的對照。
