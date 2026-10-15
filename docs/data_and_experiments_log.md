# A 資料來源紀錄 + B/C 實驗結果紀錄

這份文件持續更新，不是一次性報告。新增資料來源、重跑一次分類驗證，都回來這裡補一行。

## 一、資料來源（截至 2026-09-30）

| 資料類型 | 來源 | 抓取腳本 | 目前涵蓋範圍 | 狀態 |
|---|---|---|---|---|
| 股價 OHLCV | yfinance | `fetch_ohlcv.py` | AAPL／NVDA／MSFT／JPM 2021-01-04 ~ 2026-08-07（各 1405 筆） | AAPL 完整多模態；其餘三檔於 2026-09-24 補齊供多股票 ViT 實驗 |
| K 線圖 | 用 OHLCV 自畫（mplfinance，20 日窗口） | `chart_generator.py` | AAPL／NVDA／MSFT／JPM 各 1386 張（頭尾幾天因窗口不足缺圖） | 四檔皆完成 |
| 成交量圖 | 用 OHLCV Volume 自畫（20 日窗口） | `chart_generator.py` | AAPL／NVDA／MSFT／JPM 各 1386 張 | 四檔皆完成；作為雙圖 ViT 的第二張輸入 |
| 新聞（近期） | yfinance | `fetch_news.py` | 僅最新 8~10 則，歷史價值低 | 可用，範圍小 |
| 新聞（歷史回補） | Alpaca News API（來源 Benzinga，官方 API 非爬蟲） | `fetch_news_alpaca.py` | AAPL raw 共 16,318 則 / 1,497 個日期檔；dataset 期間 1,381 天皆有當日或 7 日內回補新聞 | 2026-09-30 全 pipeline 重抓並驗證 |
| 財報 10-K/10-Q/8-K | SEC EDGAR（官方 API，原生 requests） | `fetch_filings.py` | AAPL 共 71 份：10-K 5、10-Q 18、8-K 48；2021-01-05 ~ 2026-07-31 | 1,381 個 dataset 日皆有可用背景 chunk |
| 法說會逐字稿 | Alpha Vantage API（官方，取代原本從未成功端到端跑過的 foolcalls） | `fetch_transcripts.py` | AAPL 共 22 份，2021-04-28 ~ 2026-07-30 | 已流入 dataset 與 H_r；dataset 前 60 天尚無歷史逐字稿，其餘 1,321 天有背景 chunk |

**已放棄的路徑**（保留記錄，程式碼已刪除）：FNSPID 爬蟲工具與 FNSPID 現成 HuggingFace dataset，兩者都驗證不可行（自動化偵測擋爬蟲、頁面改版、dataset 全文欄位是空的）。詳見 `docs/decisions.md` #26。

**已解決的落差**：OHLCV/K 線圖原本只到 2025-12-30、新聞已回補到 2026-08-08 的範圍不對齊問題，已於 2026-08-09 延伸 OHLCV/圖表到 2026-08-07 解決（`configs/config.yaml` 的 `date_range.end` 同步更新）。整條 pipeline（`build_dataset.py` → `generate_vectors.py` → `fusion.train` → `classifier.py`）已針對延伸後的新範圍重跑過（decoder 訓練用的 Z_fused/y_belief 交集樣本數 1381 天即為此次重跑後的結果，見 `docs/decisions.md` decoder 相關條目）。

## 二、B/C 實驗結果紀錄（分類驗證準確率）

每次重跑 `classifier.py` 都在這裡加一行，`detail_file` 存一份對應時間點的 `classification_report.json` 備份，方便回頭比對。

| 日期 | 資料版本/這次改了什麼 | n_train/val/test | accuracy | BEARISH f1 | NEUTRAL f1 | BULLISH f1 | 備註 / detail 檔案 |
|---|---|---|---|---|---|---|---|
| 2026-08-08（之前） | 新聞僅 yfinance 近期，歷史期間新聞幾乎全空（7 天回補窗口內大多抓不到） | 861 / 184 / 185 | 0.459 | 0.000 | 0.621 | 0.090 | 模型幾乎塌縮成只猜 NEUTRAL，準確率只比「每天都猜 NEUTRAL」（44%）高一點；`data/outputs/metrics/classification_report_baseline_no_news.json` |
| 2026-08-08（之後） | + Alpaca 歷史新聞回補（重跑 build_dataset → generate_vectors → fusion.train → classifier，OHLCV/圖表範圍不變，仍是 2021~2025） | 861 / 184 / 185 | 0.465 | 0.000 | 0.626 | 0.116 | 準確率只多 0.9 個百分點，在 185 筆測試集裡等於大約多猜對 1 天，不算有意義的差異；BEARISH 完全沒動（還是 0）；BULLISH f1 從 0.090 到 0.116，support 只有 64，等於大概多對 1 題，也在雜訊範圍內；NEUTRAL 幾乎沒變、recall 仍是 1.0，模型還是幾乎每天都猜 NEUTRAL。結論：新聞資料補齊本身沒有明顯改善這個下游分類結果，瓶頸應該在別的地方（label 定義、類別不平衡、訓練集只有 861 筆、ViT 沒微調等），不是新聞量不夠。`data/outputs/metrics/classification_report.json` |
| 2026-08-08（診斷實驗，見 spec_c_accuracy_diagnostics.md） | 新聞有無 × 類別加權有無，四種組合對照（詳見下方獨立小節） | 861 / 184 / 185 | 0.465 / 0.465 / 0.314 / 0.286 | 0.000 / 0.000 / 0.196 / 0.165 | 0.626 / 0.626 / 0.406 / 0.377 | 0.116 / 0.116 / 0.308 / 0.281 | 四個數字依序對應：有新聞不加權／沒新聞不加權／有新聞加權／沒新聞加權。結論：不加權時新聞完全沒影響（結果逐位數字相同）；加權後模型不再塌縮成只猜 NEUTRAL，且加權狀態下「有新聞」四個指標全面優於「沒新聞」——新聞資料是有真實貢獻的，只是先前被類別不平衡蓋住看不出來。整體 accuracy 加權後變低是預期中的正常現象（模型不再靠猜多數類別灌水），不是變差，判讀要看 macro f1 與少數類別表現 |
| 2026-08-09（延伸日期範圍 + RAG query 修正，加權預設開啟） | OHLCV/K線圖延伸到 2026-08-07（資料集實際到 2026-07-31，最後幾天缺 label）；RAG query 從單純平均改成正規化+alpha 加權（alpha=0.5） | 966 / 207 / 208 | 0.3029 | 0.1875 | 0.3497 | 0.3212 | macro f1≈0.286，跟前一輪「有新聞+加權」（macro f1=0.303）相比持平、略降，沒有明顯進步。**重要限制**：這次同時改了兩件事（延伸範圍 + RAG 修正），且測試集因為時間序切分跟著往後移動（這次測試期間 2025-10-02~2026-07-31，跟前一輪 2025-03-31~2025-12-22 不同），無法從這次比較單獨歸因是哪個改動造成差異，也可能只是新測試期間本身難度不同。要乾淨驗證 RAG 修正的效果，需要同一個日期範圍、只切換 RAG 新舊版本的對照實驗，目前尚未做。`classification_report_2026-08_extended_range_rag_fix.json` |
| 2026-08-10（漲跌標籤改為分位數門檻，見 decisions.md #30） | 只改標籤定義（固定±2% → 分位數1/3門檻），日期範圍、RAG、加權都跟上一輪相同 | 966 / 207 / 208（跟上一輪完全相同的切分與測試期間，可乾淨對照） | 0.3269 | 0.2689 | 0.3313 | 0.3731 | macro f1 從 0.286 提升到 0.324（+0.038），是這一路診斷下來第一次有乾淨、無混雜因素的正向結果——這次 n_train/val/test 筆數與測試期間跟上一輪完全一樣，只有標籤定義變了，可以放心把差異歸因到標籤改動本身。BEARISH f1 進步最多（0.188→0.269），BULLISH 也進步（0.321→0.373），NEUTRAL 略降（0.350→0.331）。測試集類別分布也從原本 BEARISH 明顯偏少（原本 support 39~48）變成三類接近平均（support 65/67/76）。誠實記錄：macro f1=0.324 仍不算「表現良好」，只是目前為止最好的一次。過程中曾因為 sandbox 執行 `build_dataset.py` 中途被 timeout 打斷，導致新舊標籤混雜跑出一次不可信的結果（accuracy=0.3221），已作廢重跑並逐日核對 208 天全部一致才採信這次結果。`classification_report_2026-08_quantile_labels.json` |
| 2026-09-19（雙圖 ViT：K 線 + 成交量） | 同一個 frozen ViT 以 batch 一次編碼兩張獨立圖片，H_v `[197,768]` → `[2,197,768]`；fusion 加入圖別 slot embedding。兩組皆使用 1,381 天、seed=42、weighted、3 epochs | 966 / 207 / 208 | 0.3413（固定相同 H_t/H_r 的單圖對照 0.3173，+0.0240） | 0.3036（0.2810） | 0.3432（0.3353） | 0.3704（0.3281） | macro f1 0.3390，受控單圖對照 0.3148，+0.0242；三類 f1 全數上升。另有導入前完整 pipeline 基準 0.3060，但因 H_v 也參與 RAG query，主要結論採固定 H_t/H_r 的受控結果。只跑一個 seed，尚未做顯著性檢驗。報告：`data/outputs/experiments/dual_image_vit/AAPL/` |
| 2026-09-30（第一筆 strict temporal baseline） | 修正 Fusion test-label leakage；標籤門檻只用 Train；邊界各留 5 日 gap；Fusion 只用 Train 更新並用 Validation 選 epoch；classifier 用 Validation 選 C=10，Train+Validation refit 後只評估一次 Test | 966 / 202 / 203 | **0.3547** | 0.3401 | 0.3443 | 0.3796 | **macro F1=0.3547**，confusion matrix=`[[25,18,18],[28,21,15],[33,19,26]]`。Validation 候選 C 的 macro F1 為 0.3062/0.3120/0.3037/**0.3667**。這是第一筆可作正式基準的無洩漏數字；舊雙圖 0.3390 使用不同標籤門檻與切分，不能把 +0.0157 直接歸因為模型變強。`data/outputs/metrics/classification_report.json` |

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
70/15/15 切分，會讓 test label 間接洩漏進 Z_fused。上述新實驗已避開此問題；過往正式
classifier 數字仍可作同流程工程比較，但不應再稱為嚴格 held-out 成效。此問題已於
2026-09-30 正式修正（decisions.md #85）：現在先建立共用時間切分 manifest，邊界各留 5 日
gap，Fusion 只用 Train label 更新並由 Validation 選 checkpoint，classifier 沿用相同日期。
修正後的完整 pipeline 已於 2026-09-30 全量重跑；新結果見本節後方 strict baseline 與
「六、strict temporal 全 pipeline 重跑分析」。較早的舊數字仍只作歷史工程比較，不冒充
新流程的 held-out 成效。

實作期間的操作紀錄：短測 `fusion.train --fake` 一度沿用舊檔名而覆寫
`data/outputs/checkpoints/fusion.pt`。發現後已立即把 fake 權重分離為 `fusion_fake.pt`，
並以 2026-09-19 的真實 AAPL checkpoint `fusion_AAPL.pt` 回復 `fusion.pt`。原本
2026-09-22 的 `fusion.pt` 沒有獨立備份，無法逐 bit 復原；新流程本來就必須重訓，
且現在 fake 模式已固定只寫 `fusion_fake.pt`，不會再覆蓋正式權重。

2026-09-30 已實際重跑 AAPL `build_dataset.py`：共 1,381 天，切分為
Train 966／5 日 gap／Validation 202／5 日 gap／Test 203。Train 期間三類標籤
剛好各 322 天；只依 Train 報酬計算的門檻為 bearish `< -0.0106526`、
bullish `> 0.0186615`。Validation 類別計數為 75/58/69（bearish/neutral/bullish），
Test 為 61/64/78。現有 AAPL vectors 也是 1,381 天，與 manifest 日期零缺漏、
零多餘，因 B 不讀 label/prices，這次不重跑 Encoder。嚴格 Fusion 正式訓練與
當時嚴格 Fusion 與下游新指標尚未執行；現已完成並記錄如下，全程不把舊指標當成新流程結果。

2026-09-30 嚴格 Fusion 訓練也已完成（CPU、seed=42、weighted、3 epochs）。
Train loss 1.3206→1.1861→1.1649，Validation loss 1.4402→1.3747→1.2835；
Validation macro F1 三個 epoch 皆為 0.1487，因此依 tie-break 的最低 validation loss
選 epoch 3。這個 0.1487 符合臨時 Fusion 訓練頭幾乎只猜 NEUTRAL 的類別坍縮，
是需要保留的診斷警訊；但凍結 Z_fused 後另訓練的 Logistic Regression 在 Test
三類 F1 皆為 0.34～0.38，沒有跟著坍縮。Checkpoint SHA256 為
`418693e147aae5a4bba432de152fb3121a06574855737901f187ac0d1fe118ad`；1,381 個 Z_fused
皆為 finite float32 `[768]`，與 manifest 日期、split 計數完全一致。

同版本事件驗證頭在共用 manifest 下只有 103/23/22 個 Train/Validation/Test
ground-truth 日。只有 PRODUCT_LAUNCH 與 LAWSUIT 在三個區間都有足夠正樣本：
Test F1 分別為 0.000 與 0.222，合併 micro F1=0.111（TP=1, FP=12, FN=4）。
其餘五類因 Validation/Test 無正樣本或 Train 過少而不評分。這與舊的隨機
5-fold CV micro F1=0.229 不是同一評估協定，不可直接宣稱表現下降；目前更明確的
結論是事件 ground truth 對嚴格時間評估來說過少且分布不均。

### ViT-only 分類頭與部分 Fine-tune（2026-09-24）

新增 `experiments/vision/vision_classifier.py` 與
`experiments/vision/experiment_vit_adaptation.py`，隔離 H_t、H_r 和 fusion，只比較
雙圖 ViT 本身。每一天的 K 線與 Volume 圖共用同一個 `google/vit-base-patch16-224`，各取一個
CLS token 串成 1,536 維，再接 `LayerNorm + Dropout(0.1) + Linear(1536,3)` 分類頭。資料共
1,381 天；依時間切成 966 train／202 validation／203 test，兩個切分邊界各保留 5 個交易日
gap。BEARISH／NEUTRAL／BULLISH 的 1/3、2/3 分位數門檻只用 train 的未來五日報酬計算，
避免使用 validation/test 的報酬分布。固定 seed=42、balanced cross-entropy、batch=8、分類頭
learning rate=3e-4、ViT learning rate=3e-6、weight decay=1e-3，最多 10 epochs、patience=3。

| 策略 | 可訓練參數 | 最佳 epoch | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|---:|
| Frozen ViT，只訓練分類頭 | 7,683 | 6 | 0.4406 | 0.4095 |
| 解凍最後 1 block | 7,097,091 | 3 | 0.4109 | 0.3648 |
| 解凍最後 2 blocks | 14,184,963 | 3 | 0.4059 | 0.3607 |
| 解凍最後 4 blocks | 28,360,707 | 8 | **0.4653** | **0.4309** |

Validation 選出 last4 後才開啟 held-out test，並在同一次 test 評估預先定義的 Frozen baseline。

| Test 指標 | Frozen baseline | Validation 勝者 last4 | last4 - frozen |
|---|---:|---:|---:|
| Accuracy | **0.3596** | 0.3448 | -0.0148 |
| Balanced accuracy | **0.3788** | 0.3659 | -0.0129 |
| Macro F1 | **0.3377** | 0.3210 | -0.0166 |
| Log loss（越低越好） | **1.1908** | 1.2733 | +0.0825 |

Frozen 的 test 類別 F1 為 BEARISH 0.4444／NEUTRAL 0.3186／BULLISH 0.2500；last4 為
0.4393／0.3382／0.1856。last4 雖然在 validation 高出 0.0214 macro F1，到了真正未看過的
test 卻低 0.0166，且 log loss 也較差；改善沒有跨期間重現，合理解讀是對 validation period
過度適配，不能說 Fine-tune 已優於 frozen ViT。**決定：不替換正式 frozen ViT**。若繼續這條線，
先增加多股票資料或測 LoRA，再以多 seed／walk-forward 驗證；不針對已開啟的這份 test 繼續調參。

完整報告：`data/outputs/experiments/vit_adaptation/AAPL/run/report.json`；checkpoint 同目錄下的
`checkpoints/`。60 天 smoke test 只用來確認 frozen／last1 梯度、early stopping、checkpoint
與 test gate 可執行，不列入模型成效。注意本輪仍只有單一 seed，且是 ViT-only probe，尚未
重新產生 H_v 或放回 fusion，所以不能直接和 Z_fused classifier 數字做同任務比較。

### 多股票 ViT：AAPL＋NVDA 訓練、MSFT backbone holdout（2026-09-24）

新增 NVDA、MSFT 各 1,405 筆 OHLCV、1,386 張 K 線、1,386 張 Volume 圖與 1,381 筆每日
dataset。三檔取 1,381 個共同日期；AAPL＋NVDA 各自使用分類頭、共同更新同一個 ViT backbone，
避免每檔股票不同的 train-only 分位數門檻被迫共用輸出邊界。MSFT 圖片與標籤完全不參與
backbone 訓練或策略選擇；選出 backbone 後將其凍結，只用 MSFT 過去期間訓練一個新的診斷
probe head，再測 MSFT 未來期間。切分日期與單股版相同且邊界各留 5 日 gap：AAPL＋NVDA
train 1,932 筆（2021-02-01~2024-12-02）、validation 404 筆（2024-12-10~2025-10-01）；
MSFT probe train/validation 各 966/202 筆，held-out test 203 筆（2025-10-09~2026-07-31）。

| Backbone 策略 | 最佳 epoch | Pooled validation macro F1 | AAPL macro F1 | NVDA macro F1 | 兩檔平均 |
|---|---:|---:|---:|---:|---:|
| Frozen，只訓練兩個股票 head | 6 | **0.3757** | 0.3761 | **0.3339** | **0.3550** |
| Last4，共用 ViT 最後四層 | 7 | 0.3610 | **0.4478** | 0.2236 | 0.3357 |

last4 對 AAPL 比 Frozen 增加 0.0717，卻讓 NVDA 下降 0.1102，顯示它不是學到更通用的金融圖表
表示，而是更偏向其中一檔股票。執行當下以 pooled macro F1 選模，Frozen 勝出；事後再用更嚴格
的「每檔 macro F1 先算、再平均」核對，仍是 Frozen 0.3550 > last4 0.3357，結論不變。程式後續
已把多股票選模預設改成兩檔平均，避免 pooled 指標掩蓋單一股票退化。

Frozen backbone 在 MSFT 的新 probe head 以 validation 選到 epoch 4（validation macro F1
=0.4091），最後 MSFT held-out test accuracy=0.3202、balanced accuracy=0.3292、macro F1
=0.3066、log loss=1.1129；類別 F1 為 BEARISH 0.3506／NEUTRAL 0.3394／BULLISH 0.2299。
因 last4 未通過 AAPL＋NVDA validation 選擇，所以沒有再用 MSFT test 挑救 last4，保持 test gate。

正式結論：**只新增 NVDA 還不足以讓 last4 Fine-tune 穩定勝過 Frozen；正式 pipeline 繼續使用
Frozen ViT。** 多股票的價值在這輪主要是揭露單股 AAPL 看不出的過度專化。下一次若繼續，應
增加更多產業的股票或測 LoRA，並改用新的 walk-forward folds／新 held-out ticker；不能再用
已開啟的這段 MSFT test 調參。完整報告：
`data/outputs/experiments/vit_adaptation/AAPL_NVDA_holdout_MSFT/run/report.json`。

另保留一個工程 pilot：AAPL＋NVDA 共用同一分類頭、MSFT 完全零樣本分類時，Frozen/last4
validation macro F1 為 0.3498/0.3141，選出的 Frozen 在 MSFT test 為 0.2870。由於專案既定
架構要求每檔股票分開輸出頭，此 pilot 不作正式結論，存於同目錄的
`zero_shot_shared_head_run/`。

### ViT LoRA 正式實驗（2026-09-24）

為避免繼續用已開啟的 MSFT test 調參，本輪新增 JPM 作為新的跨股票 holdout；已建立 1,405 筆
OHLCV、1,386 張 K 線、1,386 張 Volume 圖與 1,381 筆每日 dataset。正式設計以
AAPL＋NVDA＋MSFT 訓練／validation，共用 ViT backbone、各自使用股票分類頭；JPM 完全不參與
backbone 訓練或 LoRA 策略選擇。選定策略後才凍結 backbone，以 JPM 過去資料訓練新 probe head，
最後開啟 JPM future test。

新增的策略語法為 `lora_rN`。預先固定比較 `frozen`、`lora_r4`、`lora_r8`；LoRA 只掛在
ViT 最後四個 transformer blocks（8～11）的 attention `query`／`value`，alpha=2×rank、
dropout=0.1，ViT/LoRA learning rate=1e-4。三股票分類頭也列入後，可訓練參數為：Frozen
23,049、LoRA r=4 為 72,201、LoRA r=8 為 121,353；相較先前 Last4 約 2,837 萬個可訓練
參數小很多。

已用共同前 60 個日期、1 epoch 做工程 smoke test，確認兩個 LoRA rank 都能完成反向傳播、
checkpoint 儲存／載入、validation 選模及 JPM probe。這次 validation 只有 12 筆、future test
只有 4 筆，**數字不得當成模型效果或 LoRA 勝出的證據**；報告只留作工程稽核：
`data/outputs/experiments/vit_adaptation/AAPL_NVDA_MSFT_holdout_JPM/smoke/report.json`。

正式實驗使用 1,381 個共同日期、seed=42、batch=8、最多 10 epochs、patience=3。AAPL／NVDA／
MSFT 合計 2,898 筆 train、606 筆 validation；策略選擇採「三檔各自 macro F1 再平均」，避免
pooled 指標掩蓋單檔退化。

| 策略 | 最佳 epoch | Pooled val macro F1 | AAPL | NVDA | MSFT | 三檔平均（選模指標） |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 1 | 0.3474 | **0.3459** | 0.2034 | 0.3339 | 0.2944 |
| LoRA r=4 | 4 | **0.3954** | 0.3414 | **0.3293** | **0.4099** | **0.3602** |
| LoRA r=8 | 1 | 0.3419 | 0.3615 | 0.3108 | 0.2955 | 0.3226 |

Validation 依預定規則選出 LoRA r=4，三檔平均比 Frozen 高 0.0658。接著才以凍結的 backbone
分別訓練 JPM probe head（兩者皆在 epoch 2 選中），並開啟同一段 203 天 JPM future test：

| JPM test 指標 | Frozen | LoRA r=4 | 差值（r4 - Frozen） |
|---|---:|---:|---:|
| Accuracy | 0.3547（72/203） | **0.3793（77/203）** | +0.0246 |
| Balanced accuracy | 0.3530 | **0.3696** | +0.0166 |
| Macro F1 | 0.3209 | **0.3458** | +0.0249 |
| Log loss（越低越好） | 1.1014 | **1.0894** | -0.0120 |

類別效果不平均：BEARISH F1 由 0.1075 大幅升至 0.3407、NEUTRAL 由 0.4500 升至 0.4945，
但 BULLISH 由 0.4052 降至 0.2022（recall 0.4844→0.1406）。因此正式結論是：**LoRA r=4
已通過這一輪 validation 與新股票 held-out test，證據比 Last4 強，列為下一個送進 fusion 的
ViT 候選；但它不是三類全面改善，而且目前只有一個 seed，不直接替換正式 Frozen ViT。**
下一步應固定 r=4 與現有超參數，不再利用已開啟的 JPM test 調整；先做 3～5 seeds／新
walk-forward fold，確認平均與變異，再把 adapter 載入正式 `vision_encoder.py`，重建 H_v 後做
Frozen vs LoRA 的嚴格 fusion 對照。正式報告：
`data/outputs/experiments/vit_adaptation/AAPL_NVDA_MSFT_holdout_JPM/run/report.json`；完整 terminal
log：`data/outputs/experiments/vit_adaptation/lora_AAPL_NVDA_MSFT_holdout_JPM.log`。

#### LoRA r4 五 seed 穩定性驗證（2026-09-29）

新增 `experiments/vision/run_vit_multiseed.py`，固定比較 Frozen 與 LoRA r4、固定既有超參數，使用 seeds
40／41／42／43／44。這一輪只彙整 AAPL／NVDA／MSFT validation 的每檔 macro F1 平均，
不傳 `--evaluate-test`，因此不會再次讀取 JPM future test。
`experiments/vision/experiment_vit_adaptation.py` 同時新增
`--run-name`，讓每個 seed 寫入獨立目錄，不覆蓋上方已開啟 test 的 `run/report.json`。

已用 seeds 940／941、共同前 60 日期、1 epoch 完成短版工程 smoke test：確認 seed 目錄隔離、
test 維持 locked、完成的 seed 可在中斷續跑時自動跳過、`--summarize-only` 可重建 mean±std。
由於 validation 僅 12 筆，smoke 數字不作模型效果解讀。正式五 seed 已在 MPS 完成，訓練計時
合計約 2 小時 18 分；每份 report 均確認 `test_was_opened=false`。

| Seed | Frozen 三檔平均 macro F1 | LoRA r4 | r4 - Frozen | 勝者 |
|---:|---:|---:|---:|---|
| 40 | **0.3674** | 0.3630 | -0.0044 | Frozen |
| 41 | 0.3240 | **0.3867** | +0.0627 | r4 |
| 42 | 0.2944 | **0.3602** | +0.0658 | r4 |
| 43 | **0.3797** | 0.3530 | -0.0266 | Frozen |
| 44 | 0.2984 | **0.3680** | +0.0696 | r4 |
| **Mean ± std** | **0.3328 ± 0.0392** | **0.3662 ± 0.0127** | **+0.0334 ± 0.0454** | r4 3/5 |

| 股票 | Frozen mean ± std | LoRA r4 mean ± std | 平均差值 | r4 勝出 seeds |
|---|---:|---:|---:|---:|
| AAPL | 0.3479 ± 0.0648 | **0.3875 ± 0.0258** | +0.0396 | 3/5 |
| NVDA | **0.3097 ± 0.0669** | 0.3096 ± 0.0360 | -0.0001 | 2/5 |
| MSFT | 0.3407 ± 0.0436 | **0.4015 ± 0.0150** | +0.0607 | 5/5 |

整體上 r4 平均高 0.0334，且自身跨 seed 標準差比 Frozen 小（0.0127 vs 0.0392），顯示它有
較穩定的正向訊號；但只勝出 3/5 seeds，平均差值小於差值標準差，差值的近似 95% 信賴區間
為 -0.0230～0.0898，包含 0。改善也不平均：MSFT 5/5 穩定受益、AAPL 3/5，NVDA 平均幾乎
完全打平且只勝 2/5。

因此本輪**沒有通過預先設定的 4/5 穩定門檻**。結論由「r4 可直接進 fusion 候選」下修為：
**r4 有平均提升且較穩，但證據不足以替換 Frozen 或投入昂貴的完整 fusion 重建。** 現階段不再
增加相同資料上的 seeds，也不使用已開啟的 JPM test 調參；下一個更有資訊量的實驗應增加不同
產業的訓練股票與至少兩檔新 holdout，或增加新的 walk-forward 時段，確認 r4 的提升不是由 MSFT
單一股票主導。正式彙整：
`data/outputs/experiments/vit_adaptation/AAPL_NVDA_MSFT_holdout_JPM/multiseed_summary/report.json`；
完整 log：`data/outputs/experiments/vit_adaptation/vit_lora_multiseed.log`。

#### Frozen ViT-B/16 vs Frozen MaxViT-T（2026-09-30）

新增 `experiments/vision/experiment_maxvit_comparison.py`，只比較視覺 backbone，不修改正式
encoder、RAG 或 Fusion。AAPL／NVDA／MSFT 使用 1,381 個共同日期，每檔 966 個 Train、5 日 gap、
202 個 Validation；三檔合計 2,898／606 筆。每支股票的漲跌分位數門檻只由自己的 Train 計算，
Test 全程鎖住。兩張輸入維持 K 線＋Volume；ViT-B/16 使用兩張圖的 CLS（合計 1,536 維），
MaxViT-T 使用官方分類頭最後 class Linear 前的兩個 512 維表示（合計 1,024 維）。兩個 backbone
完全 frozen，各自只訓練相同型態的 `LayerNorm + Dropout(0.1) + Linear` 股票分類頭；固定
seeds 40～44、balanced cross-entropy、head LR=3e-4、weight decay=1e-3、最多 10 epochs、
patience=3。主要指標為三檔各自 Validation macro F1 的平均。
這是獨立 runner，backbone 特徵先固定快取後才重設 seed 初始化分類頭；因此這裡的 ViT
0.3447 不與舊 LoRA runner 的 Frozen 0.3328 跨報告直比，模型結論只採本輪同 seed、同 runner
內的配對差值。

| Backbone | 三檔平均 macro F1 | AAPL | NVDA | MSFT | Pooled macro F1 | Pooled log loss |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ViT-B/16 | **0.3447 ± 0.0206** | **0.3712** | **0.3010** | 0.3618 | **0.3845** | 1.1530 |
| Frozen MaxViT-T | 0.3374 ± 0.0205 | 0.3393 | 0.2939 | **0.3790** | 0.3712 | **1.1339** |

逐 seed 的 MaxViT−ViT 三檔平均差值為 -0.0184／-0.0375／+0.0186／-0.0068／+0.0077；平均
-0.0073 ± 0.0220，MaxViT 只勝 2/5，近似 95% CI 為 -0.0266～+0.0120。未通過預先設定的
「平均至少 +0.02 且勝 4/5 seeds」門檻。類別效果也不平均：ViT→MaxViT 的平均 F1 為
BEARISH 0.3932→0.4321、NEUTRAL 0.4565→0.4532、BULLISH 0.3037→0.2281；MaxViT 對下跌類
較好，但明顯犧牲上漲類。工程面上 MaxViT-T 參數較少（30.9M vs 86.4M），在本機 MPS 對
3,504 筆雙圖樣本的特徵抽取速度幾乎相同（87.7 vs 89.2 秒），沒有形成足以抵銷準確度退化的
明顯速度優勢。

**結論：不替換正式 ViT，也不投入 MaxViT 的 H_v／RAG／Fusion 介面改造。** 這個結果只否定
「直接換成 frozen MaxViT-T」；沒有測 MaxViT fine-tune、金融圖表領域預訓練或 MaxViT-B，不能
外推成整個 MaxViT 家族必然無效。現階段更值得把資源放在跨產業資料、walk-forward 與既有
LoRA 候選的泛化驗證。完整報告：
`data/outputs/experiments/maxvit_comparison/AAPL_NVDA_MSFT/run/report.json`。

#### Frozen ViT-B/16 vs Frozen MaxViT-S/224（2026-10-01，正式五 seed 完成）

為釐清 MaxViT-T 未勝是否只是容量不足，新增
`experiments/vision/experiment_maxvit_s_comparison.py`。候選使用官方 TensorFlow 權重的 PyTorch
移植版 `maxvit_small_tf_224.in1k`（`timm`），約 68.2M backbone 參數、輸入 224×224、每張圖
輸出 768 維 global-pooled penultimate feature；兩張圖串成 1,536 維，與現有 ViT 分類頭輸入
寬度一致。這只是 frozen backbone 容量消融，不修改正式 encoder 或 Fusion。

實驗沿用 AAPL／NVDA／MSFT 共同日期、Train-only 標籤、5 日 gap、股票分頭、seeds 40～44、
相同分類頭與超參數；既有 ViT 特徵直接重用，Test 全程 locked。預先門檻維持 MaxViT-S−ViT
的 mean-ticker Validation macro F1 至少 +0.02，且五個 seeds 至少勝出四個。若未通過就停止
MaxViT 放大路線，不再測 Base／Large；若通過，才另行設計 token-level H_v 與 Fusion 對照。
工程 Smoke 已用最近 60 個共同日期、seed 940、1 epoch 完成。每檔只有 42 筆 Train 與 4 筆
Validation，Test 維持 locked。MaxViT-S 成功載入 68,158,956 個參數並輸出每張 768 維特徵；
138 筆雙圖樣本抽取耗時 6.88 秒（20.05 samples/s），代表模型、預處理、特徵快取與分類頭管線
均可正常執行。

| Smoke backbone | 三檔平均 macro F1 | AAPL | NVDA | MSFT | Pooled macro F1 | Pooled log loss |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ViT-B/16 | **0.2963** | 0.0000 | 0.2222 | **0.6667** | **0.3718** | **1.0535** |
| Frozen MaxViT-S | 0.1481 | 0.0000 | 0.2222 | 0.2222 | 0.2815 | 1.2281 |

Smoke 的 MaxViT-S−ViT 差值為 -0.1481，但**不能作為模型優劣結論**：Validation 合計僅
12 筆（每檔 4 筆），AAPL 與 MSFT 的 Validation 完全沒有 BEARISH，且只跑一個 seed、一次 epoch；
因此 macro F1 對單筆預測非常敏感。此階段只判定工程驗證通過，不套用正式 +0.02／4-of-5
門檻。Smoke 報告：
`data/outputs/experiments/maxvit_s_comparison/AAPL_NVDA_MSFT/smoke_60/report.json`。

正式實驗使用完整 1,381 個共同日期：每檔 966 Train、5 日 gap、202 Validation，三檔合計
2,898／606 筆；seeds 40～44、最多 10 epochs、patience=3，Test 全程 locked。五個 seeds 的
主要結果如下：

| Frozen backbone | 三檔平均 macro F1 | AAPL | NVDA | MSFT | Pooled macro F1 | Pooled log loss |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | **0.3447 ± 0.0206** | **0.3712** | **0.3010** | **0.3618** | **0.3845** | **1.1530** |
| MaxViT-S | 0.3130 ± 0.0189 | 0.3165 | 0.2728 | 0.3496 | 0.3554 | 1.1660 |

逐 seed 的 MaxViT-S−ViT 差值為 -0.0287／-0.0445／-0.0001／-0.0426／-0.0426；平均
-0.0317 ± 0.0188，勝出 0/5。探索性 paired t-test 為 `t(4)=-3.771, p=0.0196`，95% CI
`[-0.0550, -0.0084]`；seed 數只有五個，統計結果仍應保守解讀，但方向與預設 gate 完全一致。
分股票差值為 AAPL -0.0547、NVDA -0.0282、MSFT -0.0122，三檔均下降；分種類 F1 差值為
BEARISH -0.0070、NEUTRAL -0.0414、BULLISH -0.0390，也沒有任何類別取得平均改善。MaxViT-S
特徵抽取耗時 138.6 秒，較 ViT 的 89.2 秒慢約 55%，參數量則為 68.2M vs 86.4M。

**結論：MaxViT-S 未通過「至少 +0.02 且勝 4/5 seeds」門檻，正式 pipeline 維持 Frozen
ViT-B/16。依預先決策停止 MaxViT 放大路線，不再測 Base／Large，也不投入 MaxViT 的 H_v／
RAG／Fusion 介面改造。** 這項結論只適用目前 frozen ImageNet backbone＋pooled feature＋
小型分類頭，不代表 MaxViT 經金融圖表預訓練或 fine-tune 後一定無效。正式報告：
`data/outputs/experiments/maxvit_s_comparison/AAPL_NVDA_MSFT/run/report.json`。

自動格式化版本：[MaxViT-S 正式報告](../data/outputs/experiments/maxvit_s_comparison/AAPL_NVDA_MSFT/run/report.md)。
圖表統一存於該實驗同目錄的 `figures/`，可直接在報告中閱讀。

#### Vision 實驗自動視覺化與 Markdown 報告（2026-10-01）

新增 `experiments/vision/render_experiment_report.py`，可辨識 backbone comparison、ViT adaptation、
LoRA multi-seed summary 與 auxiliary vision 四種 JSON 格式。每份 `report.json` 旁會產生可直接閱讀的
`report.md`，並依資料內容輸出總覽、訓練曲線、列正規化混淆矩陣與 Test 對照圖；混淆矩陣若跨
seed 彙整，會先各自正規化再取平均，不把同一 Validation 樣本的重複預測誤當成獨立樣本。

已批次處理 `data/outputs/experiments/` 下全部 25 份既有 report，共產生 25 份 Markdown 與 76 張
PNG，無跳過項目；總索引位於 `data/outputs/experiments/experiment_reports_index.md`。四個主要
runner 已在保存 JSON 後自動呼叫產圖／排版流程；若視覺化失敗，只會顯示 warning，不會刪除或
覆蓋已完成的訓練 JSON。所有自動報告與圖表統一存於各實驗的 `data/outputs/`，`docs/` 只保留
紀錄與連結；`data/outputs/` 依專案規則不進 Git，因此產物可由相同指令隨時重建。
瀏覽入口：[全部實驗報告索引](../data/outputs/experiments/experiment_reports_index.md)。

#### Frozen ViT-B/16 vs Frozen DINOv3 ViT-B/16（2026-10-01）

新增 `experiments/vision/experiment_dinov3_comparison.py`，沿用上一節已驗證的 frozen
backbone runner，固定 AAPL／NVDA／MSFT 共同日期、Train-only 分位數標籤、5 日 gap、股票分頭、
seeds 40～44 與 Validation-only 選模。比較組為現有 `google/vit-base-patch16-224` 與
`facebook/dinov3-vitb16-pretrain-lvd1689m`；兩者各取兩張圖的 CLS 串接，只訓練相同分類頭，
不開 Test、不修改正式 encoder、RAG 或 Fusion。既有 ViT 特徵快取會直接重用，避免重複抽取；
DINOv3 權重則需先接受官方 gated license 並登入 Hugging Face。

工程 Smoke 先確認 gated 權重與 1,536 維雙圖特徵可正常產生；正式實驗再使用完整 1,381 個
共同日期，每檔 966 Train／202 Validation、5 日 gap，三檔合計 2,898／606 筆。固定 seeds
40～44、head LR=3e-4、weight decay=1e-3、dropout=0.1、最多 10 epochs、patience=3；Test
全程 locked。預先通過門檻為 DINOv3−ViT 的 mean-ticker Validation macro F1 至少 +0.02，
且五個 seeds 至少勝出四個。

| Frozen backbone | 三檔平均 macro F1 | AAPL | NVDA | MSFT | Pooled macro F1 | Pooled log loss |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | **0.3447 ± 0.0206** | **0.3712** | 0.3010 | **0.3618** | **0.3845** | **1.1530** |
| DINOv3 ViT-B/16 | 0.3263 ± 0.0122 | 0.3469 | **0.3063** | 0.3258 | 0.3574 | 1.1673 |

逐 seed 的 DINOv3−ViT 差值為 -0.0269／-0.0178／+0.0019／-0.0319／-0.0170；平均
-0.0184 ± 0.0129，只勝 1/5 seeds，未通過預設門檻。探索性的 paired t-test 為
`t(4)=-3.178, p=0.0336`，95% CI `[-0.0344, -0.0023]`；樣本只有五個 seeds，統計結果應
保守解讀，但方向與預設 gate 一致。逐類別 F1 也全面下降：BEARISH 0.3932→0.3641、NEUTRAL
0.4565→0.4480、BULLISH 0.3037→0.2601。DINOv3 只在 NVDA 小幅增加 +0.0053，在 AAPL／MSFT
分別下降 -0.0243／-0.0360。特徵抽取耗時 109.2 秒，亦慢於 ViT 的 89.2 秒。

**結論：在 frozen CLS＋線性分類頭協定下，DINOv3 沒有優於現有 ViT，不替換正式 encoder，
也不投入 DINOv3 的 H_v／RAG／Fusion 重建。** 這個結果不能外推到 DINOv3 LoRA、fine-tune、
patch/register token pooling 或金融圖表預訓練；若國科會計畫需要，可把 DINOv3 列為已評估的
自監督 backbone baseline，而不是預定的最終模型。正式報告：
`data/outputs/experiments/dinov3_comparison/AAPL_NVDA_MSFT/run/report.json`；Smoke 報告：
`data/outputs/experiments/dinov3_comparison/AAPL_NVDA_MSFT/smoke_60/report.json`。

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

## 六、strict temporal 全 pipeline 重跑分析（2026-09-30）

執行 `python scripts/run_pipeline.py --ticker AAPL`，A→B→C 全部跑完。這次 pipeline 實際包含：
OHLCV、雙圖、財報、新聞、法說會、dataset、H_v/H_t/H_r、keyword 事件抽取、Fusion、
分類 probe、預設 PPO 10,000 steps，以及三種 Test 回測。它**不包含**事件驗證頭、生成式
decoder、decoder evaluation、curriculum PPO 或 Integrated Gradients；這些模組磁碟上即使有舊報告，
也不能算成本次 pipeline 的新結果。

### 資料與中間產物完整性

- OHLCV 1,405 日（2021-01-04～2026-08-07）；K 線與 Volume 圖各 1,386 張；最後可產生
  5 日未來標籤的 dataset 共 1,381 日（2021-02-01～2026-07-31）。
- 共用切分為 Train 966／gap 5／Validation 202／gap 5／Test 203；標籤門檻只由 Train
  計算，bearish `< -1.0653%`、bullish `> +1.8662%`。Train 三類剛好各 322 天。
- 1,381 日皆有新聞與 filing 背景；法說會自 2021-04-28 起可用，共覆蓋 1,321 日。
- 1,381 組向量全部存在且無 NaN/Inf：`H_v=[2,197,768]`、`H_t=[512,768]`、
  `H_r=[3,512,768]`；`Z_fused=[1381,768]` 也全部 finite，日期與 split 零缺漏。
- Fusion checkpoint SHA256 為
  `418693e147aae5a4bba432de152fb3121a06574855737901f187ac0d1fe118ad`，與
  `AAPL_index.meta.json` 記錄一致。這也與同日上午單獨重跑 strict Fusion 的 checkpoint
  完全相同，訓練結果可重現。
- 本次未帶 `generate_vectors --balance_sources`。4,143 個 RAG top-3 名額中，逐字稿占
  3,191（77.0%）、財報占 952（23.0%）；這是已知的來源偏斜，但先前 quota 實驗沒有改善
  分類表現，因此本次保留正式預設設定。

另發現兩個**不影響本次下游計算、但會影響結果追溯**的產物問題：

1. `data/processed/dataset/AAPL_manifest.json` 仍是舊版 2026-09-19 artifact，記載舊門檻與
   961/202/208 切分；現在真正被下游讀取的是
   `data/processed/temporal_splits/AAPL.json`（966/202/203）。舊 manifest 應後續刪除或改由
   `build_dataset.py` 同步覆寫，避免人工查看時混淆。
2. 本次 keyword 事件抽取寫到 `event_extraction_report.json`；帶 ticker 的
   `event_extraction_report_AAPL.json` 仍是 2026-09-19 舊檔，不能拿錯檔。

### Fusion 與市場方向分類

Fusion 三個 epoch 的 Train loss 為 1.3206→1.1861→1.1649，Validation loss 為
1.4402→1.3747→1.2835；Validation macro F1 三次都等於 0.1487。依 Validation loss
tie-break 選 epoch 3。0.1487 正好對應 Validation 全猜 NEUTRAL 的 macro F1，表示 Fusion
訓練用的臨時分類 head 已類別坍縮；loss 下降不代表分類能力同步改善。

凍結 `Z_fused` 後的 balanced Logistic Regression 由 Validation 選出 `C=10`，再用
Train+Validation refit，Test 結果如下：

| 指標 | 結果 |
|---|---:|
| Accuracy | 0.3547（72/203） |
| Macro F1 | 0.3547 |
| BEARISH F1 | 0.3401 |
| NEUTRAL F1 | 0.3443 |
| BULLISH F1 | 0.3796 |

分類器沒有坍縮，Test 預測 BEARISH/NEUTRAL/BULLISH 為 86/58/59 天；但訊號仍弱。
它的 accuracy 低於「全部猜 Test 多數類 BULLISH」的 78/203=0.3842，不過 macro F1
高於全猜 BULLISH 的約 0.185，代表模型的價值主要是三類較平均，而不是提高總命中率。
這次數字與同日上午的第一輪 strict baseline 完全一致，所以全量重建向量沒有帶來額外提升，
也沒有破壞可重現性。

### Keyword 事件抽取

1,381 天中有 1,287 天至少抽到一個事件，共 7,748 個事件 mention；平均每個有事件的日期
約 6.0 個，密度過高，本身就是過度觸發的警訊。149 天 ground truth 上：precision=0.1610、
recall=0.8684、F1=0.2716（TP=66、FP=344、FN=10）。與舊報告 F1=0.2733 幾乎相同；加入
逐字稿事件後多出 3 個 FP，沒有增加 TP。結論仍是「高 recall、低 precision」的規則式
baseline，不適合把抽到的每個事件都當成可靠事實。

嚴格時間切分的事件驗證頭**不在本次 pipeline 內**；目前磁碟上的 F1=0.111 是本次 pipeline
之前單獨執行的結果，不能說是這輪自動重跑所得。

### PPO 與 held-out Test 回測

PPO 僅使用 966 個 Train 日，預設 10,000 timesteps、`curriculum=false`、`ent_coef=0`。
三策略都只在相同 Test 203 日（2025-10-09～2026-07-31）回測，交易成本 0.1%。

| 策略 | 累積報酬 | Sharpe | 最大回撤 | 解讀 |
|---|---:|---:|---:|---|
| Buy & Hold | **+19.314%** | **0.984** | 13.823% | 本輪明確最佳報酬基準 |
| Rule-based | -3.120% | -0.161 | 11.887% | 訊號毛報酬 +3.294%，但頻繁換倉後轉負 |
| PPO | +0.135% | 0.981 | **0.098%** | 幾乎空手，不是有效超額報酬 |

Rule-based 平均持倉 43.35%，共 91 天改變部位，累計 turnover=64；不計成本時仍只有
+3.294%，加上每次 0.1% 成本後變成 -3.120%。這表示目前分類訊號不只沒有擊敗 Buy & Hold，
而且對交易成本非常敏感。

PPO 的持倉介於 0.6634%～0.6679%，平均 0.6657%、標準差只有 0.00094 個百分點，實質上是
固定持有約 0.67% AAPL、其餘現金。它的 Sharpe 0.981 看起來接近 Buy & Hold，只是把同一段
資產報酬縮小約 150 倍；總報酬只有 0.135%。因此不能用低回撤或 Sharpe 宣稱 PPO 有效，
這是一個明確的 near-cash policy collapse。

### 本輪結論與下一步優先順序

1. **資料管線與 leakage 修正成功**：日期、向量、checkpoint、split 全部一致，Test 沒有進
   Fusion/PPO 訓練；這次結果可作正式、可重現的 strict baseline。
2. **預測能力尚未達到可交易程度**：分類 macro F1=0.355 只有弱訊號，rule-based 在成本後
   虧損，PPO 則退化成幾乎空手；目前不能宣稱模型勝過簡單持有。
3. 下一個最有資訊量的工作不是直接增加 PPO timesteps，而是先修 PPO 評估/訓練診斷：報告
   action mean/std、turnover、現金比例並加入「低於最小曝險」的 collapse 警告；再用
   Validation 選 reward penalty、`ent_coef` 與 curriculum，Test 保持鎖住。
4. Fusion 臨時 head 全猜 NEUTRAL，也應先做 epoch/LR/head 的 Validation-only 診斷；只有
   Validation macro F1 穩定高於簡單基準後，再值得重跑昂貴的下游。
5. 清理兩個 provenance 問題：淘汰舊 `AAPL_manifest.json`，並統一事件報告檔名。這不會改變
   本次數字，但可防止之後讀錯報告。

## 2026-10-01：ViT 跨股票共用分類頭實驗實作（待真實資料執行）

依使用者確認，新增獨立 `experiments/vision/run_vit_cross_stock.py`，比較 Frozen 與 LoRA r4。
目標是從雙圖學到可泛化到未見股票的漲跌訊號，**不改正式 Fusion 或舊三分類實驗**。

- 股票配置：`configs/experiments/vit_cross_stock.json`，手選跨產業大型美股 30 train／10 validation／10 test，非 point-in-time S&P 500 全集，有股票池選擇與存活者偏差。
- 日期：2016–2021 train、2022–2023 validation、2024–2025 test。按實際未來第 5 交易日排除跨期間 label；未見股票不參與任何日期的梯度更新，也不重訓新 probe head。
- 圖形：過去 20 交易日 K 線與成交量雙圖，224×224，所有股票共用 ViT 與二分類頭。
- Label：調整後 close 的未來 5 日報酬正負（DOWN／UP），精確零排除；yfinance 明確指定 `auto_adjust=True, actions=True` 並保存來源政策。不與舊三分類 F1 直接比較。
- 訓練：Frozen 僅更新頭；LoRA 最後 4 blocks query／value，r4／alpha8／dropout0.1。預設 seeds40、41、42；主指標 `validation_unseen.mean_ticker_macro_f1`。數值 Logistic Regression 與訓練集多數類別作基準。
- 防呆：資料／模型／訓練程式與配置雜湊鎖定，產圖及 Frozen 特徵可沿用，完整策略跳過；中斷策略從頭重跑。預設 Test locked，完整訓練後另加 `--evaluate-test` 才推論，開始前保存 `test_opened.json`，中斷不抹掉已開啟狀態。
- 報告：每策略完成更新 JSON、MD、訓練曲線、二分類混淆矩陣及產業／年份圖，全部位於 `data/outputs/experiments/vit_cross_stock/`。共用 renderer 已支援新 schema，docs 不另存圖。

驗證：`python -m unittest discover -s tests -v` 共 17 項通過；新增 10 項含合成 OHLCV＋本機小型隨機 ViT 的離線端到端測試，檢查 Frozen／LoRA 更新範圍、同 seed 頭初始化、切分、續跑、報告、設定不符拒絕與 Test 開啟狀態。**尚未下載這輪 50 檔真實資料，也未啟動正式大模型訓練；目前沒有改善準確度的結論。**

執行：[封存紀錄](../experiments/vision/README.md)；規劃：[實驗計畫](archive/experiment_plan_vit_cross_stock.md)。

### 同日：真實 Smoke 的 Tee 相容性修正

使用者已完成 7 檔 Smoke 的資料準備（manifest `7b0683e642155d3f`），但載入 Google ViT 時因 `Tee` 缺少 `isatty()` 中斷。Transformers 在載入含預訓練分類頭的 checkpoint 至 `AutoModel` 時會產生 UNEXPECTED／MISSING 報告，該報告檢查 `sys.stdout.isatty()`；原本完全匹配的小型 ViT 測試未涵蓋這條路徑。

修正：Tee 新增 `isatty()`，其餘 `encoding`／`fileno` 等介面委派到原始 console；新增分類 checkpoint 載入回歸測試及 stream 介面測試。完整測試共 **19 項通過**，並另以使用者既有快取的真實 Google ViT 驗證載入，沒有執行長時間訓練或開啟 Test。

原 Smoke 的資料配置雜湊未改，既有 OHLCV／雙圖可沿用；runner 原始碼雜湊已變，為保留舊失敗 run 與設定鎖定，下一次命令使用 `--smoke --run-name smoke_tee_fix`。原失敗紀錄不刪除。目前仍待使用者完成真實 Frozen／LoRA Smoke，不能作效果結論。

### 同日：真實 Smoke 成功完成，正式資料仍待準備

使用者回報完成，已核對 `vit_cross_stock/smoke/7b0683e642155d3f/smoke_tee_fix/report.json`、MD、terminal.log 與 checkpoint。Frozen／LoRA r4 各完成 seed40／1 epoch，`complete=true`；`test_was_opened=false` 且沒有 `test_opened.json`，沒有 Test 指標。4 張圖（overview、training_curves、confusion_matrices、group_breakdown）皆已輸出。

樣本：Train 24、Validation seen 24、Validation unseen 16；保留 Test seen 24／unseen 16，未評估。Train 股票 AAPL／MSFT／NVDA，未見 Validation 為 ADBE／WFC；每檔 unseen 只有 8 筆。

| 方法 | 未見 Validation 平均每股 Macro-F1 | Pooled accuracy | 正確筆數 |
|---|---:|---:|---:|
| Frozen | 0.5152 | 0.5625 | 9/16 |
| LoRA r4 | 0.4182 | 0.5000 | 8/16 |
| Numeric Logistic Regression | 0.5913 | 0.6250 | 10/16 |
| Majority | 0.3590 | 0.5625 | 9/16 |

LoRA−Frozen 平均每股 Macro-F1 = −0.0970。Frozen 與 LoRA 可訓練參數為 6,146／55,298，策略時間為 2.78／2.89 秒（Smoke 實測，不能直接當成全量時間預估）。**結論僅為程式／資料流程可跑通；24 筆訓練、1 epoch、1 seed，不能用來判定 LoRA 無效或選定正式勝者。** 舊三分類報告亦不可直接比較。

另排查 log 的 `The y_prob values do not sum to one`：已保存 Frozen head／特徵的只讀重算顯示 float32 機率轉 float64 後，最大總和偏差 5.96e-8，大於本機 sklearn 的 float64 檢查容差約 1.49e-8。評估函式現先檢查 [0,1] 範圍與總和接近 1（容差 1e-6），只校正微小誤差；真正不合法機率會拒絕，不單純關閉警告。Argmax／accuracy／Macro-F1 不因此改變，原 Smoke 報告與權重保持原樣。

新增浮點捨入及不合法機率的回歸測試，完整測試共 21 項通過。下一步由使用者先跑 `--prepare-only` 產生正式 50 檔資料，完成後检查各股票日期／類別／樣本覆蓋，再安排計時與正式多 seed 訓練。此輪沒有替使用者啟動長時間工作或開啟 Test。

實際報告：[Smoke report.md](../data/outputs/experiments/vit_cross_stock/smoke/7b0683e642155d3f/smoke_tee_fix/report.md)。

### 2026-10-02：跨股票正式資料準備完成

使用者已完成 `--prepare-only`。正式 dataset fingerprint 為 `14c913bcec83cfba`；已核對
manifest、`rows.jsonl` 與圖片完整性，Train／Validation 的檔案雜湊、日期邊界及二元 label
均能通過載入驗證。Test 僅存在 manifest 中，`test_was_evaluated=false`，本次沒有載入 Test
圖片進行模型推論或計算指標。

| Split | 股票數 | 樣本數 | DOWN | UP |
|---|---:|---:|---:|---:|
| Train（2016–2021，已見股票） | 30 | 45,095 | 19,378（43.0%） | 25,717（57.0%） |
| Validation seen（2022–2023） | 30 | 14,872 | 7,115（47.8%） | 7,757（52.2%） |
| Validation unseen（2022–2023） | 10 | 4,955 | 2,345（47.3%） | 2,610（52.7%） |
| Test seen（2024–2025，封存） | 30 | 14,893 | 未評估 | 未評估 |
| Test unseen（2024–2025，封存） | 10 | 4,962 | 未評估 | 未評估 |

共 50 檔、84,777 個樣本，每筆兩張 224×224 PNG，合計 169,554 張。dataset 約占 1.1 GB，
其中 charts 約 993 MB；空間主要來自圖片數量，而非單張尺寸。正式訓練尚未開始，因此目前
仍沒有 Frozen 與 LoRA 在大樣本上的效果結論。下一步先跑單一 seed、1 epoch 的 timing run，
確認全量特徵抽取與 LoRA 訓練的實際耗時，再決定是否直接執行 3 seeds、最多 10 epochs。

### 同日：全量單 seed／1 epoch timing 結果

使用者完成 `--seeds 40 --epochs 1 --run-name timing`。run 狀態完整，Test 顯示 LOCKED，
`test_was_opened=false`。主指標仍為完全未參與梯度更新的 10 檔 Validation 股票之
`mean_ticker_macro_f1`。

| 方法 | Unseen mean-ticker Macro-F1 | Pooled accuracy | Balanced accuracy | ROC-AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| Frozen | 0.4471 | 0.4892 | 0.5049 | 0.5026 | 0.7152 |
| LoRA r4 | **0.4776** | 0.4928 | 0.5025 | 0.5027 | 0.7085 |
| Numeric Logistic | **0.4803** | **0.5068** | 0.4964 | 0.4882 | **0.6935** |
| Majority UP | 0.3448 | **0.5267** | 0.5000 | 0.5000 | 17.0580 |

LoRA − Frozen 的 mean-ticker Macro-F1 為 +0.0304，且 10/10 檔 unseen 股票的 Macro-F1
都提高；2022、2023 及所有十個產業點也都高於 Frozen。不過這個改善主要來自預測類別較
平均：Frozen 在 unseen 只預測 20.6% UP，LoRA 為 31.9%；LoRA 的 DOWN／UP F1 分別為
0.5606／0.4004。實際標籤則有 52.7% UP。

更重要的是，LoRA 對真實 DOWN 與 UP 樣本給出的平均 UP 機率只有 0.466691 與 0.467279，
幾乎沒有分離；ROC-AUC=0.5027、balanced accuracy=0.5025 也都接近隨機。故目前只能說 LoRA
修正了 Frozen 過度預測 DOWN 的傾向，**不能宣稱已學會可泛化的漲跌訊號**。它在主指標上
仍略低於只使用 10 個數值特徵的 Logistic Regression（差 0.0027），accuracy 也低於多數類
基準；一個 seed 無法估計訓練波動。

耗時方面，Frozen 共 1,967.5 秒（32 分 48 秒），其中約 32 分 39 秒是首次抽取 Train／
Validation 特徵，真正一個 head epoch 約 8.9 秒；380 MB 特徵快取已保存，後續同模型資料可
直接沿用。LoRA 一個 epoch 為 3,168.6 秒（52 分 49 秒）。若直接跑 3 seeds、最多 10 epochs，
LoRA 最壞約需 26.4 小時，因此不建議現在立刻開 Test 或直接跑滿。下一個較省成本的判別實驗
是 seed40 跑 3 epochs，觀察 unseen ROC-AUC／balanced accuracy 是否離開 0.5，以及主指標
是否穩定超過 Numeric Logistic；若只改變分類門檻而沒有排序能力，再增加 seeds 的價值有限。

實際報告：[Timing report.md](../data/outputs/experiments/vit_cross_stock/runs/14c913bcec83cfba/timing/report.md)。

### 2026-10-02：全量 seed40／3 epochs pilot 結果

使用者完成 `--seeds 40 --epochs 3 --run-name pilot3`；報告完整且 Test 維持 LOCKED。
Frozen 特徵快取成功沿用，3 epochs 共 27.2 秒；LoRA 3 epochs 共 11,240.0 秒（約 3 小時
7 分 20 秒）。

| 方法 | Best epoch | Unseen mean-ticker Macro-F1 | Accuracy | Balanced accuracy | ROC-AUC | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 3 | **0.4922** | 0.4971 | **0.4951** | **0.4909** | **0.7095** |
| LoRA r4 | 3 | 0.4908 | **0.4979** | 0.4944 | 0.4844 | 0.7136 |
| Numeric Logistic | — | 0.4803 | **0.5068** | **0.4964** | 0.4882 | **0.6935** |

依既定主指標，Frozen 比 LoRA 高 0.0014，LoRA 僅在 4/10 檔 unseen 股票勝過 Frozen。
兩種 ViT 的 Macro-F1 都略高於 Numeric Logistic，但 accuracy、balanced accuracy、ROC-AUC
與 log loss 沒有同步改善，因此不能把這個差距解讀為方向預測能力。

學習曲線顯示：Frozen unseen mean-ticker Macro-F1 為 0.4471 → 0.4825 → 0.4922；LoRA 為
0.4776 → 0.4782 → 0.4908。相對地，Frozen unseen ROC-AUC 為 0.5026 → 0.4854 → 0.4909；
LoRA 為 0.5027 → 0.4866 → 0.4844。LoRA unseen log loss 也從第 1 epoch 的 0.7085 惡化到
0.7136。也就是 Macro-F1 上升主要來自預測比例逐漸接近兩類各半，而非正負樣本的分數分離
變好。第 3 epoch 的平均 UP 機率甚至在真實 DOWN 樣本較高（Frozen 0.5110 vs 0.5084；
LoRA 0.5180 vs 0.5127），與低於 0.5 的 AUC 一致。

結論：現有 Google ViT 圖像表示加上 LoRA r4，尚無可泛化漲跌訊號證據；目前不應開 Test，
也不建議直接執行昂貴的 3-seed／10-epoch run。主指標需搭配 ROC-AUC、balanced accuracy 與
log loss 的 sanity gate，否則隨機但預測比例平衡的分類器也可能獲得接近 0.5 的 Macro-F1。

另確認 terminal 最後 `[report] markdown`／`[report] figures` 各出現兩次不是重複訓練：最後
一個 strategy 完成時會先呼叫 renderer 產生增量報告，訓練迴圈結束後又呼叫一次產生最終報告。
兩次覆寫相同輸出，數值與 checkpoint 沒有重複計算；屬於無害但可清理的重複渲染。

實際報告：[Pilot3 report.md](../data/outputs/experiments/vit_cross_stock/runs/14c913bcec83cfba/pilot3/report.md)。

### 2026-10-02：ViT 實驗程式封存

完成結果整理後，已移除過往 ViT 訓練 runner、實驗模型、跨股票資料準備程式與專用測試，避免已結束的探索程式持續混在正式專案入口中。正式 `module_b_encoder` 沒有變更；原始 `report.json`、`report.md`、圖表、history 與 predictions 仍保留在 `data/outputs/experiments/`。

統一結論、各實驗可比性與結果來源見 [ViT 實驗總結](vit_experiments_summary_2026-10-02.md)。保留的 `render_experiment_report.py`、`visualize_comparison_report.py` 與 `cross_stock_report.py` 可由既有 JSON 重建報告。若未來要跑新的 label／horizon／SMA／walk-forward 實驗，需另外建立新版 runner，不覆寫歷史輸出。

### 2026-10-10：Frozen 窗口、三分類與科技巨頭實驗封存

2026-10-05 完成的三個 Frozen ViT 實驗（窗口 × 單／雙圖消融、Train-only q33／q67 三分類、8 檔科技巨頭 single／Volume／SMA）已結束，數字與判讀見 [ViT 實驗總結](vit_experiments_summary_2026-10-02.md) §7–§9。依使用者決定，後續 ViT 工作收斂回 AAPL／NVDA／MSFT，這三個實驗只留紀錄。

已移除 `run_frozen_window_ablation.py`、`run_vit_frozen_three_class.py`、`run_vit_tech_giants.py`；三份 `configs/experiments/*.json`、對應的 `*_report.py` 與 `data/outputs/experiments/` 內的報告、predictions、feature cache 保留。`experiments/vision/` 的三份 README 合併為單一[封存紀錄](../experiments/vision/README.md)，跨股票實驗計畫移到 `docs/archive/`。

### 2026-10-10：像素畫法、K20＋K60 與 K20＋SMA 對照（AAPL／NVDA／MSFT）

三個 Frozen ViT 輸入對照實驗，皆未改動正式 pipeline，Test 未開啟；完整數字見 [ViT 實驗總結](vit_experiments_summary_2026-10-02.md) §10–§11、決議 #100～#102。

| 實驗 | 設定 | mean-ticker Validation Macro-F1 | 結論 |
|---|---|---|---|
| 像素畫法 | mplfinance／黑底像素／白底像素，單圖 | 0.3351／0.2814／0.3132 | 像素畫法 0/5 seeds 勝出，畫法不變 |
| 像素畫法 | 同上，K 線＋Volume 雙圖 | 0.3390／0.2781／0.3112 | 同上 |
| K20＋K60 | single_k20／single_k60／K20＋Volume／K20＋K60 | 0.3410／0.3140／0.3758／0.3478 | K20＋K60 對 K20＋Volume 0/5 seeds 勝出，雙圖不變 |
| K20＋SMA | 同上樣本加入 K20＋SMA | 0.3627 | 勝單圖 5/5（+0.0217），對 K20＋Volume 2/5（−0.0131），視為打平，雙圖不變 |

K20＋K60 實驗因 60 日窗口剔除 Train 最前面 40 天，同一個 K20＋Volume 設定由 0.3390 變為 0.3758；每檔約 200 筆 Validation 下，0.03 左右的差距不應視為穩定差異。Runner `run_pixel_render_comparison.py`、`run_k20_k60_comparison.py` 與 `run_k20_sma_comparison.py` 暫時保留在 `experiments/vision/`。

### 2026-10-11：全域 vs 滾動 250 日標籤門檻對照（AAPL／NVDA／MSFT）

只改標籤定義，輸入沿用既有 Frozen ViT CLS 快取、20 日數值特徵與正式 Z_fused；Test 未開啟。滾動門檻為「第 t 天只用 t−5 以前已實現的最近 250 筆 5 日報酬」，N 事前指定。完整說明見決議 #103。

| 輸入 | ROC-AUC（global → rolling） | Balanced acc（global → rolling） |
|---|---|---|
| ViT 單張 K20 | 0.5339 → 0.5400 | 0.3681 → 0.3594 |
| ViT K20＋Volume | 0.5625 → 0.5685 | 0.3748 → 0.3680 |
| Numeric Logistic（20 日） | 0.4875 → 0.4832 | 0.3235 → 0.3427 |
| Z_fused＋LR（AAPL／NVDA） | 0.5068 → 0.5162 | 0.3383 → 0.3278 |

兩種標籤在 Validation 有 95% 相同，所有差異都在 ±0.01 左右，小於已知樣本擾動；正式標籤定義不變。滾動門檻的主要作用是修正類別比例，受益最大的 NVDA Test 因維持鎖定而未驗證。

同日加跑 N=120（決議 #104）：與全域標籤相同的比例降到 85%～93%，Validation 類別比例更平均，但下門檻在部分期間轉為正值。ROC-AUC（global → rolling120）為 ViT 單圖 0.5339→0.5271、K20＋Volume 0.5625→0.5497、Numeric Logistic 0.4875→0.4798、Z_fused＋LR 0.5068→0.5450；個股方向相反（K20＋Volume：AAPL 0.5718→0.6349、MSFT 0.5902→0.4953）。沒有一致改善，不採用；N 已非事前指定，不再用這段 Validation 挑 N。

### 2026-10-11：K 線影線加粗、換色對照（AAPL／NVDA／MSFT）

沿用像素畫法實驗的樣本與 Frozen CLS 快取，只新增兩種影線樣式；Test 未開啟。詳見決議 #105、[ViT 實驗總結](vit_experiments_summary_2026-10-02.md) §10.4。

| 畫法 | 單圖 Macro-F1 | 雙圖 Macro-F1 |
|---|---:|---:|
| mplfinance（正式） | 0.3351 | 0.3390 |
| 白底像素，影線 1 像素 | 0.3132 | 0.3112 |
| 影線 3 像素、淺色 | 0.3095 | 0.3109 |
| 最高～最低同寬淺色棒 | 0.3349 | 0.3116 |

同寬色棒在單圖上與 mplfinance 打平，雙圖仍較差；3 像素影線無效。正式畫圖方式不變。

2026-10-12 補跑直接在正式 mplfinance 圖上加粗影線（決議 #106）：影線 3 倍線寬的單／雙圖 Macro-F1 為 0.3159／0.3299，影線與實體同寬的淺色版為 0.2956／0.3214，都低於正式圖的 0.3351／0.3390（單圖皆 0/5 seeds 勝出）。影線加粗方向不再延伸。

### 2026-10-12：配色對照與分類頭訓練診斷

**配色**（決議 #107）：紅綠對調／黑白／藍橘的單圖 Macro-F1 為 0.3204／0.3119／0.3205，雙圖為 0.3134／0.3159／0.3139，都低於正式圖的 0.3351／0.3390。資訊相同的紅綠對調也差 0.015～0.026，作為本協定的雜訊底線；先前畫法、影線、K60 的小差距都在此範圍內。

**分類頭訓練診斷**（臨時腳本，未保留 runner；K20＋Volume、3 seeds × 3 檔）：線性頭 lr=1e-3 訓練 30／100／300 epochs 的 Train Macro-F1 為 0.50／0.66／0.77，Validation 為 0.333／0.352／0.367，Validation ROC-AUC 為 0.563／0.551／0.544；lr=1e-4、100 epochs 為 0.376／0.571；MLP(256) 300 epochs 為 0.361／0.535。分類頭不是訓練不足，而是可背下 Train 但無法泛化，AUC 上限約 0.57。相對地，正式 Fusion 訓練 3 epochs 後 Train loss 仍為 1.15～1.17（高於三分類亂猜的 1.099）且 Validation loss 仍在下降，有訓練不足的跡象，待以更多 epochs 重訓確認。

### 2026-10-12：長期間 walk-forward 網格（2016～2025）

`run_horizon_period_grid.py`：7 個逐年 folds、t+1／t+3／t+5、單圖／K20＋Volume／K20＋SMA／三圖、兩種分類頭設定；A 為 AAPL／NVDA／MSFT（5,091 筆 Validation），B 為 12 檔跨產業（20,364 筆）。正式 Test 期間未建立樣本。詳見決議 #108、[ViT 實驗總結](vit_experiments_summary_2026-10-02.md) §12。

- 所有 ViT 設定的 ROC-AUC：A 為 0.4987～0.5170，B 為 0.5034～0.5093；同天期內各輸入的配對差在 A ±0.011、B ±0.005 以內。
- Numeric Logistic（t+1／t+3／t+5）：A 為 0.5216／0.5065／0.5104，B 為 0.5101／0.5045／0.5023。
- A 的 2025 fold 明顯高於其他年份（t+5 為 0.549～0.582），先前單一 Validation 的數字因此偏樂觀。
- 本機 MPS 執行時間：A 約 38 分鐘、B 約 1 小時 46 分鐘。
