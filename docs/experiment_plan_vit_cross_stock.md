# ViT 跨股票圖形學習實驗計畫

> 日期：2026-10-01
> 狀態：2026-10-02 全量 seed40／3 epochs Pilot3 已完成，Test locked；結果未顯示可泛化方向訊號，實驗程式已封存移除。
> 範圍：獨立視覺實驗，不修改正式 Fusion pipeline 或覆蓋舊實驗。

## 一、研究假設

目標是從過去的價格與成交量圖，學到能預測未來報酬方向、且能轉移到未見股票的視覺表示；不是要求模型辨識人工定義的頭肩底等型態。

| 假設 | 驗證方式 |
|---|---|
| H1：金融圖形適應能改善預測 | 同一份資料、共用分類頭，比較 Frozen 與 LoRA r4 |
| H2：更多股票能改善泛化 | 固定年份與驗證集，比較巢狀的 10 檔與 30 檔訓練股票 |
| H3：更長歷史能改善泛化 | 固定股票與驗證集，比較 2019–2021 與 2016–2021 訓練資料 |

上述假設都可能不成立；不能只用訓練 loss 下降判定成功。

## 二、資料與控制變數

- 初步股票池：共 50 檔跨產業大型美股，30 檔訓練、10 檔未見驗證股票、10 檔未見測試股票。名單固定在 `configs/experiments/vit_cross_stock.json`，為手選大型股，不宣稱已取得 point-in-time 歷史成分股全集。
- 約 10 年資料：2016–2025；2016–2021 訓練、2022–2023 驗證、2024–2025 測試。正式產生 manifest 前需檢查資料覆蓋率。
- 未見驗證／測試股票的任何日期都不參與梯度更新；未見測試股票不參與模型選擇。主測試不額外訓練新股票分類頭。
- 分開報告已見股票的未來表現與未見股票的未來表現。
- 每筆輸入固定為過去 20 個交易日的 K 線圖與成交量圖，各 224×224；輸出預測未來第 5 個交易日相對當日收盤的報酬。
- Backbone 固定 `google/vit-base-patch16-224`，所有股票共用同一個分類頭，不提供 ticker 作為模型特徵。
- 日期邊界依實際 label_end_date 排除跨界訓練／驗證樣本；保留預測跨度相應的時間隔離，不隨機切分每日圖片。
- 不要求 50 檔股票的所有日期取交集；使用固定日曆邊界，各股票個別檢查完整窗口與未來標籤。
- 明確檢查價格／成交量的公司行動調整、缺值與異常，不用未來資料決定圖片縮放或特徵標準化。
- 若只能取得現今成分股回溯資料，須標註存活者／股票池選擇偏差，不宣稱為無偏的歷史 S&P 500 全集。

### Label：本次獨立實驗採用二分類

使用者同意開始實作這份規劃，本輪沿用提案，以「看漲跌」為首要問題，使用調整後收盤價報酬正負二分類：`r_5 > 0` 為漲、`r_5 < 0` 為跌；精確等於零的樣本排除並記錄數量。所有股票遵循相同定義。下載指定 `auto_adjust=True, actions=True`，調整政策及公司行動欄位隨 OHLCV 保存。

這會改變舊實驗的三分類任務，因此新舊 Macro-F1 不能直接比較。若之後需保持 bearish／neutral／bullish 三分類，需另行確認一致的 neutral 門檻及是否使用歷史波動調整。正式 pipeline 的 label 完全不變。

## 三、Baseline 與候選模型

| 方法 | 更新參數 | 目的 |
|---|---|---|
| 訓練集多數類別預測 | 無模型訓練 | 檢查是否只是利用類別比例 |
| 價格／成交量數值的 Logistic Regression | 簡單數值分類器 | 檢查圖片模型是否有額外價值；標準化僅 fit train |
| Frozen ViT＋共用分類頭 | 僅分類頭 | 必要視覺基準，ViT 本身不變 |
| LoRA r4 ViT＋共用分類頭 | LoRA 與分類頭 | 第一輪主要候選，測試金融圖形適應 |

LoRA 沿用既有設定：最後 4 個 blocks 的 query／value，rank=4、alpha=8、dropout=0.1。初始 head LR=3e-4、LoRA LR=1e-4、weight decay=1e-3；最多 10 epochs、patience=3。這些是起始配置，不宣稱最佳參數。

Frozen 與 LoRA 各自從相同預訓練 backbone、相同 seed 的分類頭初始化，獨立訓練；不預先把 Frozen 訓練後的分類頭交給 LoRA，以免增加比較因素。資料、圖片處理、採樣方式、分類頭與評估協定一致；batch 大小經 smoke test 確認後固定。避免一般自然圖像的水平／垂直翻轉、任意裁切。

## 四、評估指標與統計

- 模型選擇主指標：未見驗證股票的 Macro-F1，每檔獨立計算後等權平均。
- 同時報告已見股票驗證表現，避免只看單一總分。
- 最終主結果：未見測試股票的平均 Macro-F1；附 balanced accuracy、accuracy、各類 precision／recall、混淆矩陣與預測類別比例。
- 分產業、年份與股票呈現結果，附每組樣本數。
- 初步使用 seeds 40、41、42；必要時補 43、44。報告同 seed 配對的 LoRA−Frozen 差值、平均與標準差，不只挑最好 seed。
- seeds 的標準差不是時間外泛化的信賴區間。正式統計可對日期區塊重抽樣，同一區塊保留所有股票以維持市場共同波動；區塊長度與敏感度檢查需另記錄。
- 在開啟測試前固定模型、seed、訓練協定與主要比較；測試結果不得拿來挑設定。測試不是單次最高分展示。

## 五、執行優先順序與消融

1. 實作獨立 runner 與股票／日期 manifest，驗證共用分類頭與標籤規則；保留舊 runner 行為。
2. 小資料 smoke test：Frozen 與 LoRA 各跑 1 epoch，檢查梯度、參數更新、日期隔離、報告與圖表；不讀取正式 test 指標。
3. 10 檔訓練股票的單 seed 計時，測量完整資料處理與訓練成本。
4. 主實驗：30 檔訓練股票、2016–2021 訓練歷史，Frozen vs LoRA，3 seeds。
5. 在相同 validation 上依序做股票數與歷史長度消融，不同時改兩個因素；是否增加 seeds 依資源與事先記錄的規則決定。
6. 固定比較協定後評估保留 test；無穩定改善也照實記錄。第一輪不換 backbone、不做全模型 fine-tune、不做大型網格搜尋。

## 六、計算預算與停止點

30 檔 × 6 年 × 約 252 交易日，訓練部分約 45,360 組雙圖樣本。程式只產生有實驗用途的股票／日期組合：train 股票共 10 年、validation 股票 2 年、test 股票 2 年，約 85,680 組、171,360 張圖片；不會為 held-out 股票的早期未使用期間產圖。這只是缺值、上市時間、窗口及切分排除前的量級估算，鄰近樣本不獨立。

不以舊的三股票訓練時間直接推算新實驗完成時間。先測每張圖檔大小、繪圖／抽特徵速度、LoRA 每 epoch 時間、記憶體峰值，再估 3 seeds 的 6 個主訓練 run，預留 1.5–2 倍時間與儲存緩衝。Frozen 特徵可快取；LoRA 訓練不能直接重用凍結 backbone 的特徵。

任何大量下載、繪圖或長時間訓練開始前，提供使用者命令、輸出路徑、即時 log 查看方式與中斷／續跑方式；到該交接點停止，等待使用者執行完成後回報。

## 七、可重現性與輸出

- 保存股票池來源、選取日期、產業、分組與排除原因；保存資料取得時間、公司行動處理方式與資料版本／雜湊。
- 保存固定日期 manifest、label 規則、樣本／類別數、模型 revision、完整配置、git revision、硬體及套件版本。
- Run 輸出使用獨立實驗命名空間，不覆蓋既有 ViT adaptation／backbone comparison。
- 每個 run 保存 `report.json`、預測明細、epoch history、最佳 checkpoint；利用既有報告工具產生 `report.md`、訓練曲線與混淆矩陣。
- 共用報告工具已新增 `cross_stock` 路由，使用專用二分類圖表，不影響舊三分類報告。
- 圖表與逐次結果統一放在 `data/outputs/experiments/`；本文件記錄設計，重要結論再摘要至 `docs/data_and_experiments_log.md`。

## 參考

- [(Re-)Imag(in)ing Price Trends](https://www.aidf.nus.edu.sg/wp-content/uploads/2022/02/Xiu-Re-Imagining-Price-Trends.pdf)：多股票共用圖像模型的研究參考，使用 CNN；不能當作本實驗 ViT 的效果保證。
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)：過去訓練、未來評估及 gap 的基本概念；本實驗仍需另外處理跨股票分組與未來標籤跨度。
- [yfinance download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)：明確指定 OHLC 調整與公司行動下載；end 為不包含的右邊界。

## 封存狀態與結果位置

- 2026-10-02 結果整理完成後，訓練入口、資料準備程式、實驗模型與專用測試已移除；本文件保留方法設計，不再是可直接執行的操作手冊。
- 二分類報告工具 `experiments/vision/cross_stock_report.py` 與通用 `render_experiment_report.py` 保留，可由既有 JSON 重建 Markdown 與圖表。
- Pilot3：Frozen／LoRA 未見 Validation mean-ticker Macro-F1 為 0.4922／0.4908；ROC-AUC 為 0.4909／0.4844，未顯示可泛化方向辨識能力，Test 未開啟。
- 完整判讀見 [ViT 實驗總結](vit_experiments_summary_2026-10-02.md)，原始報告見 [Pilot3 report](../data/outputs/experiments/vit_cross_stock/runs/14c913bcec83cfba/pilot3/report.md)。Smoke 輸出已刪除，不作效果證據。
- 未來若要驗證 label horizon、neutral zone、SMA 或 walk-forward，應建立新版實驗程式與新輸出命名空間，避免覆寫此批歷史結果。
