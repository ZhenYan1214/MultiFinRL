# MultiFinRL ViT 實驗總結

> 更新日期：2026-10-02
> 範圍：僅整理 ViT／視覺輸入相關正式報告；不包含 Fusion、事件抽取、Decoder、PPO 或回測。
> 原則：不同 label、股票池與切分協定的 F1 不直接橫向排名；Smoke 輸出已刪除，不納入成效判斷。

## 1. 執行摘要

目前最合理的正式視覺基線仍是 `google/vit-base-patch16-224` 的 Frozen ViT-B/16，搭配雙圖輸入與小型分類頭。現有證據不支持把正式 encoder 改成 DINOv3、MaxViT-T 或 MaxViT-S，也不支持直接採用 last-block fine-tuning 或 LoRA。

舊三分類實驗曾觀察到 LoRA r4 的平均 Validation Macro-F1 高於 Frozen，但只在 5 個 seeds 中勝出 3 次，未通過預先設定的穩定性門檻。最新且更嚴格的跨股票共用分類頭實驗使用 30 檔股票訓練、10 檔未見股票驗證；跑 3 epochs 後，Frozen 與 LoRA 的未見股票 Macro-F1 僅差 0.0014，ROC-AUC 與 balanced accuracy 都接近或低於 0.5。這表示目前模型主要是在調整 UP／DOWN 預測比例，尚未證明能從圖形中辨識可泛化的未來方向訊號。

因此，現階段不應繼續用更多 epochs 或更大的 backbone 堆算力。下一輪應優先調整預測任務本身，例如排除接近零的報酬、加入 neutral zone、測試較長 horizon，並把 ROC-AUC、balanced accuracy 與 log loss 設為 Macro-F1 之外的必要 sanity gate。

## 2. 實驗脈絡與可比性

目前 ViT 報告分為四類，彼此的任務定義不同：

| 實驗群 | 任務 | 主要用途 | 是否可與跨股票二分類直接比較 |
|---|---|---|---|
| 雙圖／輔助圖實驗 | AAPL 三分類 | 比較視覺輸入設計 | 否 |
| ViT adaptation | 三分類、train-period quantile label | 比較 Frozen、last blocks、LoRA | 否 |
| Backbone comparison | 三分類、Frozen encoder | 比較 ViT、DINOv3、MaxViT | 否 |
| Cross-stock shared head | 二分類、未來 5 日 adjusted-close 正負 | 驗證跨股票零額外 probe 的泛化 | 最新且最嚴格 |

舊三分類的 BEARISH／NEUTRAL／BULLISH 由 train-period 分位數定義；最新跨股票實驗則只判斷未來第 5 個交易日調整後收盤價報酬的正負。因此，兩者的 Macro-F1 基準與難度不同，不能因數字較高就判定模型較好。

## 3. 視覺輸入設計

### 3.1 單圖與雙圖

早期 AAPL 報告比較單張圖與雙圖輸入：

| 輸入 | Test accuracy | Test Macro-F1 |
|---|---:|---:|
| Baseline single image | 0.3077 | 0.3060 |
| Controlled single direct | 0.3173 | 0.3148 |
| Dual image | **0.3413** | **0.3390** |

雙圖比單圖約高 0.024～0.033 Macro-F1，支持保留兩張互補圖形。不過這組報告使用較舊的 208 日 Test 與舊切分協定，應視為輸入設計的歷史證據，而不是目前正式效能。

### 3.2 Volume、RSI、SMA、MACD

AAPL 單 seed 輔助圖實驗以 Validation 選擇第二張圖：

| 第二張圖 | Validation Macro-F1 |
|---|---:|
| Volume | 0.2819 |
| RSI | 0.3067 |
| SMA | **0.3802** |
| MACD | 0.2996 |

SMA 被選中後的 Test Macro-F1 為 0.3415，高於 Volume baseline 的 0.3253，差值約 +0.0162。這是目前唯一支持「技術指標圖可能優於純成交量圖」的直接結果，但只有 seed42，且 H_t／H_r 固定為既有 production vectors。因此 SMA 適合作為下一輪受控實驗候選，尚不足以直接替換正式第二張圖。

## 4. Frozen、部分 Fine-tune 與 LoRA

### 4.1 AAPL-only：last4 在 Validation 勝出，但 Test 反轉

| 策略 | Validation Macro-F1 | Test Macro-F1 |
|---|---:|---:|
| Frozen | 0.4095 | **0.3377** |
| Last 4 blocks | **0.4309** | 0.3210 |

Last4 在 Validation 提高 0.0214，但 Test 反而下降 0.0167，顯示單股小資料下部分 fine-tuning 容易過度適配 Validation。Last1 與 Last2 的 Validation Macro-F1 也都低於 Frozen，因此沒有採用。

### 4.2 AAPL＋NVDA、MSFT holdout

分股票 probe 的設定中，Frozen Validation Macro-F1 為 0.3757，Last4 為 0.3610；Frozen 的 MSFT future Test Macro-F1 為 0.3066。改成 shared head、held-out 股票不訓練新 probe 後，Frozen／Last4 Validation Macro-F1 為 0.3498／0.3141，Frozen 的 MSFT Test Macro-F1 為 0.2870。

兩個設定都選出 Frozen，表示 Last4 的改善沒有跨股票穩定性；同時也顯示「為每支新股票訓練 probe」會使結果比真正 shared-head 泛化樂觀。

### 4.3 AAPL＋NVDA＋MSFT、JPM holdout：LoRA r4 的初步正向結果

單一 seed42 實驗中：

| 策略 | Validation mean-ticker Macro-F1 | JPM Test Macro-F1 | JPM Test log loss |
|---|---:|---:|---:|
| Frozen | 0.2944 | 0.3209 | 1.1014 |
| LoRA r4 | **0.3602** | **0.3458** | **1.0894** |
| LoRA r8 | 0.3226 | — | — |

LoRA r4 在這一輪相對 Frozen 有正向改善，r8 則沒有提供額外優勢。然而這個 holdout 設定仍會替 JPM 訓練新的分類 probe，因此不能視為完全零樣本、共用分類頭的跨股票泛化。

### 4.4 五 seed 穩定性

| 策略 | Validation mean-ticker Macro-F1 | 範圍 |
|---|---:|---:|
| Frozen | 0.3328 ± 0.0392 | 0.2944～0.3797 |
| LoRA r4 | **0.3662 ± 0.0127** | 0.3530～0.3867 |

LoRA r4 平均高出 0.0334，但只勝 3/5 seeds，沒有通過預先設定的 4/5 勝出門檻。結果顯示 LoRA 可能降低 seed 波動，但提升仍受股票與初始化影響，因此當時決定正式 pipeline 維持 Frozen。

## 5. Backbone 比較

所有 backbone 比較都固定 encoder，只訓練同型態分類頭，使用 AAPL／NVDA／MSFT、五個 seeds、Validation-only 協定：

| Backbone | Mean-ticker Macro-F1 | 相對 ViT | 勝出 seeds | Gate |
|---|---:|---:|---:|---|
| ViT-B/16 | **0.3447 ± 0.0206** | — | — | Baseline |
| MaxViT-T | 0.3374 ± 0.0205 | -0.0073 ± 0.0220 | 2/5 | FAIL |
| DINOv3 ViT-B/16 | 0.3263 ± 0.0122 | -0.0184 ± 0.0129 | 1/5 | FAIL |
| MaxViT-S | 0.3130 ± 0.0189 | -0.0317 ± 0.0188 | 0/5 | FAIL |

更大的 MaxViT-S 並沒有更好；DINOv3 的大型自然影像預訓練也沒有轉化為金融圖表優勢。現階段沒有證據支持更換 ViT-B/16。這些結果只否定「直接使用 frozen feature 替換」，不代表經金融圖表預訓練或適當 domain adaptation 後仍一定無效。

## 6. 最新跨股票共用分類頭實驗

### 6.1 設定

- Base model：`google/vit-base-patch16-224`。
- 輸入：過去 20 個交易日的 K 線圖與 Volume 圖，兩張 224×224 RGB 圖。
- 價格：yfinance `auto_adjust=True`，圖形與 label 均使用調整後 OHLC／Close。
- Label：`adjusted_close[t+5] / adjusted_close[t] - 1` 的正負；0=DOWN、1=UP，精確零排除。
- Train：30 檔股票、2016–2021、45,095 筆。
- Validation seen：30 檔已見股票、2022–2023、14,872 筆。
- Validation unseen：10 檔完全未參與梯度更新的股票、2022–2023、4,955 筆。
- Test：2024–2025，維持 LOCKED。
- 策略：Frozen 分類頭與最後四個 blocks query／value 的 LoRA r4。

### 6.2 Pilot3 結果

| 方法 | Unseen mean-ticker Macro-F1 | Accuracy | Balanced accuracy | ROC-AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| Frozen | **0.4922** | 0.4971 | **0.4951** | **0.4909** | **0.7095** |
| LoRA r4 | 0.4908 | **0.4979** | 0.4944 | 0.4844 | 0.7136 |
| Numeric Logistic | 0.4803 | **0.5068** | 0.4964 | 0.4882 | **0.6935** |
| Majority UP | 0.3448 | **0.5267** | 0.5000 | 0.5000 | 17.0580 |

依既定主指標，Frozen 僅比 LoRA 高 0.0014，實質上幾乎相同。LoRA 只在 4/10 檔 unseen 股票勝過 Frozen。更重要的是，兩種模型的 balanced accuracy 與 ROC-AUC 都接近或低於 0.5，accuracy 也沒有超過 Numeric Logistic 或 Majority baseline。

三個 epochs 中，Macro-F1 上升但 ROC-AUC 下降：

| 策略 | Epoch 1 AUC | Epoch 2 AUC | Epoch 3 AUC |
|---|---:|---:|---:|
| Frozen | 0.5026 | 0.4854 | 0.4909 |
| LoRA r4 | 0.5027 | 0.4866 | 0.4844 |

Macro-F1 的上升主要來自模型把 UP／DOWN 預測比例調得更平均，而非真實類別的機率分離變好。第 3 epoch 中，真實 DOWN 樣本的平均 UP 機率甚至略高於真實 UP 樣本，與低於 0.5 的 AUC 一致。因此目前不能宣稱 ViT 已學會可泛化的圖形漲跌訊號，也不應開啟 Test。

## 7. 綜合判斷

### 已得到支持的判斷

1. 雙圖通常比單圖好，保留兩張互補圖形是合理設計。
2. SMA 圖在單 seed 實驗中優於 Volume，值得作下一輪受控驗證。
3. 在已測試的 frozen backbones 中，ViT-B/16 優於 DINOv3、MaxViT-T 與 MaxViT-S。
4. Last-block fine-tuning 容易在單股 Validation 過度適配，沒有穩定跨股票收益。
5. LoRA r4 在舊三分類、多 probe 協定中曾有平均改善，但未達穩定採用門檻。
6. 最新 shared-head 跨股票二分類中，Frozen 與 LoRA 幾乎沒有差異，而且兩者方向辨識能力都接近隨機。

### 尚未得到支持的主張

- 不能主張 LoRA 已讓 ViT 穩定變強。
- 不能主張增加股票數量後，ViT 已學會一般化的技術圖形型態。
- 不能主張更大或更新的自然影像 backbone 會改善金融圖表預測。
- 不能用目前約 0.49 的 Macro-F1 單獨宣稱有效，因為隨機但預測比例平衡的分類器也可能得到接近 0.5 的 Macro-F1。

## 8. 建議的下一步

1. **暫停增加 epochs／模型大小**：目前瓶頸不是明顯欠擬合，而是 label 與輸入對未來 5 日方向缺乏可分訊號。
2. **先做 label ablation**：固定資料與 ViT，測試 5／10／20 日 horizon，以及用 train-only 波動度或報酬分位數設定 neutral zone，排除接近零的雜訊樣本。
3. **加入評估門檻**：主指標仍可保留 mean-ticker Macro-F1，但候選模型必須同時滿足 unseen ROC-AUC、balanced accuracy、log loss 不劣於簡單 baseline。
4. **優先測 SMA 雙圖**：比較 K 線＋Volume 與 K 線＋SMA，先用一個 seed、短 epochs 篩選，再對勝者跑多 seed。
5. **保持 Test 鎖定**：先在 Validation unseen 與新的 walk-forward folds 確認穩定改善，再一次性開啟 Test。
6. **模型選擇維持 Frozen ViT-B/16**：在出現跨 seed、跨股票、跨時間的穩定改善前，不更換正式視覺 encoder。

## 9. 報告來源

- [雙圖與單圖報告](../data/outputs/experiments/dual_image_vit/AAPL/dual_image_report.json)
- [輔助圖比較](../data/outputs/experiments/auxiliary_vision/AAPL/report.md)
- [AAPL ViT adaptation](../data/outputs/experiments/vit_adaptation/AAPL/run/report.md)
- [AAPL／NVDA，MSFT holdout](../data/outputs/experiments/vit_adaptation/AAPL_NVDA_holdout_MSFT/run/report.md)
- [Shared-head MSFT holdout](../data/outputs/experiments/vit_adaptation/AAPL_NVDA_holdout_MSFT/zero_shot_shared_head_run/report.md)
- [LoRA，JPM holdout](../data/outputs/experiments/vit_adaptation/AAPL_NVDA_MSFT_holdout_JPM/run/report.md)
- [LoRA 五 seed 彙整](../data/outputs/experiments/vit_adaptation/AAPL_NVDA_MSFT_holdout_JPM/multiseed_summary/report.md)
- [DINOv3 比較](../data/outputs/experiments/dinov3_comparison/AAPL_NVDA_MSFT/run/report.md)
- [MaxViT-T 比較](../data/outputs/experiments/maxvit_comparison/AAPL_NVDA_MSFT/run/report.md)
- [MaxViT-S 比較](../data/outputs/experiments/maxvit_s_comparison/AAPL_NVDA_MSFT/run/report.md)
- [Cross-stock timing](../data/outputs/experiments/vit_cross_stock/runs/14c913bcec83cfba/timing/report.md)
- [Cross-stock pilot3](../data/outputs/experiments/vit_cross_stock/runs/14c913bcec83cfba/pilot3/report.md)

## 10. 保存狀態

- 所有 Smoke run 輸出已刪除，不作效果證據。
- 報告、圖表、history 與 predictions 保留。
- 每個重要實驗只保留選定策略的 checkpoint；被刪策略在報告中的舊 checkpoint 路徑可能已失效。
- Cross-stock Frozen feature cache 保留，可供後續 Frozen 分類頭與 label ablation 重用；若 label 或樣本 membership 改變，仍需重新建立相符的資料／特徵索引。
- 2026-10-02 依專案整理決策，已移除完成任務使用的 ViT 訓練 runner、實驗模型、資料準備程式與專用測試；正式 `module_b_encoder` 未變更。
- 保留 `render_experiment_report.py`、`visualize_comparison_report.py` 與 `cross_stock_report.py`，可由既有 `report.json` 重建 Markdown 與圖表。
- 本文件與 `data/outputs/experiments/` 內的原始 JSON／Markdown／圖表是歷史結果依據。若未來要重跑 label horizon、SMA 或 walk-forward 等新問題，應建立新的實驗實作與新輸出命名空間，而不是改寫這批封存結果。
