# MultiFinRL ViT 實驗總結

> 更新日期：2026-10-06（2026-10-10 封存整理）
> 後續方向：ViT 工作收斂回 AAPL／NVDA／MSFT；§6–§9 的 50 檔跨股票與 8 檔科技巨頭實驗只留紀錄，各實驗設定摘要見 [封存紀錄](../experiments/vision/README.md)。
> 範圍：僅整理 ViT／視覺輸入相關正式報告；不包含 Fusion、事件抽取、Decoder、PPO 或回測。
> 原則：不同 label、股票池與切分協定的 F1 不直接橫向排名；Smoke 輸出已刪除，不納入成效判斷。

## 1. 執行摘要

目前最合理的正式視覺基線仍是 `google/vit-base-patch16-224` 的 Frozen ViT-B/16，搭配雙圖輸入與小型分類頭。現有證據不支持把正式 encoder 改成 DINOv3、MaxViT-T 或 MaxViT-S，也不支持直接採用 last-block fine-tuning 或 LoRA。

舊三分類實驗曾觀察到 LoRA r4 的平均 Validation Macro-F1 高於 Frozen，但只在 5 個 seeds 中勝出 3 次，未通過預先設定的穩定性門檻。最新且更嚴格的跨股票共用分類頭實驗使用 30 檔股票訓練、10 檔未見股票驗證；跑 3 epochs 後，Frozen 與 LoRA 的未見股票 Macro-F1 僅差 0.0014，ROC-AUC 與 balanced accuracy 都接近或低於 0.5。這表示目前模型主要是在調整 UP／DOWN 預測比例，尚未證明能從圖形中辨識可泛化的未來方向訊號。

2026-10-05 完成的 Frozen window × input-mode 實驗進一步顯示：在相同的 30 檔訓練股票與 10 檔未見 Validation 股票上，單張 K 線在 16／18／20／22／24 日窗口都優於 K 線＋Volume 雙圖；最佳 `w22_single` 的 mean-ticker Macro-F1 為 0.4949，但 ROC-AUC 只有 0.4950，仍沒有證明存在可泛化的方向訊號。窗口 16～24 日之間的差距也很小，不能只靠 seed40 宣稱 22 日就是最佳窗口。

同日完成的 Train-only q33／q67 三分類實驗也沒有改變此結論。三分類 `w22` 五 seed mean-ticker Macro-F1 為 0.3213±0.0112、balanced accuracy 0.3440±0.0104、Macro OvR AUC 0.5117±0.0041；數值 Logistic 的 balanced accuracy 與 AUC 則為 0.3612／0.5476。三分法能把接近零的報酬明確歸為 NEUTRAL，但 Frozen 圖像特徵仍沒有展現足夠的類別分離能力。

在完全相同的三分類協定下加入 Volume 雙圖後，w22 mean-ticker Macro-F1 從 0.3213 降至 0.3139，三個窗口都只有 2/5 seeds 勝過單圖。雙圖雖讓 w22 AUC 從 0.5117 微升至 0.5132、log loss 從 1.0953 降至 1.0898，但 balanced accuracy 沒有改善，且 NEUTRAL F1 從 0.2571 降至 0.1947。因此目前仍以單張 K 線作 Frozen baseline。

2026-10-05 完成的 8 檔科技巨頭、4 個股票留出 folds、5 seeds 實驗，首次在新協定中得到一致的輸入設計訊號：K 線＋SMA 在 20／22／24 日都優於相同窗口的單圖，mean-ticker Macro-F1 分別提高 0.0190、0.0160、0.0133，且 AUC 在 18/20、16/20、18/20 個 fold-seed 配對勝出。Volume 仍沒有收益。依主指標選出的 `w24_dual_sma` 為 0.3424±0.0210，但只比 `w22_dual_sma` 高 0.0011；後者 AUC 相同且 log loss 較佳，因此 22／24 日應視為打平。

這是目前最值得保留的視覺輸入方向，但還不是「ViT 已學會預測漲跌」的證明。`w24_dual_sma` 的 AUC 0.5257 幾乎等於 Numeric Logistic 0.5253，log loss 1.1005 反而差於 1.0916；同一份 Validation 又同時負責 best epoch 與九組候選選擇，結果可能偏樂觀。Test 應繼續鎖定，下一步先用獨立的時間 fold 驗證 SMA 增益，而不是再擴大 backbone 或搜尋更細的窗口。

## 2. 實驗脈絡與可比性

目前 ViT 報告分為四類，彼此的任務定義不同：

| 實驗群 | 任務 | 主要用途 | 是否可與跨股票二分類直接比較 |
|---|---|---|---|
| 雙圖／輔助圖實驗 | AAPL 三分類 | 比較視覺輸入設計 | 否 |
| ViT adaptation | 三分類、train-period quantile label | 比較 Frozen、last blocks、LoRA | 否 |
| Backbone comparison | 三分類、Frozen encoder | 比較 ViT、DINOv3、MaxViT | 否 |
| Cross-stock shared head | 二分類、未來 5 日 adjusted-close 正負 | 驗證跨股票零額外 probe 的泛化 | 最新且最嚴格 |
| Tech-giants input ablation | 三分類、per-fold Train-only q33／q67 | 比較單圖、Volume、SMA 與窗口 | 只可在本協定內比較 |

舊三分類的 BEARISH／NEUTRAL／BULLISH 由 train-period 分位數定義；最新跨股票實驗則只判斷未來第 5 個交易日調整後收盤價報酬的正負。因此，兩者的 Macro-F1 基準與難度不同，不能因數字較高就判定模型較好。

## 3. 視覺輸入設計

### 3.1 單圖與雙圖

早期 AAPL 報告比較單張圖與雙圖輸入：

| 輸入 | Test accuracy | Test Macro-F1 |
|---|---:|---:|
| Baseline single image | 0.3077 | 0.3060 |
| Controlled single direct | 0.3173 | 0.3148 |
| Dual image | **0.3413** | **0.3390** |

雙圖比單圖約高 0.024～0.033 Macro-F1，但這組報告使用較舊的 AAPL-only、208 日 Test 與舊切分協定，只能視為歷史證據。最新跨股票 Frozen 實驗中，Volume 雙圖在全部五個窗口都低於單張 K 線，因此目前不再把「雙圖通常較好」當作正式結論。

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

## 7. Frozen 窗口與單／雙圖消融實驗

### 7.1 設定

- Backbone：`google/vit-base-patch16-224`，完全 Frozen，只訓練 LayerNorm＋Linear 二分類頭。
- 窗口：16／18／20／22／24 個交易日。
- 單圖：K 線；雙圖：K 線＋Volume，兩張圖各自通過同一個 Frozen ViT 後串接 CLS。
- 所有設定沿用相同的 window20 日期與 label，避免因窗口不同而改變比較樣本。
- Train：30 檔、45,095 筆；Validation seen：30 檔、14,872 筆；Validation unseen：10 檔、4,955 筆。
- Seed40、3 epochs；Test 維持 LOCKED。
- 主指標：`validation_unseen.mean_ticker_macro_f1`。

### 7.2 結果

| 設定 | Best epoch | Unseen mean-ticker Macro-F1 | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| w16 single | 2 | 0.4908 | 0.4922 | 0.4884 |
| w16 dual | 2 | 0.4066 | 0.4973 | 0.4966 |
| w18 single | 2 | 0.4894 | 0.4917 | 0.4896 |
| w18 dual | 2 | 0.4104 | 0.4959 | 0.4757 |
| w20 single | 1 | 0.4855 | 0.4939 | 0.4857 |
| w20 dual | 1 | 0.4750 | 0.4857 | 0.4740 |
| w22 single | 2 | **0.4949** | **0.4983** | **0.4950** |
| w22 dual | 1 | 0.4818 | 0.4879 | 0.4836 |
| w24 single | 2 | 0.4931 | 0.4974 | 0.4897 |
| w24 dual | 1 | 0.4818 | 0.4877 | 0.4950 |

`w22_single` 依預先指定主指標排名第一，但只比 `w24_single` 高 0.0019、比 `w16_single` 高 0.0041，差距不足以在單一 seed 下判定 22 日有實質優勢。`w22_single` 的 pooled accuracy 為 0.5017、balanced accuracy 0.4983、ROC-AUC 0.4950、log loss 0.6956，仍接近隨機；它也低於 Majority UP 的 accuracy 0.5267，且 log loss 劣於 Numeric Logistic 的 0.6935。

以 10 檔未見股票作 post-hoc ticker bootstrap，`w22_single` mean-ticker Macro-F1 的 95% 區間約為 0.4764～0.5174，跨過 0.5。各股從 ADBE 0.4453 到 LOW 0.5766，只有 LOW 顯示較明顯的正向結果，因此目前不能主張模型已學會跨股票通用型態。

Volume 雙圖相對單圖的 Macro-F1 差值依序為 −0.0842、−0.0790、−0.0105、−0.0131、−0.0113。16／18 日雙圖在 10/10 檔股票都輸給單圖，且混淆矩陣顯示模型約 86%～87% 預測為 DOWN；20～24 日的雙圖差距較小，但仍未勝出。這表示目前直接串接兩個 Frozen CLS 不會自動利用成交量，反而可能增加分類頭最佳化與類別偏移問題。

正式 run 約從 11:10 執行至 15:51，總耗時約 4 小時 40 分；其中十組 Frozen 特徵抽取合計約 277.6 分鐘，分類頭每組只需約 1.3～2.2 秒。Feature cache 約 1.9 GB，後續相同協定的多 seed 分類頭驗證不必重新抽特徵，成本很低。

### 7.3 本輪判斷

1. 暫時選 `w22_single` 作 Validation 候選，但不升級為最終設定，也不解鎖 Test。
2. 現階段應優先使用單張 K 線；Volume 雙圖不保留為此跨股票 Frozen baseline。
3. 窗口 16～24 日對單圖的影響很小，最佳窗口可能只是 seed／Validation 波動。
4. 下一個最便宜且必要的確認是 `w20_single`／`w22_single`／`w24_single` 多 seed；若 AUC 與 balanced accuracy 仍未穩定超過 0.5，應停止調窗口，改做 label／horizon 設計。

## 8. Train-only 分位數三分類實驗

### 8.1 設定與標籤

沿用相同 Frozen K 線 CLS features，不重新畫圖或執行 ViT forward。只用 Train 期間 5 日
forward return 的 q33／q67 定義三類，Validation 不參與門檻估計：

- DOWN：return < −0.626378%。
- NEUTRAL：−0.626378% ≤ return ≤ +1.478116%。
- UP：return > +1.478116%。

Train 45,095 筆因此近似三等分；套用到 Validation unseen 後為 DOWN 1,982（40.0%）、
NEUTRAL 1,178（23.8%）、UP 1,795（36.2%）。這個分布變化不是重新平衡錯誤，而是保留
Train 門檻後觀察到的時間／股票分布漂移。實驗比較 20／22／24 日單張 K 線、seeds
40～44、每組 3 epochs，Test 全程 LOCKED。

### 8.2 五 seed 結果

| Window | Mean-ticker Macro-F1 | Balanced accuracy | Macro OvR AUC | Log loss |
|---|---:|---:|---:|---:|
| w20 | 0.3168 ± 0.0087 | 0.3385 ± 0.0059 | 0.5028 ± 0.0056 | 1.1001 ± 0.0072 |
| w22 | **0.3213 ± 0.0112** | 0.3440 ± 0.0104 | **0.5117 ± 0.0041** | **1.0953 ± 0.0024** |
| w24 | 0.3192 ± 0.0113 | **0.3451 ± 0.0072** | 0.5092 ± 0.0069 | 1.0961 ± 0.0033 |
| Numeric Logistic | 0.3043 | **0.3612** | **0.5476** | **1.0738** |

`w22` 依預先指定主指標平均值排名第一，但只比 w24 高 0.0021，且逐 seed 配對時只在
2/5 seeds 勝過 w24；w22 對 w20 也只有 3/5 勝出。w22 的平均三類 F1 為 DOWN 0.3678、
NEUTRAL 0.2571、UP 0.3710，NEUTRAL 明顯最難辨識。五 seed 的預測類別比例也大幅波動，
例如 DOWN 比例從 20.6% 到 59.5%，顯示小型 head 對初始化仍敏感。

雖然 w22 mean-ticker Macro-F1 高於 Numeric Logistic 0.0170，pooled Macro-F1 只高約
0.0045；同時 accuracy、balanced accuracy、AUC 與 log loss 全部輸給 Numeric Logistic。
Seed-pooled confusion matrix中，三種真實類別被預測成 DOWN／NEUTRAL／UP 的比例都約為
37%／23%／40%，不同真實類別的輸出分布非常接近。這表示 Macro-F1 的小幅優勢主要來自
預測比例，而非可靠的類別分離。

本輪直接重用 1.9 GB feature cache，15 組 head 加報告約 28 秒完成；正式 run 產物約 88 MB。
結論是：三分法比硬切二分類更符合任務語意，但**尚未讓 Frozen ViT 產生足以採用的跨股票
訊號**，不應據此開啟 Test。

### 8.3 單圖與雙圖比較

雙圖使用相同 Frozen ViT 分別抽取 K 線與 Volume CLS，再串接為 1,536 維；除輸入維度外，
label、股票、日期、seeds、epochs、learning rate 與選模方式完全相同。

| Window | 單圖 mean-ticker F1 | 雙圖 mean-ticker F1 | 雙圖 − 單圖 | 雙圖勝出 seeds |
|---|---:|---:|---:|---:|
| w20 | 0.3168 ± 0.0087 | 0.3080 ± 0.0111 | −0.0087 | 2/5 |
| w22 | **0.3213 ± 0.0112** | 0.3139 ± 0.0145 | −0.0074 | 2/5 |
| w24 | 0.3192 ± 0.0113 | 0.3109 ± 0.0212 | −0.0084 | 2/5 |

w22 的雙圖 accuracy 由 0.3558 提高到 0.3691，Macro OvR AUC 由 0.5117 微升到 0.5132，
log loss 由 1.0953 改善到 1.0898；但 balanced accuracy 由 0.3440 微降到 0.3433，且上述
AUC／log-loss 改善遠小於 Numeric Logistic（AUC 0.5476、log loss 1.0738）。逐 seed 配對中，
w22 雙圖的 log loss 在 4/5 seeds 改善，但主指標只在 2/5 seeds 提高。

類別 F1 顯示雙圖把 w22 的 DOWN 從 0.3678 提高到 0.4182，卻把 NEUTRAL 從 0.2571 降到
0.1947，UP 也由 0.3710 降到 0.3581。換言之，Volume 主要改變類別偏好與機率校準，沒有
帶來穩定的三類辨識能力。依預先指定主指標，本輪保留單張 K 線，不採用直接 CLS concat
的雙圖版本。

## 9. 科技巨頭單圖／Volume／SMA 實驗

### 9.1 設定

- 股票：AAPL、MSFT、NVDA、ADBE、GOOGL、META、AMZN、AVGO。
- 股票切分：4 個 folds，每次以 6 檔訓練，另外 2 檔作完全未見 Validation；每檔都輪流被留出一次。
- 時間：Train 2016–2021、Validation 2022–2023；2024 年起 Test 未建立、未抽特徵且維持 LOCKED。
- Label：每個 fold 僅以該 fold 訓練股票、Train 期間的 5 日 forward return q33／q67 定義 DOWN／NEUTRAL／UP。
- 模型：Frozen `google/vit-base-patch16-224`；只訓練 LayerNorm＋Linear head。
- 候選：20／22／24 日 × K 線單圖／K 線＋Volume／K 線＋SMA5/10/20。
- 重複：seeds 40～44、每組 3 epochs，共 9 × 4 × 5 = 180 個 head 結果。

### 9.2 彙整結果

| 設定 | Mean-ticker F1 | Balanced accuracy | Macro OvR AUC | Log loss |
|---|---:|---:|---:|---:|
| w20 single | 0.3186 ± 0.0184 | 0.3349 ± 0.0159 | 0.5024 ± 0.0125 | 1.1026 ± 0.0079 |
| w20 dual Volume | 0.3160 ± 0.0208 | 0.3340 ± 0.0144 | 0.5007 ± 0.0118 | 1.1048 ± 0.0100 |
| w20 dual SMA | 0.3376 ± 0.0192 | 0.3552 ± 0.0156 | 0.5193 ± 0.0153 | **1.0970 ± 0.0126** |
| w22 single | 0.3253 ± 0.0133 | 0.3395 ± 0.0152 | 0.5055 ± 0.0132 | 1.1040 ± 0.0078 |
| w22 dual Volume | 0.3213 ± 0.0198 | 0.3410 ± 0.0170 | 0.5041 ± 0.0116 | 1.1033 ± 0.0086 |
| w22 dual SMA | 0.3413 ± 0.0215 | 0.3587 ± 0.0183 | **0.5257 ± 0.0178** | 1.0982 ± 0.0121 |
| w24 single | 0.3291 ± 0.0165 | 0.3423 ± 0.0143 | 0.5111 ± 0.0153 | 1.1017 ± 0.0085 |
| w24 dual Volume | 0.3188 ± 0.0193 | 0.3417 ± 0.0115 | 0.5081 ± 0.0125 | 1.1031 ± 0.0090 |
| w24 dual SMA | **0.3424 ± 0.0210** | **0.3607 ± 0.0181** | **0.5257 ± 0.0140** | 1.1005 ± 0.0097 |
| Numeric Logistic | 0.3218 ± 0.0275 | 0.3456 ± 0.0209 | 0.5253 ± 0.0113 | **1.0916 ± 0.0099** |

SMA 的改善不是只由單一 seed 造成。相對同窗口單圖，20／22／24 日 SMA 的 F1 分別在
15/20、14/20、15/20 個 fold-seed 配對勝出；balanced accuracy 為 18/20、17/20、16/20，
AUC 為 18/20、16/20、18/20。四個 fold 的平均 F1 差值也都為正，只有 fold 3 的增益很小。
相反地，Volume 相對單圖的 F1 差值為 −0.0026、−0.0041、−0.0103，沒有可採用的正面證據。

依預先指定的 mean-ticker Macro-F1，`w24_dual_sma` 排名第一；但它只比
`w22_dual_sma` 高 0.0011，兩者 AUC 幾乎完全相同，且 w22 的 log loss 更低。
`w20_dual_sma` 則有最高 accuracy 與最佳 ViT log loss。因此目前能下的結論是「SMA 有幫助」，
不能說 24 日窗口明確優於 20 或 22 日。

選定模型的類別 F1 為 DOWN 0.3415、NEUTRAL 0.2999、UP 0.3882，NEUTRAL 仍最難。
各 held-out 股票的 mean F1 從 META 0.2974 到 GOOGL 0.3669，顯示跨股票差異仍大。
相較 Numeric Logistic，選定 ViT 的 F1 高 0.0206、balanced accuracy 高 0.0151，但 AUC 只高
0.0004，log loss 反而差 0.0089；它改善了硬分類結果，尚未證明機率排序與校準更好。

### 9.3 本輪判斷

1. 保留 K 線＋SMA 作下一輪主要候選；不再投入 K 線＋Volume 的直接 CLS concat。
2. 22 與 24 日視為等效候選；若要降低計算與避免 Validation 過度搜尋，可固定 22 日。
3. Test 繼續 LOCKED。應新增獨立時間 Validation fold，將 best epoch／候選選擇與最終確認分開。
4. 只有當 SMA 在新時間 fold 仍同時改善 F1、balanced accuracy，且 AUC／log loss 不劣於 Numeric Logistic，才值得開啟一次性 Test。

## 10. 綜合判斷

### 已得到支持的判斷

1. 雙圖是否有效取決於第二張圖：Volume 在跨股票二分類與三分類都沒有穩定收益；科技巨頭實驗中的 SMA 則在三個窗口、一共 60 個配對比較中穩定優於同窗口單圖。
2. SMA 是目前最有支持的視覺輸入候選，但其最佳 AUC 仍只與 Numeric Logistic 相當，尚未證明 ViT 能提供可靠且校準良好的方向機率。
3. 在已測試的 frozen backbones 中，ViT-B/16 優於 DINOv3、MaxViT-T 與 MaxViT-S。
4. Last-block fine-tuning 容易在單股 Validation 過度適配，沒有穩定跨股票收益。
5. LoRA r4 在舊三分類、多 probe 協定中曾有平均改善，但未達穩定採用門檻。
6. 最新 shared-head 跨股票二分類中，Frozen 與 LoRA 幾乎沒有差異，而且兩者方向辨識能力都接近隨機。
7. Train-only q33／q67 三分類能避免用 Validation 定義門檻，但 Frozen ViT 的五 seed AUC 與 balanced accuracy 仍輸給 Numeric Logistic，沒有解決表示能力不足。
8. 三分類雙圖在三個窗口的 mean-ticker Macro-F1 都低於單圖；Volume CLS concat 只帶來微小校準改善，沒有穩定分類收益。
9. 科技巨頭實驗沒有支持唯一最佳窗口：20 日有最佳 ViT log loss、22 日有最佳或並列最佳 AUC、24 日只有 F1／balanced accuracy 略高。

### 尚未得到支持的主張

- 不能主張 LoRA 已讓 ViT 穩定變強。
- 不能主張增加股票數量後，ViT 已學會一般化的技術圖形型態。
- 不能主張更大或更新的自然影像 backbone 會改善金融圖表預測。
- 不能用目前約 0.49 的 Macro-F1 單獨宣稱有效，因為隨機但預測比例平衡的分類器也可能得到接近 0.5 的 Macro-F1。
- 不能把科技巨頭 Validation 的小幅領先當成 Test 成效；同一 Validation 同時參與 epoch 與候選選擇，仍需獨立時間 fold 驗證。

## 11. 建議的下一步

1. **暫停增加 epochs／模型大小**：目前瓶頸不是明顯欠擬合，而是 label 與輸入對未來 5 日方向缺乏可分訊號。
2. **停止搜尋 20／22／24 日窗口**：科技巨頭結果仍顯示各指標由不同窗口勝出；後續固定 22 日即可，不再用同一 Validation 微調窗口。
3. **若繼續 label ablation，改測 horizon／波動度標準化**：q33／q67 三分類已完成；下一個有區別性的問題是 10／20 日 horizon，或先以當時可得的歷史波動率標準化 forward return，再用 Train-only 門檻定義 neutral zone。
4. **加入評估門檻**：主指標仍可保留 mean-ticker Macro-F1，但候選模型必須同時滿足 unseen ROC-AUC、balanced accuracy、log loss 不劣於簡單 baseline。
5. **用獨立時間 fold 複驗 SMA**：固定 22 日，直接比較 single 與 dual SMA；不要再加入 Volume 或搜尋更多窗口。選模期間與確認期間必須分開。
6. **保持 Test 鎖定**：先確認 SMA 在新時間 fold 仍同時改善 F1／balanced accuracy，且 AUC／log loss 至少不劣於 Numeric Logistic，再一次性開啟 Test。
7. **模型選擇維持 Frozen ViT-B/16**：在出現跨 seed、跨股票、跨時間的穩定改善前，不更換正式視覺 encoder。

## 12. 報告來源

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
- [Frozen window × input-mode ablation](../data/outputs/experiments/vit_frozen_window_ablation/runs/14c913bcec83cfba/frozen_w16_24/report.md)
- [Frozen Train-only quantile three-class](../data/outputs/experiments/vit_frozen_three_class/runs/14c913bcec83cfba/quantile_q33_q67/report.md)
- [Frozen Train-only quantile three-class（K 線＋Volume）](../data/outputs/experiments/vit_frozen_three_class/runs/14c913bcec83cfba/quantile_q33_q67_dual/report.md)
- [Tech giants：single／Volume／SMA × window](../data/outputs/experiments/vit_tech_giants/runs/a24d19902835b5f7/tech8_q33_q67/report.md)

## 13. 保存狀態

- 所有 Smoke run 輸出已刪除，不作效果證據。
- 報告、圖表、history 與 predictions 保留。
- 每個重要實驗只保留選定策略的 checkpoint；被刪策略在報告中的舊 checkpoint 路徑可能已失效。
- Cross-stock Frozen feature cache 保留，可供後續 Frozen 分類頭與 label ablation 重用；若 label 或樣本 membership 改變，仍需重新建立相符的資料／特徵索引。
- Window ablation 的單圖／Volume Frozen feature cache 約 1.9 GB，保留供相同資料協定的多 seed head 驗證；正式 run 的 checkpoints、predictions、history、圖表與報告約 47 MB。
- Train-only q33／q67 三分類的 15 組 head、predictions、history、圖表與報告約 88 MB；全部沿用上述 Frozen feature cache，沒有建立重複特徵。
- 同協定雙圖的 15 組 head 與報告約 88 MB；Smoke 輸出已刪除，正式雙圖結果保留供單／雙圖配對比較。
- Tech giants 正式 run 約 230 MB、Frozen feature cache 約 422 MB；180 組 head、predictions、history、圖表與報告均保留，Test 產物不存在。
- 2026-10-02 依專案整理決策，已移除完成任務使用的 ViT 訓練 runner、實驗模型、資料準備程式與專用測試；正式 `module_b_encoder` 未變更。
- 2026-10-10 再移除 §7–§9 使用的三個 runner（窗口消融、三分類、科技巨頭）；對應的 `configs/experiments/*.json` 與輸出保留。
- 保留 `render_experiment_report.py`、`visualize_comparison_report.py` 與各實驗的 `*_report.py`，可由既有 `report.json` 重建 Markdown 與圖表。
- 本文件與 `data/outputs/experiments/` 內的原始 JSON／Markdown／圖表是歷史結果依據。若未來要重跑 label horizon、SMA 或 walk-forward 等新問題，應建立新的實驗實作與新輸出命名空間，而不是改寫這批封存結果。
