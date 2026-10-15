# ViT 視覺實驗封存紀錄

> 最後整理：2026-10-10
> 狀態：下列實驗全部結束，Test 均維持鎖定；10-05 以前的訓練 runner 已移除。2026-10-10 的兩個畫法／輸入對照 runner 暫時保留。
> 後續方向：ViT 工作收斂回正式 pipeline 的 AAPL／NVDA／MSFT；50 檔跨股票與 8 檔科技巨頭實驗僅留紀錄，不再延伸。

這是歷史實驗紀錄，不是可直接執行的操作手冊。所有實驗都獨立於正式 pipeline，只使用 OHLCV 產生圖形；正式 `module_b_encoder`、Fusion、新聞與 RAG 均未修改。完整數字、可比性說明與判讀見 [ViT 實驗總結](../../docs/vit_experiments_summary_2026-10-02.md)。

## 目錄內容

| 檔案 | 用途 |
|---|---|
| `render_experiment_report.py` | 入口：`report.json` → 圖表＋`report.md`，依 `protocol` 分派到下列模組 |
| `visualize_comparison_report.py` | 共用繪圖樣式與 backbone comparison 圖表 |
| `cross_stock_report.py` | 跨股票二分類報告 |
| `frozen_window_report.py` | Frozen 窗口 × 單／雙圖消融報告 |
| `frozen_three_class_report.py` | Train-only 分位數三分類報告 |
| `tech_giants_report.py` | 科技巨頭 single／Volume／SMA 報告 |
| `run_pixel_render_comparison.py` | 2026-10-10 像素畫法對照 runner（暫時保留；自行輸出 `report.json`／`report.md`） |
| `run_k20_k60_comparison.py` | 2026-10-10 K20＋K60 對照 runner（暫時保留；重用上一支的分類頭與指標函式） |
| `run_k20_sma_comparison.py` | 2026-10-10 K20＋SMA 對照 runner（暫時保留；需先跑過 K20＋K60，讀其特徵快取） |
| `run_wick_style_comparison.py` | 2026-10-11 影線加粗／換色對照 runner（暫時保留；讀像素畫法實驗的特徵快取） |
| `run_mpl_wick_comparison.py` | 2026-10-12 mplfinance 影線加粗對照 runner（暫時保留；讀像素畫法實驗的特徵快取） |
| `run_mpl_color_comparison.py` | 2026-10-12 mplfinance 配色對照 runner（暫時保留；重用 `run_mpl_wick_comparison.py` 的流程） |
| `run_horizon_period_grid.py` | 2026-10-12 長期間 walk-forward × 天期 × 圖組合 × 分類頭 runner（暫時保留；可續跑，`--tickers` 可擴充） |
| `run_rolling_label_comparison.py` | 2026-10-11 全域 vs 滾動 250 日標籤對照 runner（暫時保留；讀像素畫法實驗的特徵快取與正式 Z_fused） |

## 已封存實驗一覽

| 實驗 | 股票池 | 任務 | 結論 | 總結章節 |
|---|---|---|---|---|
| 雙圖、輔助圖、Frozen／last blocks／LoRA、backbone 比較 | AAPL、NVDA、MSFT（JPM holdout） | 三分類 | 維持 Frozen ViT-B/16；LoRA r4 未過 4/5 seeds 門檻；DINOv3／MaxViT 未勝出 | §3–§5 |
| 跨股票共用分類頭 | 50 檔跨產業（30／10／10） | 二分類 | Frozen 與 LoRA r4 幾乎相同，AUC 低於 0.5 | §6 |
| Frozen 窗口 × 單／雙圖消融 | 同上 | 二分類 | 單張 K 線優於 K 線＋Volume；16～24 日窗口差距很小 | §7 |
| Train-only 分位數三分類 | 同上 | 三分類 | AUC、balanced accuracy 輸給 Numeric Logistic；Volume 雙圖無收益 | §8 |
| 科技巨頭 single／Volume／SMA | 8 檔科技巨頭，4 個股票留出 folds | 三分類 | K 線＋SMA 穩定優於單圖，但 AUC 只與 Numeric Logistic 相當 | §9 |
| 像素精確畫法 vs mplfinance | AAPL、NVDA、MSFT | 三分類 | 像素畫法 0/5 seeds 勝出，黑底明顯更差；畫法不變 | §10 |
| K20＋K60 vs K20＋Volume | AAPL、NVDA、MSFT | 三分類 | K20＋K60 0/5 seeds 勝出；雙圖維持 K 線＋Volume | §11 |
| K20＋SMA vs K20＋Volume | AAPL、NVDA、MSFT | 三分類 | SMA 勝單圖 5/5，但對 Volume 只勝 2/5，視為打平 | §11.4 |
| 影線加粗／換色（3 像素、同寬色棒） | AAPL、NVDA、MSFT | 三分類 | 同寬色棒單圖與 mplfinance 打平，雙圖較差；3 像素無效；畫法不變 | 決議 #105 |
| mplfinance 影線加粗（3 倍、同寬色棒） | AAPL、NVDA、MSFT | 三分類 | 兩種都低於現有圖（單圖 0/5 seeds）；畫法不變 | 決議 #106 |
| mplfinance 配色（紅綠對調、黑白、藍橘） | AAPL、NVDA、MSFT | 三分類 | 都低於現有圖；資訊相同的紅綠對調也差 0.015～0.026，視為雜訊底線 | 決議 #107 |
| 長期間 walk-forward：t+1／t+3／t+5 × 單／雙／三圖 × 分類頭 | AAPL、NVDA、MSFT（另有 12 檔穩健性檢查），2016～2025、7 folds | 三分類 | 所有設定 AUC 0.50～0.52、彼此無差異、不高於 Numeric Logistic；停止在 Frozen CLS 框架內搜尋 | §12 |
| 全域 vs 滾動 250 日標籤門檻 | AAPL、NVDA、MSFT | 三分類（兩種標籤定義） | 95% 的 Validation 標籤相同，AUC 變動在 ±0.01 內；正式標籤不變 | 決議 #103 |

不同 label、股票池與切分協定的 F1 不能直接橫向排名。第一列的早期實驗沒有保留 config，設計細節見總結 §3–§5 與 `docs/data_and_experiments_log.md`。

## 各實驗紀錄

### 跨股票共用分類頭（2026-10-02）

| 項目 | 設定 |
|---|---|
| 股票池 | 50 檔手選跨產業大型美股；30 Train、10 未見 Validation、10 未見 Test |
| 期間 | Train 2016–2021；Validation 2022–2023（分 seen／unseen）；Test 2024–2025 未開啟 |
| 輸入 | 過去 20 個交易日的 224×224 K 線圖與成交量圖，`auto_adjust=True` 的調整後 OHLCV |
| Label | 未來第 5 個交易日相對當日收盤報酬；DOWN／UP 二分類，精確零報酬排除 |
| 策略 | `google/vit-base-patch16-224`；Frozen＋共用 head，對比最後四個 blocks query／value 的 LoRA r4＋共用 head |
| 主指標 | 未見 Validation 股票逐檔 Macro-F1 等權平均 |

Pilot3 為全量資料、seed40、3 epochs；LoRA 約 3 小時 7 分。

| 方法 | Unseen mean-ticker Macro-F1 | Balanced accuracy | ROC-AUC | Log loss |
|---|---:|---:|---:|---:|
| Frozen | **0.4922** | 0.4951 | **0.4909** | 0.7095 |
| LoRA r4 | 0.4908 | 0.4944 | 0.4844 | 0.7136 |
| Numeric Logistic | 0.4803 | **0.4964** | 0.4882 | **0.6935** |

LoRA 只在 4/10 檔 unseen 股票勝出，兩個 ViT 的 AUC 都低於 0.5；沒有證據顯示學到可轉移到未見股票的漲跌訊號，因此沒有開啟 Test，也不繼續多 seed／10 epoch 訓練。

- Config：[`vit_cross_stock.json`](../../configs/experiments/vit_cross_stock.json)（後面兩個實驗沿用同一份股票與日期）
- 原始計畫：[跨股票實驗計畫](../../docs/archive/experiment_plan_vit_cross_stock.md)
- 報告：[Pilot3 report](../../data/outputs/experiments/vit_cross_stock/runs/14c913bcec83cfba/pilot3/report.md)

### Frozen 窗口 × 單／雙圖消融（2026-10-05）

沿用跨股票資料集 `14c913bcec83cfba` 與二分類 label，ViT 全部凍結，只訓練 LayerNorm＋Linear 分類頭。比較 16／18／20／22／24 日窗口 × `single`（K 線）／`dual`（K 線＋Volume，共用同一個 ViT 後串接 CLS），共 10 組、seed40、3 epochs。所有窗口沿用 20 日資料集的共同樣本日期，不因窗口較短而增加樣本；10 檔 Test 股票不做前向推論。

最佳 `w22_single` 的 mean-ticker Macro-F1 為 0.4949、ROC-AUC 0.4950，只比 `w24_single` 高 0.0019。Volume 雙圖在五個窗口都低於單圖（差值 −0.0842～−0.0105）。結論：改用單張 K 線作 Frozen baseline，窗口差距不足以在單一 seed 下判定。

- Config：[`vit_frozen_window_ablation.json`](../../configs/experiments/vit_frozen_window_ablation.json)
- 報告：[frozen_w16_24 report](../../data/outputs/experiments/vit_frozen_window_ablation/runs/14c913bcec83cfba/frozen_w16_24/report.md)
- Frozen CLS feature cache 約 1.9 GB，位於 `data/outputs/experiments/vit_frozen_window_ablation/features/`

### Train-only 分位數三分類（2026-10-05）

重用上一個實驗的 feature cache，不重新畫圖或執行 ViT forward。只用 Train 期間 5 日 forward return 的 q33／q67 定義 DOWN／NEUTRAL／UP（−0.626378%／+1.478116%），Validation 不參與門檻估計。比較 20／22／24 日、seeds 40～44、每組 3 epochs；單圖與 K 線＋Volume 雙圖各跑一輪。

`w22` 單圖五 seed mean-ticker Macro-F1 為 0.3213±0.0112、Macro OvR AUC 0.5117±0.0041；Numeric Logistic 的 balanced accuracy／AUC 為 0.3612／0.5476，全面較佳。雙圖在三個窗口都只有 2/5 seeds 勝過單圖。結論：三分法更符合任務語意，但沒有讓 Frozen ViT 產生可採用的跨股票訊號。

- Config：[`vit_frozen_three_class_quantile.json`](../../configs/experiments/vit_frozen_three_class_quantile.json)（保存的是單圖設定；雙圖 run 只改 `input_mode`）
- 報告：[單圖](../../data/outputs/experiments/vit_frozen_three_class/runs/14c913bcec83cfba/quantile_q33_q67/report.md)、[K 線＋Volume](../../data/outputs/experiments/vit_frozen_three_class/runs/14c913bcec83cfba/quantile_q33_q67_dual/report.md)

### 科技巨頭 single／Volume／SMA（2026-10-05）

| 項目 | 設定 |
|---|---|
| 股票 | AAPL、MSFT、NVDA、ADBE、GOOGL、META、AMZN、AVGO |
| 股票留出 | 4 folds，每 fold 6 檔 Train、2 檔完全未見 Validation；每檔恰好被留出一次 |
| 期間 | Train 2016–2021；Validation 2022–2023；2024 年起不建立 rows、不抽特徵、不評估 |
| Label | 每 fold 只用該 fold Train 股票、Train 日期的 5 日 forward return q33／q67 |
| 候選 | 20／22／24 日 × K 線單圖／K 線＋Volume／K 線＋SMA5／10／20 |
| 規模 | seeds 40～44、每組 3 epochs；4 folds × 3 windows × 3 inputs × 5 seeds = 180 個分類頭 |

Dataset fingerprint `a24d19902835b5f7`，共 16,009 筆 Train／Validation rows，資料只到 2023-12-21。AAPL／MSFT／NVDA／ADBE 沿用既有 adjusted OHLCV；GOOGL／META／AMZN／AVGO 保存 Yahoo Chart API 原始 JSON，再以 `Adj Close / raw Close` 比例調整 OHLC，Volume 保持原值。

K 線＋SMA 在 20／22／24 日都優於同窗口單圖（mean-ticker Macro-F1 +0.0190／+0.0160／+0.0133），Volume 仍無收益。`w24_dual_sma` 為 0.3424±0.0210，只比 `w22_dual_sma` 高 0.0011；其 AUC 0.5257 幾乎等於 Numeric Logistic 0.5253，log loss 1.1005 反而較差。結論：SMA 是目前最有支持的第二張圖，但還不是「ViT 已學會預測漲跌」的證明；同一份 Validation 同時負責 best epoch 與候選選擇，結果可能偏樂觀。

- Config：[`vit_tech_giants.json`](../../configs/experiments/vit_tech_giants.json)
- 報告：[tech8_q33_q67 report](../../data/outputs/experiments/vit_tech_giants/runs/a24d19902835b5f7/tech8_q33_q67/report.md)

### 像素畫法與 K20＋K60 對照（2026-10-10）

兩個實驗共用協定：AAPL／NVDA／MSFT、AAPL 的 `strict_temporal_v1` 日期、每檔 Train-only q33／q67 三分類、Frozen ViT-B/16 CLS（雙圖為兩個 CLS 串接）＋LayerNorm／Dropout／Linear 分類頭、固定 30 epochs 不挑 epoch、seeds 40～44，Test 未開啟。門檻沿用 +0.02 且勝 4/5 seeds。

| 實驗 | 候選 vs 基準 | Macro-F1 差 | 勝出 |
|---|---|---:|---:|
| 像素畫法 | 黑底像素 vs mplfinance（單／雙圖） | −0.0536／−0.0609 | 0/5、0/5 |
| 像素畫法 | 白底像素 vs mplfinance（單／雙圖） | −0.0219／−0.0278 | 0/5、0/5 |
| K20＋K60 | K20＋K60 vs K20＋Volume | −0.0280 | 0/5 |
| K20＋K60 | K20＋K60 vs 單張 K20 | +0.0068 | 3/5 |
| K20＋SMA | K20＋SMA vs K20＋Volume | −0.0131 | 2/5 |
| K20＋SMA | K20＋SMA vs 單張 K20 | +0.0217 | 5/5 |

K20＋K60 實驗因 60 日窗口剔除 Train 最前面 40 天（926 Train）；同一個 K20＋Volume 設定因此由 0.3390 變為 0.3758，顯示 0.03 左右的差距在這個樣本量下不穩定。兩個實驗都沒有改動正式 pipeline。

```bash
python experiments/vision/run_pixel_render_comparison.py   # 已有特徵快取時只重訓分類頭
python experiments/vision/run_k20_k60_comparison.py        # 首次會畫 K60 圖並抽特徵，本機 MPS 約 3～4 分鐘
python experiments/vision/run_k20_sma_comparison.py        # 需先跑上一支；首次約 1～2 分鐘
```

- 報告：[像素畫法](../../data/outputs/experiments/vit_pixel_render/px_vs_mpl_w20/report.md)、[K20＋K60](../../data/outputs/experiments/vit_k20_k60/k20_k60/report.md)、[K20＋SMA](../../data/outputs/experiments/vit_k20_sma/k20_sma/report.md)；各自的 `samples/` 有畫法並排對照圖。

## 重建報告

報告程式不會訓練模型或開啟 Test：

```bash
python experiments/vision/render_experiment_report.py \
  data/outputs/experiments/vit_tech_giants/runs/a24d19902835b5f7/tech8_q33_q67/report.json

python experiments/vision/render_experiment_report.py \
  --all data/outputs/experiments
```

原始 `report.json`、`report.md`、圖表、history 與 predictions 保留在 `data/outputs/experiments/`（不進 git）。所有 Smoke 輸出已刪除，不作效果證據；部分舊報告所記錄的 checkpoint 路徑可能因清理而失效。

## 後續原則

- 新實驗應建立新的 runner 與獨立輸出命名空間，不修改這批結果，也不把已看過的 Validation／Test 用於新設定選擇。
- 主指標可保留 mean-ticker Macro-F1，但候選設定必須同時檢查 balanced accuracy、ROC-AUC、log loss 與 Numeric Logistic baseline；只提高 Macro-F1 不視為學到可靠方向訊號。
