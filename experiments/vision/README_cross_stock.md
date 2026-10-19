# ViT 跨股票二分類實驗（已封存）

> 封存日期：2026-10-02
> 狀態：Pilot3 已完成；Test 維持鎖定；訓練與資料準備 Python 已移除。

這是一份歷史實驗紀錄，不是目前可直接執行的操作手冊。實驗獨立於正式 pipeline，只使用 OHLCV 產生 K 線／成交量雙圖，比較共用分類頭的 Frozen ViT 與 LoRA r4；正式 `module_b_encoder`、Fusion、新聞與 RAG 均未修改。

## 實驗設計

| 項目 | 設定 |
|---|---|
| 股票池 | 50 檔手選跨產業大型美股；30 Train、10 未見 Validation、10 未見 Test |
| Train 期間 | 2016–2021，只用 Train 股票更新參數 |
| Validation 期間 | 2022–2023；分 seen 與 unseen 股票報告 |
| Test 期間 | 2024–2025；本輪未開啟 |
| 圖形輸入 | 過去 20 個交易日的 224×224 K 線圖與成交量圖 |
| 價格處理 | `auto_adjust=True` 的調整後 OHLCV |
| Label | 未來第 5 個交易日相對當日收盤報酬；DOWN／UP 二分類，精確零報酬排除 |
| Backbone | `google/vit-base-patch16-224` |
| 策略 | Frozen backbone＋共用 head；最後四個 blocks query/value 的 LoRA r4＋共用 head |
| 主指標 | 未見 Validation 股票逐檔 Macro-F1 等權平均 |

未見股票不參與梯度更新，股票名稱也不作為模型特徵。切分使用固定日期邊界，未來標籤跨越區間邊界的樣本會被排除。這是二分類任務，不能與舊三分類實驗的 F1 直接比較。

## Pilot3 結果

執行規模為全量資料、seed40、3 epochs。Frozen 特徵快取沿用後訓練約 27 秒；LoRA 約 3 小時 7 分。Test 全程維持鎖定。

| 方法 | Best epoch | Unseen mean-ticker Macro-F1 | Accuracy | Balanced accuracy | ROC-AUC | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 3 | **0.4922** | 0.4971 | **0.4951** | **0.4909** | **0.7095** |
| LoRA r4 | 3 | 0.4908 | **0.4979** | 0.4944 | 0.4844 | 0.7136 |
| Numeric Logistic | — | 0.4803 | **0.5068** | **0.4964** | 0.4882 | **0.6935** |

LoRA 比 Frozen 的主指標低 0.0014，只在 4/10 檔 unseen 股票勝出。兩個 ViT 的 Macro-F1 雖略高於 Numeric Logistic，但 accuracy、balanced accuracy、ROC-AUC 與 log loss 沒有同步改善；第 3 epoch 的 AUC 也都低於 0.5。結論是目前沒有證據顯示 LoRA 或 Frozen ViT 學到可轉移到未見股票的漲跌排序訊號，因此沒有開啟 Test，也不繼續昂貴的多 seed／10 epoch 訓練。

## 結果與報告工具

- 統一判讀：[ViT 實驗總結](../../docs/vit_experiments_summary_2026-10-02.md)
- 方法計畫：[跨股票實驗計畫](../../docs/experiment_plan_vit_cross_stock.md)
- 原始報告：[Pilot3 report](../../data/outputs/experiments/vit_cross_stock/runs/14c913bcec83cfba/pilot3/report.md)
- 固定股票／日期配置：[`configs/experiments/vit_cross_stock.json`](../../configs/experiments/vit_cross_stock.json)

保留的報告程式不會訓練模型或開啟 Test：

```bash
python experiments/vision/render_experiment_report.py \
  data/outputs/experiments/vit_cross_stock/runs/14c913bcec83cfba/pilot3/report.json

python experiments/vision/render_experiment_report.py \
  --all data/outputs/experiments
```

原始 `report.json`、`report.md`、圖表、history 與 predictions 仍保留在 `data/outputs/experiments/`。所有 Smoke 輸出已刪除，不作效果證據；部分舊報告所記錄的 checkpoint 路徑可能因清理而失效。

## 後續原則

若未來要研究 5／10／20 日 horizon、neutral zone、SMA 輸入或 walk-forward，應建立新的實驗 runner 與獨立輸出命名空間。不要修改本批結果，也不要把已看過的舊 Test 用於新設定選擇。
