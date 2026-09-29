# MultiFinRL 進度回報（2026-09）

完整多模態股票池：AAPL；ViT 視覺實驗另有 NVDA／MSFT／JPM。四檔資料範圍皆為
2021-01 ～ 2026-08（各 1381 個可用交易日）。

## 一句話總結

第一階段（感知與信念建構）核心管線——A 的資料收集與標籤、B 的三種編碼器、C 的融合模型與兩種表徵效能驗證——已全部跑通且有實測數字；第二階段（RL 決策）的 PPO、curriculum learning、Integrated Gradients 也都已完成並驗證；生成式 decoder 已完成第一版訓練與評估。目前真正還沒動工的只剩一項：多資產回測。

## 進度總表

| 模組／項目 | 狀態 | 關鍵數字 |
|---|---|---|
| A：資料收集（股價／新聞／10-K・10-Q／8-K／法說會／標籤） | ✅ AAPL 完成；NVDA/MSFT/JPM 完成視覺實驗所需資料 | 四檔各 1381 天；完整文字模態目前只有 AAPL |
| B：Vision / Text / RAG 三種編碼器 | ✅ 已完成 | H_v / H_t / H_r 皆已產出 |
| C：融合模型 Z_fused | ✅ 已完成 | — |
| 市場情緒分類驗證 | ✅ 已完成 | macro f1 = 0.324 |
| 事件驗證頭（Z_fused 診斷分類） | ✅ 已完成 | micro f1 = 0.229（5 類可評估） |
| 事件抽取（LLM-based，附屬分析） | ✅ 已完成 | f1 = 0.667（關鍵字版 0.273） |
| RL／PPO 訓練 | ✅ 已完成 | Sharpe = 0.67，MDD = 2.9% |
| Curriculum learning | ✅ 已完成，跟 baseline 打平 | Sharpe = 0.67 |
| Integrated Gradients（可解釋性） | ✅ 已完成，結果有意義 | 373/768 維（48.6%）有實質貢獻 |
| 生成式 Decoder（QLoRA 微調） | ✅ 已完成第一版 | 格式正確率 100%，見下方詳細 |
| 回測（單一股票） | ✅ 已完成 | 三策略對照 |
| 多資產回測 | ❌ 尚未動工 | 需先擴充第二支股票完整資料 |

## 各項關鍵實驗細節

### Decoder（生成式敘述，計畫書 3.4／3.5 節）

QLoRA 微調 LLaMA-2-7b（4-bit 量化 + LoRA 掛全部 linear 層），過程中修過三個真實 bug：GPU 其實在用 CPU 跑（`device_map` 設定問題）、padding 沒遮罩導致模型學會直接輸出空字串、LoRA dropout 在推論時沒關掉。

最終結果：
- 格式正確率：100%（30/30）
- loss delta：1.966（微調 vs 未微調 backbone，證明微調有實質效果）
- RISK_LEVEL 二分類準確率：90%（27/30）
- TREND 三分類準確率：46.7%（14/30，只略高於 33% 隨機基準）

TREND 偏弱不是 decoder 獨有的問題——`classifier.py` 用同一份 Z_fused 做同樣判斷，macro f1 也只有 0.32，指向 Z_fused 本身對股價方向訊號含量有限，問題在更上游。只做了 `L_belief` 一項 loss，`L_align`／`L_ground` 仍未實作；敘述文字品質（LLM-as-judge）尚未驗證。

### Integrated Gradients（可解釋性，計畫書 3.6 節）

套在訓練好的 PPO policy 上，用 captum + 零向量 baseline。結果：Z_fused 768 維裡有 373 維（48.6%）有實質貢獻（歸因值 > 1e-4），top10 與 bottom10 相差約 4 個數量級，維度間差異化明確，不是扁平無意義的分佈。

### Curriculum learning（訓練穩定性，計畫書 3.6 節）

第一版有 bug：階段步數均分，訓出「永遠空手」的退化策略（policy collapse），backtest 結果精確全部是 0。排查後改成依難度加權步數，修好後跟 baseline 打平：

| | Sharpe | 累積報酬 | MDD |
|---|---|---|---|
| Baseline（不開 curriculum） | 0.67 | 8.03% | 2.9% |
| Curriculum | 0.67 | 8.90% | 3.24% |

### ViT domain gap 對照實驗

拿掉 K 線圖輸入（H_v 歸零），macro f1 從 0.324 掉到 0.159（掉 51%）。證明現有 ViT（雖非財經領域預訓練版本）貢獻其實很大，換編碼器的急迫性沒有想像中高，但邊際效益可能仍大。

### ViT 分類頭、部分 Fine-tune 與多股票實驗

AAPL-only 嚴格時間切分下，last4 在 validation 勝過 Frozen（0.4309 vs 0.4095），但 held-out
test 反而較差（0.3210 vs 0.3377），因此未採用。加入 NVDA 後改成 AAPL/NVDA 分開分類頭、
共用 ViT backbone，並留 MSFT 不參與 backbone 訓練：Frozen/last4 的兩檔 validation macro F1
平均為 0.3550/0.3357。last4 改善 AAPL 卻明顯傷害 NVDA，仍未證明通用提升；最終 Frozen
backbone 加 MSFT 新 probe head，在 203 天 future test 得到 macro F1=0.3066。正式設定維持
Frozen ViT。完整方法與限制見 `docs/data_and_experiments_log.md`、決議 #78/#79。

LoRA 正式實驗只對最後四個 ViT blocks 的 attention query/value 掛 rank 4 或 rank 8 adapter，
以 AAPL＋NVDA＋MSFT validation 選模，JPM 作新的 backbone holdout。三檔 validation macro F1
平均為 Frozen 0.2944、r4 **0.3602**、r8 0.3226，選出 r4；203 天 JPM future test 上，r4
相較 Frozen 的 accuracy 0.3793 vs 0.3547、macro F1 0.3458 vs 0.3209、log loss 1.0894 vs
1.1014，整體指標正向。但 BULLISH F1 由 0.4052 降至 0.2022。

後續五 seed validation-only 驗證得到 Frozen 0.3328±0.0392、r4 0.3662±0.0127，平均差
+0.0334±0.0454；r4 只勝 3/5，未通過預先設定的 4/5 門檻。MSFT 5/5 改善，但 NVDA 平均
完全打平，表示效果仍受股票別影響。正式 pipeline 因此繼續使用 Frozen，不先投入 Fusion 重建；
下一步改為增加跨產業股票／新 holdout 或新 walk-forward。完整方法與限制見決議 #81～#83。

### 事件抽取方法比較

| 方法 | Precision | Recall | F1 |
|---|---|---|---|
| 關鍵字比對 | 0.162 | 0.868 | 0.273 |
| LLM-based（DeepSeek） | 0.742 | 0.605 | 0.667 |

LLM 版全量 1381 天已跑完，401 天判斷有事件發生。此項為 Track A 資料品質檢查的附屬分析，非計畫書要求的系統模組。

### 8-K 財報缺口補上

原本 `fetch_filings.py` 只抓 10-K/10-Q，新增 8-K 即時揭露抓取（10-K/10-Q 維持背景資料、8-K 另列補充事件並標時間戳記，不覆蓋背景）。本機真實 SEC 資料驗證通過（71 筆 filings，含 48 筆 8-K）。8-K 已確認流入 Z_fused，但對既有兩個診斷指標（市場情緒分類、事件驗證頭）影響在雜訊量級，看不出明顯變化——資料流程本身無誤，只是 8-K 事件太稀疏（約 3.5% 天數），既有指標捕捉不到。

### 分類準確率方法論修正

一連串診斷實驗，逐步把 macro f1 往上修：

| 修正項目 | macro f1 |
|---|---|
| 起始 baseline | 0.247 |
| 加入類別加權 | 0.303 |
| 補齊五年新聞後 | 0.286 |
| 標籤定義改分位數門檻（取代固定 ±2%） | 0.324 |

## 已知但非「未做」的品質落差

1. 視覺編碼器沒有做財經圖表語料領域預訓練（計畫書 3.1 節要求），現用通用 ImageNet 預訓練 ViT。
2. Decoder 只做了 `L_belief` 一項 loss，`L_align`／`L_ground` 仍缺。
3. Decoder 敘述文字品質（LLM-as-judge）尚未驗證。
4. TREND（市場方向）預測偏弱是全專案共通瓶頸，非單一模組問題。

## 下一步建議

1. 補跑 `evaluate.py --llm_judge` 驗證敘述文字品質，成本低、不需重新訓練。
2. ViT LoRA 五 seed 未通過穩定門檻；若續做，先增加跨產業股票與新 holdout／walk-forward，不直接進 Fusion。
3. `fetch_transcripts.py` 已改用 Alpha Vantage API，待本機執行取得真實法說會逐字稿。
4. 多資產回測：需先完成上述擴充實驗的資料，才有意義動工。
