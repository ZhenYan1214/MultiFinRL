# MultiFinRL Text Encoder（H_t）實驗總結

> 更新日期：2026-11-02
> 範圍：只整理 Track B 文字編碼器（H_t）的實驗；不包含 H_v、H_r、Fusion、Decoder、PPO 或回測。
> 原則：只看 H_t 本身的線性 probe，不經過 fusion。不同 ground truth、不同切分方式算出的分數不直接橫向比較。股票只有 AAPL。

## 1. 執行摘要

舊版 H_t 把當天所有新聞串成一段、截到 512 token。在 AAPL 1,381 天上實測：99.9% 的天數被截斷，平均只讀到 **7.7%** 的文字，等於只看到前一晚的第一則新聞。

新做法（已實作為 `encoders.text_method: chunk_relevance_max`，見 `docs/decisions.md` #99）：
1. 每則新聞切成 ≤510 token 的段落，各自用 ProsusAI/finbert 編碼，取 masked mean。
2. 只保留內文提到公司名稱或 ticker ≥3 次的段落。
3. 逐維取 max，得到一個 768 維向量，放在 H_t 第 0 列。

主要證據：在 2026-10 用 Haiku 5.5 全量重標的 1,381 天事件 ground truth 上，用時間切分（walk-forward，測試共 875 天）評估，事件 micro F1 從 **0.280 提升到 0.469**（+0.189，95% CI [+0.132, +0.245]）。在重標前的 149 天 ground truth（+0.146）和 1,381 天同日市場反應指標上，結論也一致。

不支持的主張：
- 這個改動**不會**讓 H_t 對未來 5 日漲跌變得比較好預測。所有版本的情緒 probe 都跟隨機差不多。
- 標題過濾（E7 max）、內文過濾（A）、finbert-tone（C）三者之間分不出高下。

## 2. 實驗脈絡與可比性

| 階段 | 日期 | ground truth | 事件 probe 切分 | 用途 |
|---|---|---|---|---|
| E0–E7、bootstrap | 2026-10-26 | 舊版 149 天（Claude 對話標記） | 5-fold CV，10 seeds | 找問題、篩選方向 |
| X1–X3 | 2026-10-29 | 舊版 149 天 | 5-fold CV，10 seeds，加上市場指標 | 掃描組合、挑入圍 |
| N1 | 2026-10-29 | **Haiku 5.5 全量 1,381 天** | **時間切分**（official + walk-forward） | 在獨立資料上確認 |

- 舊 149 天的分數和 N1 的分數**不能互相比較**：標記者、樣本數、切分方式都不同。
- N1 改用時間切分，是因為相鄰幾天的新聞高度重複，隨機 5-fold 可能把幾乎相同的內容同時放進 train 和 test，造成洩漏。
- 「同日市場反應」指標（異常成交量、當日漲跌幅大小、開盤跳空大小）都用 1,381 天、Ridge、7 個半年窗 walk-forward，算 Spearman 相關。它量的是「H_t 有沒有反映今天有事」，不是交易預測，也不需要標記。

## 3. 舊版 H_t 的問題（E0）

### 3.1 截斷

| 指標 | 數值 |
|---|---|
| 每天新聞 | 平均 11.2 則（中位數 10，最多 37），沒有一天是 0 則 |
| 每天串接後的 token 數 | 平均 8,840，中位數 8,144（p10 4,097 / p90 14,471） |
| 被截斷的天數 | 1,379 / 1,381（99.9%） |
| 實際保留的 token 比例 | 平均 7.7%，中位數 6.3% |
| 進到 512 窗口的新聞則數 | 平均 1.35 則 |
| 窗口裡連一則完整新聞都沒有的天數 | 901 天（65%） |
| 第一則是前一天盤後的新聞 | 76.1% |

`days_ago` 在 1,381 天裡全部是 0：往前 7 天回補的機制從來沒有觸發過，`[0 days ago]` 前綴對每則新聞都一樣，沒有帶到資訊。

### 3.2 pooling 與 padding

- 舊版因為幾乎每天都被截滿 512 token，C 的 `model.py` 對 512 列取平均時，padding 的影響很小（有效 token 數跟向量長度的相關係數只有 0.14）。
- 文字變短時（例如只用標題的 E1），這個相關係數會升到 0.996，向量大小幾乎完全由文字長度決定。所以新做法改成只在第 0 列放值，並建議 C 直接取第 0 列。

## 4. 舊 ground truth 上的篩選（149 天，5-fold，10 seeds）

### 4.1 E0–E7

| 版本 | 做法 | 事件 micro F1 | 情緒 walk-forward（隨機 0.324） |
|---|---|---|---|
| E0 masked mean | 現況：串接截 512，token 平均 | 0.386 ± 0.016 | 0.323 |
| E0 cls | 現況：取 [CLS] | 0.335 ± 0.024 | 0.347 |
| E1 | 只用標題 | 0.341 ± 0.021 | 0.333 |
| E2 | 每則新聞分配 token 預算（每則約 51 token） | 0.369 ± 0.011 | 0.304 |
| E3 token mean | 全部段落、依 token 加權平均 | 0.290 ± 0.009 | 0.287 |
| E3 chunk max | 全部段落、逐維 max | 0.405 ± 0.016 | 0.317 |
| E7 mean | 標題相關過濾、平均 | 0.389 ± 0.014 | 0.308 |
| E7 max | 標題相關過濾、逐維 max | 0.478 ± 0.011 | 0.297 |

- 全部段落直接平均時，訊號會被稀釋：每天的向量跟全期平均的 cosine 達 0.985，跨天差異只剩 E0 的 1/2.7。
- 標題過濾使用 repo 既有的 `is_company_relevant()`，會保留 50.5% 的新聞。

### 4.2 X1：過濾 × 取段 × 聚合（FinBERT 段落，36 組）

前 6 名（依事件 F1）：

| 組合 | 事件 F1 | 異常成交量 | 漲跌幅大小 | 跳空大小 |
|---|---|---|---|---|
| body3 / 全部段落 / max | **0.531 ± 0.012** | 0.264 | 0.151 | 0.187 |
| head_or_b3 / 第一段 / max | 0.509 ± 0.023 | 0.256 | 0.139 | 0.191 |
| body1 / 第一段 / max | 0.505 ± 0.028 | 0.273 | 0.152 | 0.238 |
| body1 / 全部段落 / max | 0.502 ± 0.015 | 0.277 | 0.173 | 0.245 |
| body3 / 第一段 / max | 0.500 ± 0.024 | 0.260 | 0.141 | 0.193 |
| head / 全部段落 / max（= E7 max） | 0.478 ± 0.011 | 0.246 | 0.139 | 0.183 |
| 對照：E0 masked mean | 0.386 ± 0.016 | 0.118 | 0.039 | 0.081 |

- `body1` / `body3`：段落內文提到 apple 或 aapl 至少 1 / 3 次。`head`：標題相關。`head_strict`：標題相關，但排除多檔股票的綜合報導。
- 不論哪種過濾，**max 都是最好的聚合方式**；平均和 log-sum-exp（τ=0.3）都明顯比較差。
- 排除綜合報導反而變差（head_strict max 0.437 < head max 0.478）。
- 只取第一段跟取全部段落差不多。

### 4.3 X2：換 encoder

| encoder | 最佳組合 | 事件 F1 | 異常成交量 / 漲跌幅大小 / 跳空大小 |
|---|---|---|---|
| ProsusAI/finbert（含前綴） | masked mean / body3 / max | 0.531 ± 0.012 | 0.264 / 0.151 / 0.187 |
| ProsusAI/finbert，拿掉前綴 | masked mean / body3 / max | 0.526 ± 0.012 | 0.259 / 0.151 / 0.185 |
| yiyanghkust/finbert-tone | cls / body3 / max | 0.545 ± 0.016 | 0.246 / 0.176 / 0.160 |
| BAAI/bge-base-en-v1.5 | masked mean / head / max | 0.458 ± 0.019 | 0.218 / 0.091 / 0.152 |
| FinLang/finance-embeddings-investopedia | masked mean / head / max | 0.470 ± 0.030 | 0.254 / 0.143 / 0.150 |

四個 encoder 都是 768 維、上限 512 token，換上去不需要改 H_t 的 shape。

### 4.4 X3：入圍版本（配對 bootstrap 10,000 次，只重抽評估天數）

| 版本 | 事件 F1 | 對 E0 差值 [95% CI] | 對 E7 max 差值 [95% CI] |
|---|---|---|---|
| E7 max | 0.478 | +0.092 [+0.002, +0.181] | — |
| A：FinBERT body3 max | 0.531 | +0.146 [+0.066, +0.226] | +0.054 [−0.028, +0.135] |
| B：FinBERT body1 max | 0.502 | +0.116 [+0.037, +0.198] | +0.024 [−0.046, +0.092] |
| C：finbert-tone body3 max | 0.545 | +0.159 [+0.071, +0.245] | +0.067 [−0.015, +0.145] |
| D：finbert-tone body1 max | 0.505 | +0.120 [+0.023, +0.208] | +0.027 [−0.060, +0.108] |

市場指標與 E0 逐窗比較（7 窗中贏幾窗）：

| 版本 | 異常成交量 | 漲跌幅大小 | 跳空大小 |
|---|---|---|---|
| A | 7/7 | 6/7 | 5/7 |
| B | 6/7 | 5/7 | 6/7 |
| C | 7/7 | 6/7 | 4/7 |
| D | 7/7 | 6/7 | 4/7 |

## 5. 新 ground truth 上的確認（N1）

### 5.1 ground truth 重標

2026-10-29 用 `claude-haiku-5-5` 重新標記全部 1,381 天，prompt 沿用 `event_ground_truth_prompt.py`。總共約 2,110 萬 input token，花費約 2.4 美金。跟舊標記重疊的 149 天比較如下：

| 類別 | 舊正例 | 新正例 | 一致率 | kappa | 新版 1,381 天正例 |
|---|---|---|---|---|---|
| DIVIDEND | 6 | 8 | 0.987 | 0.850 | 36 |
| EARNINGS | 10 | 9 | 0.966 | 0.719 | 50 |
| GUIDANCE | 7 | 6 | 0.980 | 0.759 | 29 |
| LAWSUIT | 36 | 35 | 0.899 | 0.723 | 345 |
| MA | 1 | 0 | 0.993 | 0.000 | 16 |
| MANAGEMENT_CHANGE | 1 | 7 | 0.960 | 0.241 | 69 |
| PRODUCT_LAUNCH | 15 | 32 | 0.846 | 0.433 | 277 |

PRODUCT_LAUNCH 和 MANAGEMENT_CHANGE 新版明顯偏寬鬆，例如把傳言、CEO 發言、接班後的後續報導都標成事件，所以下面另外報告排除這兩類的「clean」分數。

### 5.2 設定

- **official**：跟 C 的 `event_validation_head` 相同的時間切分（70/15/15、gap 5，test 203 天）。
- **walk-forward（主要）**：7 個半年窗（2023H1–2026H1），每窗只用之前的資料訓練、中間空 5 天，測試共 875 天。
- 配對 block bootstrap：每 10 天一塊、整塊重抽，B=10,000。

### 5.3 結果（micro F1）

| 版本 | official all / clean | walk-forward all / clean |
|---|---|---|
| 隨機高斯特徵 | 0.167 / 0.168 | 0.202 / 0.199 |
| E0 masked mean（舊版） | 0.235 / 0.291 | 0.280 / 0.309 |
| E0 cls | 0.305 / 0.358 | 0.294 / 0.356 |
| E1 只用標題 | 0.310 / 0.358 | 0.397 / 0.435 |
| E2 每則分配 token | 0.305 / 0.336 | 0.374 / 0.403 |
| E3 article mean | 0.402 / 0.430 | 0.445 / 0.482 |
| E3 chunk max | 0.327 / 0.359 | 0.417 / 0.468 |
| E7 mean | 0.345 / 0.339 | 0.421 / 0.464 |
| E7 max | 0.450 / 0.489 | 0.465 / 0.551 |
| **A FinBERT body3 max（採用）** | 0.423 / 0.450 | **0.469 / 0.565** |
| B FinBERT body1 max | 0.352 / 0.400 | 0.454 / 0.504 |
| C finbert-tone body3 max | 0.457 / 0.512 | 0.465 / 0.521 |
| D finbert-tone body1 max | 0.445 / 0.516 | 0.391 / 0.425 |

walk-forward 配對 block bootstrap（差值 [95% CI]）：

| | vs E0（all） | vs E0（clean） | vs E7 max（all） | vs E7 max（clean） |
|---|---|---|---|---|
| E7 max | +0.185 [+0.136, +0.234] | +0.242 [+0.174, +0.309] | — | — |
| A | +0.189 [+0.132, +0.245] | +0.255 [+0.184, +0.324] | +0.004 [−0.031, +0.039] | +0.014 [−0.028, +0.055] |
| B | +0.174 [+0.127, +0.221] | +0.195 [+0.131, +0.261] | −0.011 [−0.047, +0.025] | −0.047 [−0.093, −0.001] |
| C | +0.185 [+0.136, +0.234] | +0.211 [+0.142, +0.279] | +0.001 [−0.041, +0.042] | −0.030 [−0.086, +0.028] |
| D | +0.111 [+0.065, +0.157] | +0.116 [+0.053, +0.176] | −0.074 [−0.118, −0.030] | −0.126 [−0.192, −0.062] |

所有版本對 E0 的 P(差值 ≤ 0) 都 ≤ 0.0003。

各類別的 walk-forward F1：

| 類別 | E0 | E7 max | A |
|---|---|---|---|
| EARNINGS | 0.282 | 0.393 | 0.431 |
| PRODUCT_LAUNCH | 0.288 | 0.397 | 0.377 |
| LAWSUIT | 0.304 | 0.577 | 0.600 |
| GUIDANCE | 0.500 | 0.615 | 0.558 |
| DIVIDEND | 0.393 | 0.591 | 0.571 |
| MANAGEMENT_CHANGE | 0.072 | 0.194 | 0.219 |
| MA | 0.000 | 0.000 | 0.000 |

MA 在測試期間只有 11 個正例，所有版本都學不到。

## 6. 綜合判斷

### 已得到支持的判斷

- **舊版的截斷是真問題**：在新 ground truth 上，連只用標題的 E1（0.397）都贏過舊版（0.280）。
- **「段落級編碼 → 公司相關性過濾 → 逐維 max」明顯比舊版好**，而且在三組獨立證據上結論一致：
  - 舊 149 天 ground truth：+0.146
  - 新 1,381 天 ground truth + 時間切分：+0.189，clean +0.255
  - 1,381 天同日市場反應指標
- 入圍版本是在舊 149 天上挑的，N1 換了標記者、換成時間切分，測試天數大多不在那 149 天裡，所以可以排除「挑選偏差讓新做法看起來比較好」的疑慮。
- **max 比平均好**：平均會把當天少數幾則重要新聞，稀釋在一堆例行報導裡。
- **兩個 FinBERT 都比通用句向量模型好**。

### 尚未得到支持的主張

- 標題過濾（E7 max）、內文 ≥3 次（A）、finbert-tone（C）之間的優劣：差距 ≤ 0.014，CI 都跨 0。
- H_t 能預測未來 5 日漲跌：所有版本的情緒 probe 都跟隨機差不多（walk-forward 0.28–0.35，隨機約 0.324）。
- 換掉 ProsusAI/finbert：finbert-tone 沒有顯著比較好，而且換 encoder 會連帶影響 H_r。
- 新做法在端到端 pipeline（Z_fused、分類、事件驗證頭）上的效果：還沒用新做法跑完整 pipeline。

## 7. 建議的下一步

1. 用新版 `build_dataset` 重建 dataset，再跑完整 `generate_vectors` 和 Phase 1 pipeline，比較 Z_fused 在事件驗證頭上的分數。
2. 請 C 評估 `fusion/model.py` 是否改成直接取 `H_t[0]`。目前對 512 列取平均，等於使用 `H_t[0]/512`，資訊不變，但倍數是人為造成的。
3. 事件資訊在舊 pipeline 融合後流失明顯：H_t 單獨 0.387，Z_fused 0.229（皆為舊 149 天 ground truth）。C 的訓練目標只有 5 日漲跌，若要保留事件資訊，需要加入事件相關的 loss。
4. PRODUCT_LAUNCH、MANAGEMENT_CHANGE 的新標記雜訊多，可考慮用更嚴格的 prompt 或更強的模型重標這兩類。
5. 擴充到 NVDA、MSFT 時，確認公司名稱規則。`text_encoder.COMPANY_NAMES` 目前有 AAPL、NVDA、MSFT。

## 8. 報告來源

原始結果檔都在 `data/outputs/experiments/text_encoder/`（依 `.gitignore` 不進版控，跟 ViT 實驗輸出相同）：

| 檔案 | 內容 |
|---|---|
| `2026-10-26_E0_results.json`、`2026-10-26_E0_token_stats_per_day.csv`、`2026-10-26_E0_robustness.json` | E0 截斷統計、逐日 token 數、置換檢定與 walk-forward |
| `2026-10-26_E1E2_results.json`、`2026-10-26_E012_event_seed_noise.json` | E1、E2 與 10-seed 雜訊估計 |
| `2026-10-26_E3_results.json`、`2026-10-26_E7_results.json` | E3 段落聚合、E7 相關性過濾 |
| `2026-10-26_bootstrap_event_probe.json` | E0 / E3 / E7 配對 bootstrap |
| `2026-10-29_X1_sweep_results.json` | X1 36 組 |
| `2026-10-29_X2_results_*.json` | X2 四個 encoder |
| `2026-10-29_X3_finalists_results.json` | X3 入圍比較 |
| `2026-10-29_gt_compare_old_new.json`、`2026-10-29_event_ground_truth_AAPL_before_haiku55_149days.json`、`2026-10-29_gt_haiku55_run.log` | ground truth 新舊比較、舊版 149 天備份、重標執行紀錄 |
| `2026-10-29_N1_new_gt_probe_results.json`、`2026-10-29_N1_stdout.log` | N1 新 ground truth 確認 |

## 9. 保存狀態

- 正式實作：`module_b_encoder/encoders/text_encoder.py` 的 `encode_news_chunk_max()`，由 `configs/config.yaml` 的 `encoders.text_method`、`text_min_mentions` 控制，舊做法保留為 `concat_truncate`。
- 實驗腳本與段落向量（約 1.2 GB）只保存在負責人本機，沒有放進 repo。
- 這些腳本依賴 2026-11 之前的 `classifier.time_split` 與 5-fold 版本的 `event_validation_head.evaluate_category`，在目前的 repo 上不能直接重跑。之後若要重做，應以 `shared/temporal_split.py` 建立新的評估入口。
