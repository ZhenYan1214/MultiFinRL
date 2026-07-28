# 執行順序（照這個順序打，A → B → C）

## A：資料工程

```bash
py -m module_a_data.crawler.fetch_ohlcv --ticker AAPL --start 2021-01-01 --end 2025-12-31
# 抓股價，存成 data/raw/ohlcv/AAPL.csv

py -m module_a_data.preprocess.chart_generator --ticker AAPL --limit 100
# 依股價畫 K 線圖，存成 data/raw/charts/AAPL/{日期}.png（--limit 100 = 先測前 100 天，拿掉就是全部）

py -m module_a_data.crawler.fetch_filings --ticker AAPL --start 2021-01-01
# 抓 SEC 財報（10-K/10-Q），存到 data/raw/filings/AAPL/

py -m module_a_data.crawler.fetch_news --ticker AAPL
# 抓新聞（目前只有近期新聞，歷史新聞來源還沒定案，先跳過也不會報錯）

# fetch_transcripts.py：法說會逐字稿來源尚未定案，現在執行會直接報錯，先不要跑

py -m module_a_data.build_dataset --ticker AAPL --limit 100
# 把以上所有資料彙整成一天一筆的 JSON，存到 data/processed/dataset/AAPL/
# 這一步一定要放在其他 A 指令「之後」，因為它是彙整、不是抓取
```

## B：Encoder + RAG + 事件抽取

```bash
py -m module_b_encoder.generate_vectors --fake --n 10
# 第一次先測：不需要 A 的資料，用隨機數字測存檔格式對不對

py -m module_b_encoder.generate_vectors --ticker AAPL --limit 100
# 正式執行：讀 A 的 JSON，跑 ViT/FinBERT/RAG，產出 H_v/H_t/H_r，存到 data/vectors/AAPL/

py -m module_b_encoder.event_extraction --ticker AAPL
# 從新聞/財報文字抽財經事件，報告存到 data/outputs/metrics/event_extraction_report.json
```

## C：Fusion + 驗證 + RL + 回測

```bash
py -m module_c_fusion.fusion.train --fake --n 32 --epochs 1
# 第一次先測：用假向量測 Cross-Modal Transformer 架構通不通

py -m module_c_fusion.fusion.train --ticker AAPL
# 正式執行：讀 B 的向量，訓練融合模型，產出每日 Z_fused，跑完自動彙整成
# data/outputs/z_fused/AAPL_index.npz（給下面三支直接讀，不用再逐日掃描）

py -m module_c_fusion.validation.classifier --ticker AAPL
# 用 Z_fused 做情緒分類驗證，準確率存到 data/outputs/metrics/classification_report.json

py -m module_c_fusion.rl.train_ppo --ticker AAPL
# 訓練 PPO 投資組合配置 agent，存到 data/outputs/checkpoints/ppo_agent.zip

py -m module_c_fusion.backtest.backtest --ticker AAPL --strategy buy_and_hold
# 回測策略一：全程滿倉（baseline 對照組）

py -m module_c_fusion.backtest.backtest --ticker AAPL --strategy rule_based
# 回測策略二：用分類結果決定持股比例（無 RL 對照組）

py -m module_c_fusion.backtest.backtest --ticker AAPL --strategy ppo
# 回測策略三：用訓練好的 PPO agent 決定持股比例（RL 組，跟上面兩組比較績效）
```

## pipeline：一次串完 A→B→C（整合階段用，平常各自開發不用跑這個）

```bash
py scripts/run_pipeline.py --fake
# 只驗證 B 產出的格式 C 讀不讀得進去，不是真的分析

py scripts/run_pipeline.py --ticker AAPL --limit 50
# 真跑一次完整流程：抓資料 → 畫圖 → 彙整 → 產向量 → 訓練 → 驗證 → 回測
```
