# 執行順序（照這個順序打，A → B → C）

複製貼上就能跑的操作手冊。跟 README 的差別：README 講架構跟現況，這份只給指令。
日期範圍以 `configs/config.yaml` 的 `date_range` 為準，不在這裡寫死，避免跟設定檔不同步。

## A：資料工程

```bash
py -m module_a_data.crawler.fetch_ohlcv --ticker AAPL
# 抓股價，存成 data/raw/ohlcv/AAPL.csv（日期範圍讀 config.yaml）

py -m module_a_data.preprocess.chart_generator --ticker AAPL --limit 100
# 依股價畫 K 線圖，存成 data/raw/charts/AAPL/{日期}.png（--limit 100 = 先測前 100 天，拿掉就是全部）

py -m module_a_data.crawler.fetch_filings --ticker AAPL --start 2021-01-01
# 抓 SEC 財報（10-K/10-Q，原生 requests + SEC EDGAR 官方 API），存到 data/raw/filings/AAPL/
# 只有這一支，edgartools 候選方案已刪除（decisions.md #41）。

py -m module_a_data.crawler.fetch_news_alpaca --ticker AAPL --start 2021-01-01 --end 2026-08-09
# 新聞（Alpaca News API，decisions.md #27，取代原本的 fetch_news.py/yfinance 只抓近期新聞）。
# 需要 .env 設定 ALPACA_API_KEY / ALPACA_API_SECRET（見檔頭說明）

py -m module_a_data.crawler.fetch_transcripts --ticker AAPL --start 2021-01-01 --end 2026-08-09
# 抓法說會逐字稿（Alpha Vantage API，取代原本不穩定的 foolcalls 第三方爬蟲）
# 需要 .env 設定 ALPHA_VANTAGE_API_KEY；免費方案速率限制較嚴，建議加 --delay 12.0
# 跑失敗不影響其他步驟（transcript_chunks 留空即可）

py -m module_a_data.build_dataset --ticker AAPL --limit 100
# 把以上所有資料彙整成一天一筆的 JSON，存到 data/processed/dataset/AAPL/
# 漲跌標籤門檻只依 Train 報酬分布計算，再固定套用 Validation/Test
# 這一步一定要放在其他 A 指令「之後」，因為它是彙整、不是抓取
```

## B：Encoder + RAG + 事件抽取

```bash
py -m module_b_encoder.generate_vectors --fake --n 10
# 第一次先測：不需要 A 的資料，用隨機數字測存檔格式對不對

py -m module_b_encoder.generate_vectors --ticker AAPL --limit 100
# 正式執行：讀 A 的 JSON，跑 ViT/FinBERT/RAG，產出 H_v/H_t/H_r，存到 data/vectors/AAPL/

py experiments/vision/experiment_vit_adaptation.py --strategies frozen last1 last2 last4 --epochs 10 --patience 3 --evaluate-test
# ViT-only 調整實驗：雙圖 CLS 接三分類頭，嚴格時間切分 + 5 日 gap；validation 選策略後才看 test
# 正式 test 已於 2026-09-24 開啟並記錄，除非改成新的 walk-forward/multi-stock protocol，勿反覆用同一 test 調參

py experiments/vision/experiment_vit_adaptation.py --evaluate-existing
# 不重訓，從既有 checkpoints 重算 validation 勝者與 frozen baseline 的同一份 test 指標

py experiments/vision/experiment_vit_adaptation.py --train-tickers AAPL NVDA --held-out-ticker MSFT --strategies frozen last4 --epochs 10 --patience 3 --evaluate-test
# 多股票 ViT：AAPL/NVDA 各自分類頭、共用 backbone；MSFT 只訓練新 probe head，backbone 不看 MSFT 梯度
# 這份 MSFT test 已於 2026-09-24 開啟；後續調參須換 walk-forward fold 或新的 held-out ticker

mkdir -p data/outputs/experiments/vit_adaptation
set -o pipefail
python -u experiments/vision/experiment_vit_adaptation.py \
  --train-tickers AAPL NVDA MSFT --held-out-ticker JPM \
  --strategies frozen lora_r4 lora_r8 --vit-lr 1e-4 \
  --epochs 10 --patience 3 --batch 8 --evaluate-test \
  2>&1 | tee data/outputs/experiments/vit_adaptation/lora_AAPL_NVDA_MSFT_holdout_JPM.log
# LoRA 正式實驗（2026-09-24 已完成）：只在 ViT 最後四個 blocks 的 query/value 掛 adapter。
# 執行中的 terminal 會逐 epoch 顯示 loss / validation macro F1；另一個 terminal 可用：
# tail -f data/outputs/experiments/vit_adaptation/lora_AAPL_NVDA_MSFT_holdout_JPM.log
# 完成後報告位於 data/outputs/experiments/vit_adaptation/AAPL_NVDA_MSFT_holdout_JPM/run/report.json
# JPM future test 已開啟並記錄；不要再依這份 test 調 LoRA rank/LR 後重跑，後續請換 walk-forward fold 或新股票。

set -o pipefail
python -u experiments/vision/run_vit_multiseed.py \
  --seeds 40 41 42 43 44 \
  --epochs 10 --patience 3 --batch 8 \
  2>&1 | tee data/outputs/experiments/vit_adaptation/vit_lora_multiseed.log
# LoRA r4 穩定性驗證（2026-09-29 已完成）：每個 seed 比較 Frozen vs r4，只使用 validation，未開 JPM test。
# 本次 MPS 訓練約 2 小時 18 分；原 terminal 會即時顯示 seed/strategy/epoch/loss/val_macro_f1。
# 另一個 terminal 可監看：
# tail -f data/outputs/experiments/vit_adaptation/vit_lora_multiseed.log
# 中途 Ctrl+C 後可重跑同一指令；已有完整 report 的 seed 會自動跳過。
# 完成後彙整：data/outputs/experiments/vit_adaptation/AAPL_NVDA_MSFT_holdout_JPM/multiseed_summary/report.json
# seeds 40～44 已完成且 report 齊全，除非做重現性稽核，否則不用重跑。

py -m module_b_encoder.event_extraction --ticker AAPL
# 從新聞/財報文字抽財經事件（獨立於 Z_fused 的分析，不影響 B/C 的向量或訓練），
# 報告存到 data/outputs/metrics/event_extraction_report.json
# --method llm 可切換成 LLM 版（需 API 金鑰，見 docs/spec_b_event_extraction_llm.md）
```

## C：Fusion + 驗證 + RL + 回測

```bash
py -m module_c_fusion.fusion.train --fake --n 32 --epochs 1
# 第一次先測：用假向量測 Cross-Modal Transformer 架構通不通
# fake checkpoint 只會寫到 data/outputs/checkpoints/fusion_fake.pt，不會覆蓋正式 fusion.pt

py -m module_a_data.build_dataset --ticker AAPL
# 先重建 daily JSON 與共用時間切分 manifest；分位數門檻只使用 Train 報酬

# 2026-09-30 已驗證 AAPL manifest 預期 1,381 日與現有 vectors 1,381 日完全一致，
# 且 B 不讀 label/prices，這次只改標籤與切分，所以不必重跑耗時的 generate_vectors。
# 若以後改圖、新聞、文件或 Fusion 回報缺少 manifest 日期，才需重跑：
# py -u -m module_b_encoder.generate_vectors --ticker AAPL

py -m module_c_fusion.fusion.train --ticker AAPL --weighted
# 正式執行：只用 manifest 的 Train 日期更新 Fusion，以 Validation 選最佳 epoch；
# Test label 全程封存。產出每日 Z_fused，跑完自動彙整成
# data/outputs/z_fused/AAPL_index.npz（給下面三支直接讀，不用再逐日掃描）
# --weighted：類別加權，decisions.md #28 定案為預設做法，不加這個旗標分類效果會明顯變差

py -m module_c_fusion.validation.classifier --ticker AAPL --weighted
# 沿用同一 manifest；Validation 選 C，Test 最後評估一次
# 準確率與 macro F1 存到 data/outputs/metrics/classification_report.json

py -m module_c_fusion.validation.event_validation_head --ticker AAPL
# 用 Z_fused 做事件類型多標籤驗證（7 類），也沿用 Train/Validation/Test manifest
# 需要 data/labels/event_ground_truth/AAPL.json

py -m module_c_fusion.rl.train_ppo --ticker AAPL
# 訓練 PPO 投資組合配置 agent，存到 data/outputs/checkpoints/ppo_agent.zip
# --curriculum 開啟 curriculum learning（依滾動波動度分階段訓練，decisions.md #52~#55）

py -m module_c_fusion.explainability.integrated_gradients --ticker AAPL
# 只在 Test 期間對訓練好的 PPO policy 做跨模態歸因（captum Integrated Gradients），
# 需要先有 Z_fused 跟 ppo_agent.zip，輸出到 data/outputs/explainability/

py -m module_c_fusion.backtest.backtest --ticker AAPL --strategy buy_and_hold
# 回測策略一：全程滿倉（baseline 對照組）

py -m module_c_fusion.backtest.backtest --ticker AAPL --strategy rule_based --weighted
# 回測策略二：用分類結果決定持股比例（無 RL 對照組）

py -m module_c_fusion.backtest.backtest --ticker AAPL --strategy ppo
# 回測策略三：用訓練好的 PPO agent 決定持股比例（RL 組，跟上面兩組比較績效）
```

## Decoder：Z_fused → 結構化市場敘述（計畫書 3.4/3.5 節，需 GPU）

```bash
py -m module_c_fusion.decoder.generate_y_belief --ticker AAPL --limit 50
# 先用 --limit 50 測試品質，人工看過幾筆再拿掉 --limit 跑全量（需 DeepSeek API 金鑰）
# 訓練目標存到 data/labels/y_belief/AAPL.json，支援中斷續跑

py -m module_c_fusion.decoder.train --ticker AAPL --epochs 3
# QLoRA 微調 LLaMA-2-7b（4-bit + LoRA 全掛 attention+MLP），需要先有 Z_fused 跟 y_belief
# 跑訓練時不要同時跑其他吃 GPU 的程式（VRAM 預算吃緊，見 decisions.md #63）
# 中途中斷可加 --resume 從上次的 epoch 接續訓練，不用整個重來

py -m module_c_fusion.decoder.evaluate --ticker AAPL --n_generate 30
# 在 test set 上驗證：微調前後 loss 差異、結構化標籤準確率；--llm_judge 可選（會花 API 費用）
```

## pipeline：一次串完 A→B→C（整合階段用，平常各自開發不用跑這個）

```bash
py scripts/run_pipeline.py --fake
# 只驗證 B 產出的格式 C 讀不讀得進去，不是真的分析

py scripts/run_pipeline.py --ticker AAPL
# 真跑一次完整流程：抓資料 → 畫圖 → 彙整 → 產向量 → 訓練 → 驗證 → 回測
# 用 --skip_transcripts 可以跳過法說會抓取那步（foolcalls 套件不穩定時用）
```
