# Decoder 微調（QLoRA）技術重點摘要

> 給簡報/進度報告用的整理版本，詳細技術決策與除錯過程見 `docs/decisions.md`（#57~#59）。

## 目標

讓 LLaMA-2 學會讀取 Z_fused（Module A/B/C 融合後的市場向量），生成對應的市場信念文字
`y_belief`（結構化趨勢標籤 + 風險程度 + 敘述說明），對應計畫書 3.4/3.5 節的 decoder 設計。

## 微調方法：QLoRA

- backbone（LLaMA-2 原始 70 億參數）全程凍結，4-bit 量化載入，不更新。
- 只訓練兩個新增的小部分：
  - Z_fused 轉換成模型輸入前綴的投影層（自己設計，對應計畫書「soft-prompt」概念）
  - LoRA adapter（掛在 backbone 上的小型可訓練補丁）

## LoRA 掛載範圍（依 2026 年業界共識調整）

- 原本只掛在 attention 的 q_proj / v_proj（2 組），已擴大成全部 7 組
  （q/k/v/o_proj + gate/up/down_proj）。
- 原因：MLP 層（gate/up/down_proj）是模型處理知識與推理邏輯的主要部位，只調
  attention 效果有限；全掛之後下游任務表現約提升 1~3 個百分點（業界 benchmark），
  可訓練參數量只從約 0.2% 增加到 1.5%，代價是每個訓練 step 運算量變大、速度變慢。

## 訓練設定

- 學習率 2e-4，搭配 cosine 排程 + 5% warmup（2026 年 QLoRA 微調常見做法，比全程
  固定學習率更穩定）。
- Gradient clipping（max_norm=1.0），避免訓練中途梯度爆掉。
- Train / Val / Test 依時間切分 70/15/15（跟既有的 `classifier.py` 同一套慣例），
  不隨機打散，避免用未來資料訓練、洩漏到過去的評估上。
- Checkpoint 只在驗證集 loss 創新低時才存檔，避免存到過擬合之後、表現反而變差
  的版本。

## 訓練資料（y_belief）的來源

- 計畫書沒有規定 `y_belief` 這份訓練目標要怎麼取得。
- 沿用事件抽取 ground truth 的既有做法（教授已確認這個做法可行），改用 LLM API
  （DeepSeek）生成：讀當天的新聞/財報/法說會片段，讓 LLM 判斷風險程度並寫敘述，
  趨勢標籤直接沿用 A 模組已用未來報酬分位數算好的 label（不讓 LLM 重猜）。
- 目前 AAPL 已生成 1223/1381 天（DeepSeek 帳戶餘額用罄，暫停在此，之後補值可
  再補齊剩餘天數）。

## 目前進度

- 本機 GPU 相容性問題已解決（RTX 5060 Ti 是較新架構，需要 PyTorch nightly 版本
  才能啟用 GPU 加速，原本誤用 CPU 訓練，速度差異很大）。
- 訓練流程健檢已通過（loss 正常下降），train/val 對照機制、學習率排程、LoRA
  全模組掛載已於本次更新補齊。
- 下一步：確認 val loss 是否跟著 train loss 一起下降（排除死記硬背疑慮）；
  之後依負責人指示，用同一套流程比較 Qwen / GLM 等模型。
