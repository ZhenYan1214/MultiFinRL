# Module C：Fusion + 驗證 + RL + 回測

**負責人：**（填入姓名）
**輸入：** B 的每日向量（`data/vectors/`），格式見 `docs/data_format.md` 第 2 節；A 的標籤（分類驗證用）。
**交付物：** 可依市場狀態輸出投資組合建議的完整系統 + 回測績效報告（`data/outputs/`），格式見 `docs/data_format.md` 第 3 節。

## 職責

1. 設計 Cross-Modal Transformer，把 H_v、H_t、H_r 融合成 **Z_fused**（`fusion/model.py`）
2. 用 QLoRA 微調融合模型，使訓練可在單台 RTX 5090 上執行（`fusion/train.py`）
3. 以市場情緒分類任務驗證 Z_fused 品質，對照 A 產生的標籤計算準確率（`validation/classifier.py`）
4. 設計 PPO 強化學習環境與獎勵函數，把 Z_fused 餵給 RL agent 訓練（`rl/env.py`、`rl/train_ppo.py`）
5. 完成回測：累積報酬、Sharpe Ratio、最大回撤（`backtest/backtest.py`）
6. 比較加入 RL 前後的投資績效變化

## 第一階段（等 B 真實向量期間）

用假向量把架構跑通：以 `docs/data_format.md` 規定的 shape 隨機產生 H_v、H_t、H_r，確認三個向量丟進去能跑出 Z_fused。等 B 的真實向量到位後，才開始正式訓練、接 RL、做回測。

## 指令（在 repo 根目錄執行）

```bash
python -m module_c_fusion.fusion.train --fake --n 32         # 第一階段：假向量測通（需 torch）
python -m module_c_fusion.fusion.train --ticker AAPL         # 第二階段：真實向量訓練 + 輸出 Z_fused
python -m module_c_fusion.validation.classifier --ticker AAPL  # 情緒分類驗證（時間切分 70/15/15）
python -m module_c_fusion.rl.train_ppo --fake                # PPO 測通
python -m module_c_fusion.backtest.backtest --ticker AAPL --strategy rule_based   # 回測
```

回測策略：`buy_and_hold`（baseline）/ `rule_based`（分類訊號，無 RL 對照組）/ `ppo`（RL 組）。
基礎版訓練目標是情緒分類 cross-entropy；QLoRA 與對比損失是後續強化，介面不變。

## 注意事項

- 回測時嚴禁使用 `future_closes` 以外的未來資訊；`future_closes` 只用於計算已成立部位的損益。
- 第一年的回測先做「簡單版」（單股比例增減或少數標的的組合）；是否需在第一年就跑完整 PPO 尚未定案（見 `docs/decisions.md`）。
- 分類驗證與回測都需要按時間切分 train/val/test，不可隨機打散（避免時間洩漏）。
