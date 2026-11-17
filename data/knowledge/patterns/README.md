# RAG-1 型態定義知識庫

7 段英文定義，跟 `shared/patterns.py` 的 `PATTERN_IDS` 一一對應（檔名前綴就是順序）。每段分成
Pattern（型態）、Volume（成交量）、Price-volume reading（量價判讀）三節，控制在 512 個 FinBERT token 以內。

- 現況：**草稿**，由 Claude（LLM）撰寫，尚未人工校對（`status: draft`）。
- 校對方式：對照教科書或技術分析資料庫改寫；備用來源 StockCharts ChartSchool（無授權聲明，只參考改寫並註明出處）、
  Hugging Face `propfirmkey/trading-finance-glossary-nlp`（CC BY 4.0，標註出處可直接引用）。校對完把 `status` 改成 `reviewed`、
  `source` 寫上實際出處。
- 改完任何一段都要重建向量與 α*：
  `python -m module_b_encoder.rag.pattern_kb`、`python -m module_c_fusion.grounding.alpha_star_rag1 --ticker AAPL`。
- 決策：docs/decisions.md #112。
