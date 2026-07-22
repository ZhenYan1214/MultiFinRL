# Module B：Encoder + RAG + 事件抽取

**負責人：**（填入姓名）
**輸入：** A 的每日 JSON（`data/processed/dataset/`），格式見 `docs/data_format.md` 第 1 節。
**交付物：** `data/vectors/{TICKER}/{YYYY-MM-DD}/` 內的 `H_v.npy`、`H_t.npy`、`H_r.npy` + `index.json`，格式見 `docs/data_format.md` 第 2 節。

## 職責

1. 從 HuggingFace 下載 ViT 預訓練模型，將 K 線圖 PNG 轉成特徵向量 **H_v**（`encoders/vision_encoder.py`）
2. 用 FinBERT 或 LLaMA 將新聞文字轉成特徵向量 **H_t**（`encoders/text_encoder.py`）
3. 用 FAISS 或 Milvus 建向量資料庫，存入財報/法說會 Chunk（`rag/vector_db.py`）
4. 以 H_v 和 H_t 合併成 query，檢索 top-K（K=3）相關文件，產生 **H_r**（`rag/retriever.py`）
5. 比較不同版本 ViT 與不同文字模型，找出最佳組合（實驗結果可寫進論文）
6. 從新聞、財報及法說會文字抽取重大財經事件，評估 Precision / Recall / F1（`event_extraction.py`）
7. 主程式每日產出三個向量（`generate_vectors.py`）

## 第一階段（等 A 樣本期間）

用假資料把架構跑通：確認可以把一張 PNG 丟進去、跑出一個向量、存成 `docs/data_format.md` 規定的格式。假資料可用隨機產生的 224×224 圖片與任意英文財經句子。

## 指令（在 repo 根目錄執行）

```bash
python -m module_b_encoder.generate_vectors --fake --n 10    # 第一階段：不需 GPU/模型，測通儲存格式
python -m module_b_encoder.generate_vectors --ticker AAPL --limit 50   # 第二階段：真實資料
python -m module_b_encoder.event_extraction --ticker AAPL    # 事件抽取（需 A 的資料）
```

程式進入點：`generate_vectors.py` 的 `run_real()` 串起 encoders/ 與 rag/，
要換模型只改 `configs/config.yaml` 的 `encoders` 區塊（shape 定案後不可變）。

## 注意事項

- 向量 shape 一旦定案不可再變；換 encoder 需全員同意（見 data_format.md）。
- `index.json` 需記錄 `encoder` 的 HuggingFace model id 與 `retrieved_chunk_ids`。
- ViT 預訓練是自然圖片，跟 K 線圖有 domain gap，實測時記錄不同版本的比較結果。
- 事件抽取的 ground truth 來源尚未定案（見 `docs/decisions.md` 待確認事項），先完成抽取程式本體。
