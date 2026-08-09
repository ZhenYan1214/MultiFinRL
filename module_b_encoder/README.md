# Module B: Encoders + RAG + Event Extraction

**Owner:** (fill in name)
**Input:** A's daily JSON records (`data/processed/dataset/`). Format defined in `docs/data_format.md`, section 1.
**Deliverable:** `H_v.npy`, `H_t.npy`, `H_r.npy`, and `index.json` under `data/vectors/{TICKER}/{YYYY-MM-DD}/`. Format defined in `docs/data_format.md`, section 2.

## Responsibilities

1. Encode each candlestick chart PNG into a feature vector **H_v** with a pretrained ViT from HuggingFace (`encoders/vision_encoder.py`).
2. Encode news text into a feature vector **H_t** with FinBERT or LLaMA (`encoders/text_encoder.py`).
3. Build a FAISS vector index over filing/transcript chunks (`rag/vector_db.py`).
4. Combine H_v and H_t into a query and retrieve the top-K (K=3) most relevant chunks to produce **H_r** (`rag/retriever.py`).
5. Compare different ViT and text-encoder options and record which combination performs best.
6. Extract notable financial events from news, filings, and transcripts, and evaluate precision / recall / F1 once ground truth exists (`event_extraction.py`).
7. Produce all three vectors for every day from a single entry point (`generate_vectors.py`).

## Phase 1 (while waiting on real data from A)

Wire up the architecture against synthetic data first: confirm that a single PNG in produces a single vector out, saved in the format `docs/data_format.md` requires. Synthetic inputs can be randomly generated 224×224 images and arbitrary English financial-sounding sentences.

## Commands (run from the repo root)

```bash
python -m module_b_encoder.generate_vectors --fake --n 10              # phase 1: no GPU/model needed, just checks the output format
python -m module_b_encoder.generate_vectors --ticker AAPL --limit 50   # phase 2: real data
python -m module_b_encoder.event_extraction --ticker AAPL              # event extraction (needs A's data)
```

Entry point: `run_real()` in `generate_vectors.py` wires together `encoders/` and `rag/`. To swap encoders, change the `encoders` block in `configs/config.yaml` (vector shapes are locked in once finalized, see `data_format.md`).

## Notes

- Vector shapes are locked in once finalized; changing an encoder requires agreement across all three tracks.
- `index.json` must record the encoder's HuggingFace model id and the `retrieved_chunk_ids` used.
- ViT is pretrained on natural images, which is a domain gap from candlestick charts. This comparison is flagged in `docs/decisions.md` (#10) but has not actually been run — the encoder is currently used frozen, with no fine-tuning and no benchmark against the gap.
- Event extraction is currently keyword-based (e.g. matching "earnings" to an EARNINGS event) with no semantic understanding. Its ground-truth source and event taxonomy are still undecided (`docs/decisions.md`, open questions table) — build the extraction code first, but treat both the ground truth and the extraction method itself as open work.
