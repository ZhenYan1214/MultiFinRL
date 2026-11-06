# Module B: Encoders + RAG + Event Extraction

**Owner:** (fill in name)
**Input:** A's daily JSON records (`data/processed/dataset/`). Format defined in `docs/data_format.md`, section 1.
**Deliverable:** `H_v.npy`, `H_t.npy` (chart caption), and `index.json` under `data/vectors/{TICKER}/{YYYY-MM-DD}/`; RAG-1 definitions in `data/vectors/_patterns/`; RAG-2 snippets in `data/outputs/rag2/`. Format defined in `docs/data_format.md`, section 2. Architecture: `docs/architecture/architecture.html`; decisions #108–#131.

## Responsibilities

1. Encode the candlestick and auxiliary chart (volume for equities; FOMC dot plot for the current macro configuration) together with one shared-weight
   pretrained ViT (`encoders/vision_encoder.py`). **H_v** preserves both token grids as
   `[2, 197, 768]`; the second image can later be replaced by a technical-indicator chart without
   changing the encoder interface.
2. (The daily **chart caption** is produced by module A, `module_a_data/chart_caption.py`, and read from `data/labels/chart_caption/{TICKER}.json`.)
3. Encode the caption with frozen FinBERT into **H_t** (`encoders/text_encoder.py`, via `generate_vectors.py`). News is no longer encoded (#110).
4. **RAG-1** (`rag/pattern_kb.py`): encode the 7 pattern definitions in `data/knowledge/patterns/` (currently an LLM draft pending human review) into `data/vectors/_patterns/patterns.npz`. Which K definitions a day uses is decided inside the fusion layer.
   **RAG-2** (`rag/rag2.py`): for each day, cut the date-filtered news, 8-Ks, background 10-K/10-Q sections (MD&A, risk factors) and the latest earnings call down to 15–20 short snippets for the Decoder context. No similarity ranking, no loss.
5. Compare different ViT and text-encoder options and record which combination performs best.
6. **Not a proposal-mandated step, kept as a Track A dataset-quality check** — extract notable financial events from A's raw news, filings, and transcripts, and evaluate against ground truth with precision / recall / F1 (`event_extraction.py`, keyword or LLM-based method via `--method`). This is a standalone analysis of A's raw data, separate from the H_v/H_t/Z_fused pipeline, and since #130 it is no longer called by `scripts/run_pipeline.py` (the Z_fused event validation head was removed at the same time). Its only role is confirming A's collected text captures real events; the ground truth now covers all 1,381 AAPL days (relabeled with Claude Haiku 5.5, 657 days with at least one event), so there's no need to re-run it.
7. Produce H_v and H_t for every day from a single entry point (`generate_vectors.py`); existing H_v files are reused when the encoder and chart order match.

## Phase 1 (while waiting on real data from A)

Wire up the architecture against synthetic data first: confirm that two PNGs in produce one H_v out,
saved in the format `docs/data_format.md` requires. Synthetic inputs can be randomly generated
224×224 images and arbitrary English financial-sounding sentences.

## Commands (run from the repo root)

```bash
python -m module_b_encoder.generate_vectors --fake --n 10              # no GPU/model needed, checks the output format
python -m module_b_encoder.generate_vectors --ticker AAPL              # H_v (reused) + H_t (caption)
python -m module_b_encoder.rag.pattern_kb                              # RAG-1 definitions -> FinBERT vectors
python -m module_b_encoder.rag.rag2 --ticker AAPL                      # RAG-2 daily snippets (--date D --show to inspect)
python -m module_b_encoder.event_extraction --ticker AAPL              # event extraction, keyword method (free)
```

Entry point: `run_real()` in `generate_vectors.py` reads the caption answer sheet and wires together `encoders/`. To swap encoders, change the `encoders` block in `configs/config.yaml` (vector shapes are locked in once finalized, see `data_format.md`).

## Notes

- Vector shapes are locked in once finalized; changing an encoder requires agreement across all three tracks.
- `index.json` must record the encoder's HuggingFace model id and the caption labels (`caption.model`, `pattern`, `volume_label`).
- Fake vectors are isolated under `data/vectors/_AAPL_FAKE/` (or the selected ticker's `_..._FAKE`
  namespace) and never overwrite production dates.
- ViT is pretrained on natural images, which is a domain gap from candlestick charts. This comparison is flagged in `docs/decisions.md` (#10) but has not actually been run — the encoder is currently used frozen, with no fine-tuning and no benchmark against the gap.
- Event extraction has two methods, switchable with `--method` (`event_extraction.py`). `keyword` (default, free, `EVENT_KEYWORDS`) has no semantic understanding; measured on AAPL against the 149-day ground truth: precision=0.162, recall=0.868, f1=0.273 — recall is high (rarely misses a real event) but precision is low (plenty of false positives, e.g. it can't reliably tell "discussing" something apart from "announcing" it that day). See `docs/decisions.md` #32 for the full before/after breakdown and root-cause analysis. `llm` calls an LLM API per day (`--provider claude/openai/deepseek`, shared calling logic in `llm_client.py`) using the same prompt and judging standard as the ground truth labeling; supports resumable caching and a `--limit_to_ground_truth` flag to validate cheaply on the 149-day sample before spending on a full run. Design: `docs/spec_b_event_extraction_llm.md`. **Measured on AAPL with `--provider deepseek` on the 149-day sample: precision=0.742, recall=0.605, f1=0.667** — a large improvement over keyword (f1 +144%); precision is far better (the LLM rarely over-triggers) but recall dropped (it's more conservative and misses some real events, notably LAWSUIT: 21/36 caught, and the single MA day: missed entirely). See `docs/decisions.md` #35 for the full per-category breakdown. **Full 1381-day run on AAPL is complete**: 401 days have at least one detected event (LAWSUIT 218, PRODUCT_LAUNCH 129, MANAGEMENT_CHANGE 49, EARNINGS 43, GUIDANCE 38, DIVIDEND 34, MA 10); P/R/F1 is unchanged from the 149-day test since `evaluate()` only ever scores against the ground-truth-covered dates. `module_b_encoder/event_ground_truth_llm.py` and `event_extraction.py --method llm` share the same `llm_client.py`, so ground truth labeling and production extraction stay consistent.
