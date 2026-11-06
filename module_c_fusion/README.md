# Module C: Fusion + Validation + RL + Backtest

**Owner:** (fill in name)
**Input:** B's daily vectors (`data/vectors/`), format defined in `docs/data_format.md` section 2; A's labels for classification validation.
**Deliverable:** a system that outputs portfolio recommendations from market state, plus backtest reports (`data/outputs/`), format defined in `docs/data_format.md` section 3.

## Responsibilities

1. Fusion layer `CrossModalTransformer` (`fusion/model.py`, a 2-layer × 8-head Transformer encoder): `[CLS ; H_t (caption tokens) ; 394 image tokens ; K RAG-1 definition tokens] → CLS → Z_fused [768]`. The projections P_v, P_t, P_r (Linear + LayerNorm) are shared by L_align, L_ground and the fusion input. RAG-1 retrieval happens inside the model (query = pooled image tokens after P_v, key = definitions after P_t, cosine top-K). α̂ is the last layer's attention from the H_t tokens to the K definition tokens (`docs/decisions.md` #114–#117, #123).
2. Train the fusion layer before the Decoder (`fusion/train.py`) with `λ1·L_align + λ2·L_ground` (`grounding/alpha.py`; proposal formulas unchanged). Batches never contain two days of the same pattern; α* comes from `grounding/alpha_star_rag1.py` (frozen FinBERT cosine, Q13 option A) or `--alpha_star_space projected`. Only Train updates weights; Validation picks the epoch. The report includes input sensitivity (masking H_t, zeroing H_v, Z_fused pairwise cosine). `--mask_ht` is the H_t ablation; `--ce_weight` adds the optional 3-class CE proxy. Use `--tag` so checkpoints, `z_fused_{tag}/` and reports do not overwrite the formal outputs.
3. Validate Z_fused with a pattern probe against the caption pattern (`validation/pattern_probe.py`). The probe reads only the fusion layer trained and exported with H_t masked (`fusion/train.py --mask_ht --tag mask_ht`), because the caption names the pattern and an unmasked Z_fused can read it straight from the text (`docs/decisions.md` #125). The 3-class market-sentiment classifier (`validation/classifier.py`) is kept as an auxiliary reference only, not a main result.
4. (Removed in #130: the event validation head that predicted event types from Z_fused. Z_fused no longer sees news, so the probe could not succeed. `data/labels/event_ground_truth/AAPL.json` — 1,381 days relabeled with Claude Haiku 5.5 — is kept for a possible future check of the Decoder narrative.)
5. Design the PPO environment and reward function, and train an RL agent on Z_fused (`rl/env.py`, `rl/train_ppo.py`). Includes curriculum learning (`--curriculum`, proposal 3.6 anticipated challenge (2)): trains in stages of increasing rolling volatility, with per-stage timestep budgets weighted by difficulty (an equal split across stages was found to collapse into an "always hold cash" degenerate policy — see `docs/decisions.md` #52–#55 for the full diagnosis). Verified working and on par with the non-curriculum baseline (Sharpe 0.67 both).
6. Run backtests: cumulative return, Sharpe ratio, max drawdown (`backtest/backtest.py`).
7. Compare investment performance with and without RL.
8. Cross-modal interpretability: Integrated Gradients on the trained PPO policy's action distribution (`explainability/integrated_gradients.py`, proposal 3.6 (4)), evaluated on the manifest's Test period with captum + zero-vector baseline; attributions use the actual holding trajectory the agent would see (not a detached hypothetical). Historical results before the strict split remain legacy (`docs/decisions.md` #49, #56, #85).
9. Decoder — the main axis of phase 1 (`decoder/`, proposal 3.4/3.5): LLaMA-2 7B (4-bit, frozen) + LoRA, context `[4 prefix tokens from Z_fused ; RAG-2 snippets ; y_belief]`, one generation pass. `decoder/train.py` trains jointly: Z_fused is computed on the fly by the fusion layer, and `λ1·L_align + λ2·L_ground + λ3·L_belief` goes through one backward, so L_belief reaches the fusion layer. Main line `--rag2` (≈1,100–1,300 tokens, needs a GPU); optional `--no_rag2` 512 version for the Mac; `--tiny_random` is a CPU smoke test. `decoder/evaluate.py` reports TREND/RISK accuracy, the pattern-consistency rate (main) and the shuffled-Z_fused check (auxiliary), plus optional LLM-as-judge. Results of the old (v1) decoder are in `docs/decisions.md` #57–#64.
10. Produce a provenance-checked Phase 1 report (`reporting/pipeline_report.py`). The summary shows L_align, L_ground, α̂/α* agreement, RAG-1 hit@1, Z_fused collapse and the masked-H_t pattern probe; the sentiment classifier (selection, Test, confusion matrix) follows as an auxiliary reference (event extraction is no longer part of the report, #130). `scripts/run_pipeline.py` calls it automatically and writes `data/outputs/reports/{YYYYMMDD_HHMMSS}/`; ticker is stored inside the report rather than in the run ID.

The current automated pipeline stops after Fusion training, classification validation, and report generation. Existing PPO, Integrated Gradients, and backtest modules remain available for manual historical checks but are marked **[第二階段 - Pending]** until the multi-asset protocol is finalized.

## Phase 1 (while waiting on real vectors from B)

Wire up the architecture against synthetic vectors first (`--fake`): random H_v / H_t and definitions at the shapes defined in `docs/data_format.md` should produce a `Z_fused` and finite L_align / L_ground. Once B's real vectors are available, move on to real training, RL, and backtesting.

## Commands (run from the repo root)

```bash
python -m module_c_fusion.fusion.train --fake --n 64                   # synthetic vectors
python -m module_c_fusion.grounding.alpha_star_rag1 --ticker AAPL      # optional: cosine version of α* (only for --alpha_star_space frozen)
python -m module_c_fusion.fusion.train --ticker AAPL                   # L_align + L_ground, exports Z_fused
python -m module_c_fusion.fusion.train --ticker AAPL --mask_ht --tag mask_ht   # H_t ablation
python -m module_c_fusion.validation.classifier --ticker AAPL --weighted
python -m module_c_fusion.fusion.train --ticker AAPL --mask_ht --tag mask_ht   # fusion layer without H_t (for the probe)
python -m module_c_fusion.validation.pattern_probe --ticker AAPL                 # reads z_fused_mask_ht by default
python -m module_c_fusion.decoder.train --ticker AAPL --tiny_random --limit 24 --epochs 1   # joint-training smoke test (CPU)
python -m module_c_fusion.decoder.train --ticker AAPL --no_rag2        # optional 512 version (Mac)
python -m module_c_fusion.decoder.train --ticker AAPL                  # main line with RAG-2 (GPU)
python -m module_c_fusion.decoder.evaluate --ticker AAPL --n_generate 30
python -m module_c_fusion.reporting.pipeline_report --ticker AAPL --run_id 20261004_153000
python -m module_c_fusion.rl.train_ppo --fake                          # PPO smoke test
python -m module_c_fusion.backtest.backtest --ticker AAPL --strategy rule_based   # backtest
```

Backtest strategies: `buy_and_hold` (baseline), `rule_based` (classification signal, non-RL comparison), `ppo` (RL). It currently allocates a single ticker between the stock and cash — multi-asset allocation across several tickers has not been built yet.

`--weighted` re-weights the training loss (and, in `classifier.py`, uses scikit-learn's `class_weight="balanced"`) by inverse class frequency. Diagnostic testing (`docs/decisions.md` #28) found this necessary for the model to learn anything measurable from the news input at all, so it is now the recommended default rather than a purely diagnostic flag. `--ablate_news` (train.py only) zeroes out H_t in memory without touching any files on disk, and exists for the same diagnostic comparison.

The fusion layer's base training objective (`fusion/train.py`) is still classification cross-entropy against A's labels via a throwaway linear head, not the three composite losses (alignment, evidence grounding, belief consistency) described in the original proposal. The generative decoder itself (item 9 above) has since been built separately (`decoder/`, its own QLoRA training loop) and implements the `L_belief` loss; `L_align`/`L_ground` remain unimplemented in both the fusion layer and the decoder — see `docs/decisions.md` #29 for the original deferral, and #57–#64 for what was actually built and its results. Curriculum learning and Integrated Gradients (also originally listed as deferred in #29) are likewise done — see items 5 and 8 above.

## How Z_fused Is Stored

After `train.py` finishes a full run, in addition to saving one file per day (`data/outputs/z_fused/{TICKER}/{date}.npy`, useful for debugging or later interpretability work), it calls `fusion/consolidate.py` automatically to build a single indexed file:

```
data/outputs/z_fused/{TICKER}_index.npz        # dates / z / label / return_next arrays
data/outputs/z_fused/{TICKER}_index.meta.json  # split manifest + Fusion checkpoint SHA256 + summary
```

All downstream consumers reuse `data/processed/temporal_splits/{TICKER}.json`. Any new task must reuse that manifest rather than calculate another 70/15/15 boundary. If the index wasn't generated automatically, rebuild it with the exact checkpoint that produced those vectors:

```bash
python -m module_c_fusion.fusion.consolidate --ticker AAPL --checkpoint data/outputs/checkpoints/fusion_AAPL.pt
```

## Notes

- Never use information beyond `future_closes` during backtesting; `future_closes` is only for computing the P&L of positions already taken.
- Phase 1's backtest is intentionally simple (single-stock position sizing or a small multi-asset mix); whether a full PPO backtest is required within phase 1 itself is still open (`docs/decisions.md`).
- Classification, event validation, decoder, PPO, explainability, and backtesting must reuse the chronological manifest generated by Module A. The two boundaries keep a 5-day gap; random shuffling or recalculating a separate split is forbidden.
