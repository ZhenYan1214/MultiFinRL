# Module C: Fusion + Validation + RL + Backtest

**Owner:** (fill in name)
**Input:** B's daily vectors (`data/vectors/`), format defined in `docs/data_format.md` section 2; A's labels for classification validation.
**Deliverable:** a system that outputs portfolio recommendations from market state, plus backtest reports (`data/outputs/`), format defined in `docs/data_format.md` section 3.

## Responsibilities

1. Design a Cross-Modal Transformer that fuses H_v, H_t, and H_r into **Z_fused** (`fusion/model.py`).
   Dual-image H_v enters as `[B,2,197,768]`; learned vision-slot embeddings distinguish the
   candlestick and auxiliary image before the two token grids are flattened. Z_fused remains
   `[B,768]`, so validation, decoder, RL, and backtest interfaces do not change.
2. Train the fusion model (`fusion/train.py`) only on the manifest's Train dates, using Validation macro F1 to retain the best checkpoint. Test labels never update or select the Fusion model. Full-parameter training is used today; QLoRA is planned so training stays feasible on a single high-end GPU, but is not yet implemented (`docs/decisions.md` #29/#85).
3. Validate Z_fused quality with a held-out market-sentiment classification task against A's labels (`validation/classifier.py`).
4. Validate Z_fused a second, independent way: predict the day's event types (multi-label) directly from Z_fused and score against `data/labels/event_ground_truth/` (`validation/event_validation_head.py`, `docs/decisions.md` #33/#34/#37). This probe uses the same temporal manifest: Train fits, Validation selects C, and Test is evaluated once; sparse categories without enough positives are reported but not scored. This is a diagnostic/probing classifier, and it is not an optional extra — the formal proposal (p.19, "表徵效能驗證") explicitly requires diagnostic classification testing on Z_fused for both market-sentiment and event accuracy; `classifier.py` covers the sentiment half, this covers the event half. Classification validation checks predictive power for future price direction; this checks whether Z_fused faithfully retains same-day event information — the two measure different things and are read together, not merged.
5. Design the PPO environment and reward function, and train an RL agent on Z_fused (`rl/env.py`, `rl/train_ppo.py`). Includes curriculum learning (`--curriculum`, proposal 3.6 anticipated challenge (2)): trains in stages of increasing rolling volatility, with per-stage timestep budgets weighted by difficulty (an equal split across stages was found to collapse into an "always hold cash" degenerate policy — see `docs/decisions.md` #52–#55 for the full diagnosis). Verified working and on par with the non-curriculum baseline (Sharpe 0.67 both).
6. Run backtests: cumulative return, Sharpe ratio, max drawdown (`backtest/backtest.py`).
7. Compare investment performance with and without RL.
8. Cross-modal interpretability: Integrated Gradients on the trained PPO policy's action distribution (`explainability/integrated_gradients.py`, proposal 3.6 (4)), evaluated on the manifest's Test period with captum + zero-vector baseline; attributions use the actual holding trajectory the agent would see (not a detached hypothetical). Historical results before the strict split remain legacy (`docs/decisions.md` #49, #56, #85).
9. Generative decoder (`decoder/`, proposal 3.4/3.5): QLoRA-finetuned LLaMA-2 that generates `<TREND>`/`<RISK_LEVEL>` + narrative text conditioned on Z_fused as a soft prompt. Only the `L_belief` loss is implemented (`L_align`/`L_ground` are not — see `model.py`'s header for why). First full train+eval run: 100% structural format correctness, loss delta 1.97 vs. an untuned backbone (fine-tuning has a real, measurable effect), 90% RISK_LEVEL accuracy, but only 47% TREND accuracy — consistent with `classifier.py`'s own weak market-direction signal from Z_fused, not a decoder-specific shortfall. Narrative quality itself (LLM-as-judge, `evaluate.py --llm_judge`) has not yet been run. Full history including three real bugs found and fixed during training (GPU not actually being used, unmasked padding tokens causing empty-string generation, LoRA dropout left on during inference): `docs/decisions.md` #57–#64.

## Phase 1 (while waiting on real vectors from B)

Wire up the architecture against synthetic vectors first: generate random H_v / H_t / H_r at the shapes defined in `docs/data_format.md` and confirm the three vectors in produce a `Z_fused` out. Once B's real vectors are available, move on to real training, RL, and backtesting.

## Commands (run from the repo root)

```bash
python -m module_c_fusion.fusion.train --fake --n 32                   # phase 1: synthetic vectors (needs torch)
python -m module_c_fusion.fusion.train --ticker AAPL --weighted        # phase 2: real training, exports Z_fused
python -m module_c_fusion.validation.classifier --ticker AAPL --weighted   # reuses the shared strict split manifest
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
python -m module_c_fusion.fusion.consolidate --ticker AAPL --checkpoint data/outputs/checkpoints/fusion.pt
```

## Notes

- Never use information beyond `future_closes` during backtesting; `future_closes` is only for computing the P&L of positions already taken.
- Phase 1's backtest is intentionally simple (single-stock position sizing or a small multi-asset mix); whether a full PPO backtest is required within phase 1 itself is still open (`docs/decisions.md`).
- Classification, event validation, decoder, PPO, explainability, and backtesting must reuse the chronological manifest generated by Module A. The two boundaries keep a 5-day gap; random shuffling or recalculating a separate split is forbidden.
