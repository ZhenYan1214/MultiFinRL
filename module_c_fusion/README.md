# Module C: Fusion + Validation + RL + Backtest

**Owner:** (fill in name)
**Input:** B's daily vectors (`data/vectors/`), format defined in `docs/data_format.md` section 2; A's labels for classification validation.
**Deliverable:** a system that outputs portfolio recommendations from market state, plus backtest reports (`data/outputs/`), format defined in `docs/data_format.md` section 3.

## Responsibilities

1. Design a Cross-Modal Transformer that fuses H_v, H_t, and H_r into **Z_fused** (`fusion/model.py`).
2. Train the fusion model (`fusion/train.py`). Full-parameter training is used today; QLoRA is planned so training stays feasible on a single high-end GPU, but is not yet implemented (`docs/decisions.md` #29).
3. Validate Z_fused quality with a held-out market-sentiment classification task against A's labels (`validation/classifier.py`).
4. Validate Z_fused a second, independent way: predict the day's event types (multi-label) directly from Z_fused and score against `data/labels/event_ground_truth/` (`validation/event_validation_head.py`, `docs/decisions.md` #33/#34). Classification validation checks predictive power for future price direction; this checks whether Z_fused faithfully retains same-day event information — the two measure different things and are read together, not merged.
5. Design the PPO environment and reward function, and train an RL agent on Z_fused (`rl/env.py`, `rl/train_ppo.py`).
6. Run backtests: cumulative return, Sharpe ratio, max drawdown (`backtest/backtest.py`).
7. Compare investment performance with and without RL.

## Phase 1 (while waiting on real vectors from B)

Wire up the architecture against synthetic vectors first: generate random H_v / H_t / H_r at the shapes defined in `docs/data_format.md` and confirm the three vectors in produce a `Z_fused` out. Once B's real vectors are available, move on to real training, RL, and backtesting.

## Commands (run from the repo root)

```bash
python -m module_c_fusion.fusion.train --fake --n 32                   # phase 1: synthetic vectors (needs torch)
python -m module_c_fusion.fusion.train --ticker AAPL --weighted        # phase 2: real training, exports Z_fused
python -m module_c_fusion.validation.classifier --ticker AAPL --weighted   # classification validation (70/15/15 time split)
python -m module_c_fusion.rl.train_ppo --fake                          # PPO smoke test
python -m module_c_fusion.backtest.backtest --ticker AAPL --strategy rule_based   # backtest
```

Backtest strategies: `buy_and_hold` (baseline), `rule_based` (classification signal, non-RL comparison), `ppo` (RL). It currently allocates a single ticker between the stock and cash — multi-asset allocation across several tickers has not been built yet.

`--weighted` re-weights the training loss (and, in `classifier.py`, uses scikit-learn's `class_weight="balanced"`) by inverse class frequency. Diagnostic testing (`docs/spec_c_accuracy_diagnostics.md`) found this necessary for the model to learn anything measurable from the news input at all, so it is now the recommended default rather than a purely diagnostic flag. `--ablate_news` (train.py only) zeroes out H_t in memory without touching any files on disk, and exists for the same diagnostic comparison.

The base training objective is classification cross-entropy against A's labels via a throwaway linear head; a generative decoder and the three composite losses (alignment, evidence grounding, belief consistency) described in the original proposal are a later addition, not yet implemented — see `docs/decisions.md` #29 for the full list of what's deferred pending advisor sign-off, including cross-modal interpretability (Integrated Gradients on the PPO policy).

## How Z_fused Is Stored

After `train.py` finishes a full run, in addition to saving one file per day (`data/outputs/z_fused/{TICKER}/{date}.npy`, useful for debugging or later interpretability work), it calls `fusion/consolidate.py` automatically to build a single indexed file:

```
data/outputs/z_fused/{TICKER}_index.npz        # dates / z / label / return_next arrays
data/outputs/z_fused/{TICKER}_index.meta.json  # human-readable summary: day count, date range, z_dim, etc.
```

`classifier.py`, `backtest.py`, and `train_ppo.py` all read this index first and only fall back to scanning per-day files if it's missing. Any new downstream task should read `{TICKER}_index.npz` directly rather than rescanning the vector directory. If the index wasn't generated automatically (e.g. after only running `train.py --fake`), rebuild it with:

```bash
python -m module_c_fusion.fusion.consolidate --ticker AAPL
```

## Notes

- Never use information beyond `future_closes` during backtesting; `future_closes` is only for computing the P&L of positions already taken.
- Phase 1's backtest is intentionally simple (single-stock position sizing or a small multi-asset mix); whether a full PPO backtest is required within phase 1 itself is still open (`docs/decisions.md`).
- Both classification validation and backtesting must use a chronological train/val/test split, never a random shuffle — shuffling would leak future information into training.
