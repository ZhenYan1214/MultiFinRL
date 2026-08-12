# MultiFinRL

**A Multimodal Retrieval-Augmented Financial Decision Framework with Reinforcement Learning**

MultiFinRL turns three kinds of daily market data — candlestick charts (visual), financial news (text), and filings / earnings-call transcripts (external knowledge) — into a single vector, `Z_fused`, that represents a stock's market state on a given trading day. A vision encoder (ViT), a text encoder (FinBERT), and a retrieval-augmented generation (RAG) module each process one modality; a Cross-Modal Transformer fuses the three into `Z_fused`. That vector is then used both to validate market-sentiment classification and event extraction, and as the state input for a PPO reinforcement-learning agent that allocates a portfolio.

---

| Phase | Goal |
|---|---|
| Phase 1 | Build the full data-to-`Z_fused` pipeline; validate `Z_fused` quality via market-sentiment classification and event extraction; run a simple portfolio backtest (single-stock position sizing or a small multi-asset mix). |
| Phase 2 | Connect `Z_fused` to a PPO agent for portfolio allocation; run a full backtest; compare performance with and without RL (Sharpe ratio, max drawdown, etc.). |

Both phases run within the same year and the pipeline is expected to cover the output of both — classification, event extraction, a simple backtest, and an RL-based backtest — not one phase per year.

## Architecture

```mermaid
flowchart LR
    subgraph A["Track A — Data Engineering"]
        A1["OHLCV, charts,\nnews, filings"]
    end
    subgraph B["Track B — Encoding + RAG"]
        B1["H_v (ViT)\nH_t (FinBERT)\nH_r (RAG top-K)"]
    end
    subgraph C1["Track C — Fusion"]
        C1a["Cross-Modal\nTransformer"]
        C1b["Z_fused"]
    end
    subgraph C2["Track C — Validation, RL, Backtest"]
        C2a["Sentiment /\nevent classification"]
        C2b["PPO portfolio\nagent"]
        C2c["Backtest\n(Sharpe, MDD, return)"]
    end
    A1 --> B1 --> C1a --> C1b
    C1b --> C2a
    C1b --> C2b --> C2c
```

This runs once per trading day for the configured date range, producing one `Z_fused` vector per day — several thousand for a multi-year run on a single stock.

---

## Ownership and Directory Layout

|---|---|---|---|
| A | `module_a_data/` | Data engineering: crawlers, chart generation, text cleaning, chunking, price-movement labels | One JSON record per day (`data/processed/dataset/`) |
| B | `module_b_encoder/` | ViT / FinBERT encoding, RAG index and retrieval, event extraction | Daily `H_v`, `H_t`, `H_r` vectors (`data/vectors/`) in a fixed format |
| C | `module_c_fusion/` | Cross-Modal Transformer, classification validation, PPO, backtesting | A system that outputs portfolio recommendations from market state, plus backtest reports |

Shared code (schema validation, path constants, utilities) lives in `shared/` and is jointly maintained; changes there should be flagged to the other tracks before merging.

## Repository Structure

```
MultiFinRL/
├── README.md                       # this file
├── requirements.txt                 # single source of truth for the dev environment
├── .gitignore
├── configs/
│   └── config.yaml                  # global parameters: tickers, date range, chart/label settings
├── docs/
│   ├── data_format.md               # data contract between A / B / C
│   ├── decisions.md                 # decision log, including open questions
│   ├── data_and_experiments_log.md  # data sources and classification results over time
│   └── conduct_script.md            # copy-paste command cheat sheet, A → B → C
├── samples/
│   ├── DataStruct.example.json      # example of A's output format
│   └── vectors_index.example.json   # example of B's output format
├── shared/
│   ├── schemas.py                   # validates records against the data contract
│   ├── paths.py                     # path constants
│   └── utils.py
├── module_a_data/                   # Track A — data engineering
│   ├── README.md
│   ├── crawler/
│   │   ├── fetch_ohlcv.py           # OHLCV via yfinance
│   │   ├── fetch_news.py            # recent news
│   │   ├── fetch_news_alpaca.py     # historical news backfill via Alpaca News API
│   │   ├── fetch_filings.py         # SEC EDGAR filings
│   │   └── fetch_transcripts.py     # earnings-call transcripts
│   ├── preprocess/
│   │   ├── chart_generator.py       # mplfinance candlestick charts, 20-day window
│   │   ├── text_cleaner.py          # HTML/noise cleanup
│   │   └── chunker.py               # document chunking (≤512 tokens)
│   ├── labeling.py                  # BULLISH / BEARISH / NEUTRAL label generation
│   └── build_dataset.py             # assembles the daily JSON records
├── module_b_encoder/                # Track B — encoders + RAG + event extraction
│   ├── README.md
│   ├── encoders/
│   │   ├── vision_encoder.py        # ViT -> H_v
│   │   └── text_encoder.py          # FinBERT -> H_t
│   ├── rag/
│   │   ├── vector_db.py             # FAISS index over filing/transcript chunks
│   │   └── retriever.py             # top-K retrieval -> H_r
│   ├── event_extraction.py          # event extraction: keyword (default) or --method llm, spec_b_event_extraction_llm.md
│   ├── llm_client.py                 # shared LLM-calling helpers (claude/openai/deepseek)
│   └── generate_vectors.py          # main entry point: produces H_v / H_t / H_r per day
├── module_c_fusion/                 # Track C — fusion, validation, RL, backtest
│   ├── README.md
│   ├── fusion/
│   │   ├── model.py                 # Cross-Modal Transformer
│   │   ├── train.py                 # trains the fusion model, exports Z_fused
│   │   └── consolidate.py           # merges per-day Z_fused into one index file
│   ├── validation/
│   │   └── classifier.py            # held-out classification test on Z_fused
│   ├── rl/
│   │   ├── env.py                   # PPO environment and reward function
│   │   └── train_ppo.py
│   └── backtest/
│       └── backtest.py              # cumulative return, Sharpe ratio, max drawdown
├── scripts/
│   └── run_pipeline.py              # runs A -> B -> C end to end
└── data/                            # not tracked in git; synced locally/via cloud storage
    ├── raw/                         # Track A's raw inputs (ohlcv, charts, news, filings, transcripts)
    ├── processed/dataset/           # Track A's deliverable: one JSON per day
    ├── vectors/                     # Track B's deliverable: .npy vectors + index JSON
    └── outputs/                     # Track C's output: model checkpoints, Z_fused, backtest reports
```

## Data Flow

Full definition in `docs/data_format.md`; the handoff points are:

1. **A → B**: `data/processed/dataset/{TICKER}/{YYYY-MM-DD}.json` — one record per day, containing the chart path, a news list (with a `days_ago` field), filing/transcript chunks, and the price-movement label.
2. **B → C**: `data/vectors/{TICKER}/{YYYY-MM-DD}/` — `H_v.npy`, `H_t.npy`, `H_r.npy`, plus an `index.json` recording shapes and sources.
3. **C output**: `data/outputs/` — `Z_fused` (per day and as a consolidated index), model checkpoints, and backtest reports.

## Global Specifications

| Item | Spec |
|---|---|
| Market / initial universe | US equities, starting with AAPL; expansion to more large-cap names (e.g. NVDA) is a later step. Indices/ETFs are excluded for now since they have no filings. |
| Data range | 2021-01 onward, continuously extended (currently through 2026-08; see `configs/config.yaml`) |
| Charts | 20-day trailing window, PNG, 224×224, RGB |
| Price-movement label | Return from close to the close 5 trading days later: > +2% → BULLISH, < −2% → BEARISH, otherwise NEUTRAL |
| Text chunking | ≤512 tokens per chunk (FinBERT's input limit) |
| RAG retrieval | top-K = 3 |
| Missing daily news | backfilled from prior days, with a `days_ago` field so the model can weigh relevance |
| Fine-tuning | full-parameter training currently; QLoRA is planned to keep fusion-model training feasible on a single high-end GPU, not yet implemented (`docs/decisions.md` #29) |
| Dev environment | `requirements.txt` in this repo is the single source of truth |

## Current Status and Roadmap

Operational today, on AAPL:

- Data (A): OHLCV, charts, recent news, and historical news (via the Alpaca News API, 2021–2026) are all in place. Filings are fetched via `edgartools`. Earnings-call transcripts are not yet fetched — the crawler exists but has never completed an end-to-end run.
- Encoding (B): ViT and FinBERT encoders and FAISS-based RAG retrieval are working. Event extraction has a 149-day LLM-labeled ground truth (`data/labels/event_ground_truth/`) and two extraction methods: the default keyword rules (precision/recall/F1 0.162 / 0.868 / 0.273 on AAPL) and an LLM-based method (`--method llm`, `docs/spec_b_event_extraction_llm.md`) that measured 0.742 / 0.605 / 0.667 on the same 149-day sample (f1 +144%); a full 1381-day run has completed, with 401 days showing at least one detected event.
- Fusion and validation (C): the Cross-Modal Transformer, held-out classification validation, event validation head (multi-label probe of Z_fused against the same ground truth, micro F1 0.229 on AAPL), PPO training, and backtesting (buy-and-hold / rule-based / PPO strategies) all run end to end. Class-weighted training is the current default after diagnostic testing showed it was necessary for the model to learn anything from the news input at all.

Known gaps, tracked in `docs/decisions.md`, not yet started:

- QLoRA fine-tuning and the three composite training losses (alignment, evidence grounding, belief consistency) described in the original proposal — training currently uses a simpler classification proxy loss instead.
- A generative decoder that turns `Z_fused` into a structured narrative — requires new ground-truth text data from Track A that does not exist yet.
- Cross-modal interpretability (Integrated Gradients on the PPO policy).
- A domain-gap comparison for ViT on candlestick charts vs. its natural-image pretraining (flagged as a question since early on, never run).
- Curriculum learning for PPO training.
- Multi-asset portfolio backtesting — the current backtest allocates between a single stock and cash, not across multiple tickers.

## Getting Started

```bash
git clone <repo-url>
cd MultiFinRL
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

See each module's own `README.md` for how to run it.

## Execution Order

**Phase 1 (parallel)**
- A starts pulling data, delivering an initial 50–100 sample days early.
- B wires up ViT / FinBERT / RAG against synthetic data first — one image or chunk in, one correctly formatted vector out.
- C wires up the Fusion Transformer against synthetic vectors — three vectors in, `Z_fused` out.

**Phase 2 (once A has enough real data)**
- A keeps extending coverage.
- B switches to real data, producing real `H_v` / `H_t` / `H_r`.
- C switches to real vectors and begins actual training, RL, and backtesting.

## Collaboration Guidelines

- `main` stays runnable; work happens on `feat/a-*`, `feat/b-*`, `feat/c-*` branches, merged via PR.
- Data (`data/`) is not committed to git and is synced separately; the repo holds only code and format examples.
- Any change to the **data format** must update `docs/data_format.md` and `shared/schemas.py` first, and be flagged to the other tracks before downstream code changes.
