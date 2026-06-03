# LOB-Impact

Market-impact evaluation for generative limit-order-book models — a submodule of **LOBS5**, run
alongside `lob_bench`. LOB-Bench checks whether a model's order flow *looks* realistic; LOB-Impact
checks whether it *reacts* realistically: inject an aggressive metaorder into the model's generated
book and test the price response against known laws — square-root impact (`β ≈ 0.5`), decay toward a
permanent level, Kyle's λ, the propagator, and the Hurst exponent.

Methodology and definitions: `FRAMEWORK.md`. The two reference images are the pipeline diagram
(`1. FRAMEWORK-*.jpg`) and the model leaderboard (`2. models-leaderboard.jpeg`).

## The pipeline (5 steps)

Run the steps in order; each writes timestamped outputs to its own `results/` folder and a full log
to its own `logs/` folder.

| Step | Folder | What it produces | How to run |
|------|--------|------------------|-----------|
| 1. Pick stocks | `1_data_prep/` | `msgs_between` table over the S&P500 + histograms → choose 3 stocks | `bash 1_data_prep/run_msgs_btw.sh` |
| 2. Daily stats | `2_daily_stats/` | per-stock daily High/Low + execution volume (`daily_h_l_<STOCK>.csv`); order-depth calibration | `bash 2_daily_stats/run_daily_stats.sh` |
| 3. Experiments | `3_scenarios/` | injected-metaorder sequences for each model × stock × shape | `bash 3_scenarios/run_experiments.sh smoke` then `… full` |
| 4. Diagnostics | `4_diagnostics/` | master curves, book-update and participation-rate plots; interactive notebook | *(in progress)* |
| 5. Analysis | `5_analysis/beta/`, `5_analysis/decay/` | β (square-root law) and decay (relaxation, Hurst, propagator) | *(in progress)* |

## Two scenario shapes (step 3)

The launcher runs both shapes for every model × stock; in each config you change only **stock, mb,
and model**:

| Config | Shape | Used for |
|--------|-------|----------|
| `config_bet_composition.yaml` | 100 insertions, no cooling | β analysis |
| `config_beta_decay.yaml` | 10 insertions, 100 cooling steps | decay analysis |

## Running it

Data and model checkpoints live on the cluster and are referenced by environment variables, so
nothing is hard-coded to one machine:

```bash
# point at a mounted month of S&P500 data (one subdirectory per ticker)
export DATA_MOUNT=/path/to/mounted/shard

# Step 1 — choose stocks
bash lob_impact/1_data_prep/run_msgs_btw.sh

# Step 2 — daily stats for the chosen stocks
STOCKS="EA NVDA AMD" bash lob_impact/2_daily_stats/run_daily_stats.sh

# Step 3 — fill the MODELS list in 3_scenarios/run_experiments.sh, then:
bash lob_impact/3_scenarios/run_experiments.sh smoke   # quick check (1 combo, tiny sample)
bash lob_impact/3_scenarios/run_experiments.sh full    # the grid
```

Each run prints to the terminal and saves a copy under the step's `logs/`; outputs land under the
step's `results/<name>_<timestamp>/`.

## Requirements

Run from the LOBS5 repository root. LOB-Impact uses the parent project's model and simulator code
(`lob/`, `preproc.py`, `s5/`, and the `Alphatrade` JAX-LOB simulator); it does not duplicate them.
