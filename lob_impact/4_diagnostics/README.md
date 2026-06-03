# Action 4 — Diagnostics & visualization

Visual sanity checks before the final analysis. To be built this/next round.

- **`diagnostics.py`** (to write): master curves, book-update plots, participation-rate control.
  Reuse the metric functions currently in `../5_analysis/run_300_analyze_one.py`
  (`compute_master_curve`, `compute_combined_impact`, `compute_kyle_lambda`) — these should be
  extracted into `../core/impact_metrics.py` (importable as `lob_impact.core.impact_metrics`) so
  both `4_diagnostics` and `5_analysis` import them instead of the digit-folder path.
- **`interactive_explorer.ipynb`** (present): click through model×stock tracks (e.g. S5 on NVDA),
  inspect master curves interactively, or submit a job to render figures.

Consumes scenario outputs from `3_scenarios/run_experiments.sh` (`save_dir/{data_cond,data_gen}`,
`aggressive_indices.csv`).
