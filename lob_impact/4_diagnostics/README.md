# Action 4 — Diagnostics & visualization

Visual sanity checks before the final analysis. To be built this/next round.

- **`diagnostics.py`** (to write): master curves, book-update plots, participation-rate control.
  Reuse the metric functions currently in `../5_analysis/run_300_analyze_one.py`
  (`compute_master_curve`, `compute_combined_impact`, `compute_kyle_lambda`) — these should be
  extracted into `../core/impact_metrics.py` (importable as `lob_impact.core.impact_metrics`) so
  both `4_diagnostics` and `5_analysis` import them instead of the digit-folder path.
- **`interactive_explorer.ipynb`** (present): click through `<STOCK>-<MODEL>-<beta|relaxation>`
  tracks, inspect σ-normalized master curves interactively (Plotly hover), and per-sample midprice
  trajectories. Re-pointed at the **new-pipeline grid** `3_scenarios/results/grid/…`; σ comes from
  `2_daily_stats/results/daily_*/daily_h_l_all.csv` (Parkinson, keyed by ticker+date). Filter via
  `STOCKS/MODELS/SHAPES` in the discover cell (default = all 12 leaves).

### Running the explorer (JupyterLab on a compute node — never the login node)
```bash
sbatch lob_impact/4_diagnostics/run_explorer.sbatch          # CPU-only, no GPU needed
tail -f lob_impact/4_diagnostics/logs/explorer_<jobid>.out   # prints the tunnel line + URL
# then on your LAPTOP:  ssh -N -L 8899:<compute-node>:8899 <user>@login40
# open the printed http://127.0.0.1:8899/lab?token=… ; when done: scancel <jobid>
```
`run_explorer.sh` is the same launcher for a quick login-node/dev run (no SLURM).
A static `interactive_explorer_preview.html` (git-ignored) is produced by `nbconvert --to html`
for a zero-setup peek.

Consumes scenario outputs from `3_scenarios/run_experiments.sh` (`save_dir/{data_cond,data_gen}`,
`aggressive_indices.csv`).
