# Action 5 — Beta analysis  (Scenario Shape I: bet_composition, i=100, c=0)

Square-root impact law `I ~ (Q/V)^β`, target β ≈ 0.5. To be built from the existing code in
`../` (moved here as reference) — split out cleanly into this folder.

Reuse (currently in `../run_300_analyze_one.py`, to extract into `../../core/impact_metrics.py`):
- `compute_global_beta` — three estimators (origin / free-intercept / ratio)
- `bootstrap_beta` — 95% CI by resampling paired samples
- `compute_kyle_lambda` — per-insertion price impact per share (ticks/share)
- beta variants: I_mid / I_inst / k≥3 / incremental / V_local
- `../run_beta_report.py` — the comprehensive beta report (fix its `lob_impact.analysis.*` import
  to `lob_impact.core.impact_metrics` after extraction)

Inputs: `bet_composition/` scenario outputs from `3_scenarios/run_experiments.sh` + daily stats from
`2_daily_stats/daily_h_l_<STOCK>.csv` (Q/V normalization, Parkinson sigma).
