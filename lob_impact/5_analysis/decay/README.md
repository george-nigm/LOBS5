# Action 5 — Decay analysis  (Scenario Shape II: beta_decay, i=10, c=100)

Post-peak impact relaxation toward the permanent level (Bouchaud ~2/3). To be built from the
existing code in `../` (moved here as reference) — split out cleanly into this folder.

Reuse (currently in `../run_300_analyze_one.py`, to extract into `../../core/impact_metrics.py`):
- `compute_relaxation_ratio` — I(u=3)/I(u=1), expected → 2/3
- `fit_decay` — power-law decay exponent γ: I(u) = c(1+u)^(-γ)
- `compute_hurst_dfa` — Hurst exponent of order-flow signs (target ≈ 0.7)
- `compute_propagator` — impact memory kernel G(l) ~ l^(-0.5)
- `stability_vote` — asymptotic-stability check; plus the scorecard in `run_300_analyze_one.run()`

Inputs: `beta_decay/` scenario outputs from `3_scenarios/run_experiments.sh` (the 100-cooling tail
is what carries the relaxation signal).
