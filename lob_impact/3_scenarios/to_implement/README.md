# to_implement — inactive scenarios

Scenario engines not yet wired to the new data/models. They target the OLD data world
(L500 wide book, 24-tok S5) and are incompatible with the current L10 / 26-tok S&P500 data,
so they are parked here until adapted (the way `mamba3_scenario.py` was adapted from the S5 one).

- `0.null_baseline_s5.py` — null/drift counterfactual (S5)
- `1.aggressive_scenario_s5.py`, `1.aggressive_scenario_s5_v3.py` — S5 aggressive (24-tok)
- `3.heuristic_scenario.py` — heuristic price-shift baseline
- `4.aggressive_scenario_cst.py` — Stoikov–Talreja parametric (needs vendored cst.py/param_estimation.py)
- `5v2.aggressive_scenario_cgan.py` — CGAN (needs abides_markets)
- `6.twap_scenario_s5.py` — TWAP passive variant (S5)

To re-activate one: adapt it to the new model/encoding (see CLAUDE.md "Mamba3 integration" for the
pattern), move it back up to `3_scenarios/`, fix its `parent_folder_path` (one extra `dirname` while
it sits one level deeper here), and add a `MODELS` entry in `run_experiments.sh`.

Active scenarios live one level up: `historic_scenario.py`, `mamba3_scenario.py`.
