# S5-4k integration status — RESOLVED ✅

**S5 (exp_H2-context-scale, ctx=4000, ~55M params, ckpt j2504167 @102965) now runs end-to-end on GPU.**

## The blocker (was): 64-device-mesh checkpoint
The ckpt was saved across a **64-device mesh**; orbax 0.11.14 could not reshard it onto 1 GPU
(8 failed restore attempts — StandardRestore rebuilt the saved mesh → "available devices different
from devices used to save"; restore_type=np.ndarray → "sharding Got None").

## The fix: convert once on FAKE devices, then graft from npz
1. **`convert_s5_ckpt.py`** — sets `XLA_FLAGS=--xla_force_host_platform_device_count=64` BEFORE importing
   jax, so the saved 64-device mesh reconstructs cleanly on host. Plain orbax restore then succeeds
   (Strategy A); every leaf is `device_get`→numpy and dumped to a flat `.npz` keyed by '/'-joined
   pytree path. Result: **167 arrays, 55,498,861 params** → `<grid>/_ckpt_converted/s5_4k_j2504167_102965_params.npz` (222 MB).
   Run via CPU sbatch (heavy Lustre read), NOT a login node.
2. **`s5_scenario.py`** — npz-graft is the PRIMARY restore path (orbax kept as fallback): flatten the
   freshly-built `params` with `tree_flatten_with_path`, match each leaf to `npz[key]` by the same
   keying, shape-check, unflatten. Enabled via `S5_PARAMS_NPZ=<npz>` env. Validated: **grafted 167/167 params**.
3. **`core/inference_w_insertions_s5.py`** — two exp_H2-vs-mamba3 codebase deltas fixed:
   - exp_H2 `LOBSTER_Dataset` has no `wide_book_files` arg → feed the wide L500 book in via `book_files`.
   - exp_H2 `validation_helpers` has no `_TP_MESH` (tensor-parallel is mamba3-only) → guard to None
     (S5 runs tp_size=1, single-device jax.jit path).

## Validation (smoke, EA-S5_4k-beta/buy, 2026-07-01)
Sane evolving L10 book with real EA prices (mid ≈ 2,044,350 ≈ $204.4) and correct **buy-side impact**
(best ask 2044500→2044700, best bid 2044200→2044300 over 3 insertions). `smoke run complete`, no traceback.

## Launch
```
sbatch --gres=gpu:1 --mem=96G --time=08:00:00 \
  --export=ALL,PER_DAY=1,N_PER_DAY=48,BSZ=8,S5_PARAMS_NPZ=<npz>,STOCKS=EA,ONLY_SHAPE=<beta|relaxation>,ONLY_DIR=<buy|sell> \
  run_experiments.sh full s5_4k
```
**Note:** at ctx=4000, EA days yield only ~20 conditioning windows/day → replacement sampling
(N_PER_DAY=48 > windows) with distinct gen RNG; realistic ceiling ~300–960/side within the 8h wall.
