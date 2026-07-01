#!/usr/bin/env python3
"""Convert a 64-device-mesh S5 checkpoint into a single-file host-numpy params bundle.

The exp_H2 S5 checkpoints were saved across a 64-device mesh. orbax cannot reshard them onto
1 GPU (rebuilds the saved mesh -> "available devices different from devices used to save"), and
restore_type=np.ndarray fell back to "sharding Got None". ROOT CAUSE = device-count mismatch.

FIX: expose N fake CPU devices (xla_force_host_platform_device_count) BEFORE importing jax, so the
saved mesh reconstructs cleanly on host. Then device_get every leaf to numpy and save a flat .npz
(keys = '/'-joined pytree path). s5_scenario loads this npz directly, bypassing orbax on the GPU run.

Usage:
  python convert_s5_ckpt.py --ckpt <path> --step <int|latest> --out <params.npz> [--ndev 64]
Run on CPU via sbatch (heavy Lustre read); never on a login node.
"""
import os, argparse, sys

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--step', default='latest')
ap.add_argument('--out', required=True)
ap.add_argument('--ndev', type=int, default=64)
args = ap.parse_args()

# MUST set fake device count before importing jax.
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = f"{os.environ.get('XLA_FLAGS','')} --xla_force_host_platform_device_count={args.ndev}".strip()

import numpy as onp
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

print(f"[convert] jax sees {len(jax.devices())} devices: {jax.devices()[:2]}...", flush=True)

mngr = ocp.CheckpointManager(os.path.abspath(args.ckpt), item_names=('state', 'metadata'),
                             options=ocp.CheckpointManagerOptions())
step = mngr.latest_step() if args.step == 'latest' else int(args.step)
print(f"[convert] restoring step={step} from {args.ckpt}", flush=True)

abstract = mngr.item_metadata(step)['state']

# Strategy A: plain restore using on-disk metadata (incl. saved sharding). With ndev fake devices the
# saved mesh reconstructs; then device_get -> numpy. Fall back to restore_type=np.ndarray if it fails.
raw = None
try:
    restored = mngr.restore(step, args=ocp.args.Composite(state=ocp.args.PyTreeRestore()))
    raw = restored['state']
    print("[convert] Strategy A (plain restore on fake mesh) OK", flush=True)
except Exception as e:
    print(f"[convert] Strategy A failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
    base = ocp.checkpoint_utils.construct_restore_args(abstract)
    rargs = jax.tree_util.tree_map(
        lambda ra: ocp.ArrayRestoreArgs(restore_type=onp.ndarray),
        base, is_leaf=lambda x: isinstance(x, ocp.RestoreArgs))
    restored = mngr.restore(step, args=ocp.args.Composite(state=ocp.args.PyTreeRestore(restore_args=rargs)))
    raw = restored['state']
    print("[convert] Strategy B (restore_type=np.ndarray on fake mesh) OK", flush=True)

# Extract params subtree.
params = raw['params'] if (hasattr(raw, '__contains__') and 'params' in raw) else getattr(raw, 'params', raw)

# Flatten to {'/'.join(path): numpy array}.
flat = {}
def _key(path):
    parts = []
    for p in path:
        parts.append(getattr(p, 'key', getattr(p, 'idx', str(p))))
    return '/'.join(str(x) for x in parts)

leaves_with_path = jax.tree_util.tree_flatten_with_path(params)[0]
for path, val in leaves_with_path:
    arr = onp.asarray(jax.device_get(val))
    flat[_key(path)] = arr

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
onp.savez(args.out, **flat)
tot = sum(a.size for a in flat.values())
print(f"[convert] saved {len(flat)} arrays ({tot:,} params) -> {args.out}", flush=True)
print("[convert] DONE", flush=True)
