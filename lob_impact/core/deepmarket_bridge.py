"""
Import bridge for the DeepMarket checkpoints (TRADES / CGAN).

DeepMarket uses generic top-level module names (`constants`, `configuration`,
`utils`, `models`, `preprocessing`). By the time a scenario reaches the model
load, JAX / gymnax / our own tree have already put THEIR `utils` (a plain
module, not a package) into `sys.modules`, so DeepMarket's
`from utils.utils import noise_scheduler` fails with
"'utils' is not a package" even though its root is first on `sys.path`.

`deepmarket_imports()` stashes the colliding `sys.modules` entries for the
duration of the import, puts the DeepMarket root first on `sys.path`, and
restores everything afterwards. DeepMarket's own modules stay loaded (the
unpickler needs them), so we keep them under their real names and only put the
stashed ones back for the rest of the process.
"""

from __future__ import annotations

import contextlib
import os
import sys

# top-level names DeepMarket owns that commonly collide with our tree
COLLIDING = ('constants', 'configuration', 'utils', 'models', 'preprocessing')


@contextlib.contextmanager
def deepmarket_imports(deepmarket_root: str):
    """Context manager: inside it, `import constants` etc. resolve to DeepMarket."""
    stashed = {k: sys.modules.pop(k) for k in list(sys.modules)
               if k in COLLIDING or k.split('.')[0] in COLLIDING}
    if stashed:
        print(f"[deepmarket_bridge] stashed colliding modules: {sorted(stashed)}")

    # Path order alone is NOT enough: DeepMarket's `utils/` has no __init__.py, so it is a
    # NAMESPACE package, and a regular module (our repo root's utils.py) wins over a namespace
    # portion no matter which entry comes first. Drop every sys.path entry that shadows a
    # colliding name with a plain .py module for the duration of the import.
    # NB: an empty sys.path entry means "current directory" — resolve it, or a cwd that
    # holds the shadowing utils.py slips through the filter.
    shadowing = [p for p in sys.path
                 if any(os.path.isfile(os.path.join(p or os.getcwd(), f'{n}.py'))
                        for n in COLLIDING)]
    saved_path = list(sys.path)
    if shadowing:
        print(f"[deepmarket_bridge] hiding shadowing path entries: {shadowing}")
        sys.path[:] = [p for p in sys.path if p not in shadowing]
    sys.path.insert(0, deepmarket_root)
    try:
        yield
    finally:
        sys.path[:] = saved_path
        # keep DeepMarket's modules loaded (the checkpoint unpickler needs them by
        # name), but restore ours for everything else in the process
        dm_loaded = {k: sys.modules[k] for k in list(sys.modules)
                     if k in COLLIDING or k.split('.')[0] in COLLIDING}
        sys.modules.update(stashed)
        for k, v in dm_loaded.items():
            sys.modules.setdefault(k, v)


def load_engine(deepmarket_root: str, ckpt_path: str, engine: str, device):
    """
    Load a DeepMarket Lightning engine ('gan' | 'diffusion') from a checkpoint.
    Returns (engine_instance, dm_constants_module).
    """
    import torch
    with deepmarket_imports(deepmarket_root):
        import constants as dm_cst
        if engine == 'gan':
            from models.gan.gan_engine import GANEngine as Engine
        elif engine == 'diffusion':
            from models.diffusers.diffusion_engine import DiffusionEngine as Engine
        else:
            raise ValueError(f'unknown engine {engine!r}')
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
        try:
            eng = Engine.load_from_checkpoint(ckpt_path, map_location=device)
        finally:
            torch.load = _orig_load
    eng.eval()
    eng.to(device)
    for p in eng.parameters():
        p.requires_grad_(False)
    return eng, dm_cst
