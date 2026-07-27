"""
One source of truth for how models are named, ordered and coloured in every figure.

Ten figure scripts each carried their own palette dict, and most were incomplete: Figure 7 had no
entry for OW, QR or S5, so those three fell through to the default grey and looked like a different
palette; mid_trajectory.py was missing five models including GDN and S5-120M. A model silently
changing colour between figures is the same class of defect as a model silently missing from one —
the reader cannot tell whether it is the same object.

Scripts load this with `load_style()`, which walks up from the script to find this file, so it works
from `4_diagnostics/` and `5_analysis/beta/` alike without any package/sys.path setup. If the file
cannot be found the caller keeps its own dict, so nothing breaks when a script is copied elsewhere.

CANON is also the canonical ORDER: replay baselines, then mechanical/parametric by increasing
complexity, then the neural generators.
"""

CANON = ['Historic', 'Heuristic', 'Propagator', 'OW', 'CST', 'NMZI', 'Hawkes', 'QR',
         'S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN']

NEURAL = {'S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN'}

COLORS = {
    # replay / mechanical
    'Historic':   '#C0392B',
    'Heuristic':  '#7F8C8D',
    'Propagator': '#8B5E3C',
    'OW':         '#6A1B9A',
    # parametric
    'CST':        '#27AE60',
    'NMZI':       '#117864',
    'Hawkes':     '#D4AC0D',
    'QR':         '#00ACC1',
    # neural
    'S5':         '#5D6D7E',
    'S5_120M':    '#F06292',
    'S5_4k':      '#E67E22',
    'Mamba3':     '#2F5DA3',
    'Mamba3_4k':  '#16A085',
    'GDN':        '#D81B60',
}

REAL = '#111111'      # the real stream / empirical anchor
GREY = '#666666'      # theory and reference curves


def missing(palette):
    """Canonical models absent from `palette` — they would fall back to a default colour."""
    return [m for m in CANON if m not in palette]


def load_style(script_file, up=4):
    """Return this module's COLORS, found by walking up from `script_file`.

    Returns {} if not found, so a caller can fall back to its own dict rather than crash.
    Uses importlib directly, so `core/` needs no __init__.py and nothing touches sys.path.
    """
    import os
    import importlib.util
    d = os.path.dirname(os.path.abspath(script_file))
    for _ in range(up):
        p = os.path.join(d, 'core', 'model_style.py')
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location('_lob_model_style', p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return dict(mod.COLORS)
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return {}
