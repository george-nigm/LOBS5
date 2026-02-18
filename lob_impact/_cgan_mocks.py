"""
Shared mock setup for CGAN (Coletta) ABIDES dependencies.

Import this module BEFORE importing ganmodels/gan_utils.
It mocks problematic ABIDES modules that have import-time side effects
(config.py creates dirs, worldmodel_evaluation imports py_latex, etc.).

Usage:
    import lob_impact._cgan_mocks  # noqa: F401  — registers mocks in sys.modules
    from abides_markets.agents.gan.v2_41 import ganmodels, gan_utils
"""

import os
import sys
import types

import pandas as pd
import numpy as np
from copy import deepcopy

# ============================================================================
# PYTHONPATH for ABIDES modules
# ============================================================================

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_script_dir)

cgan_markets = os.path.join(_parent, 'abides_worldmodel_offline', 'abides-markets')
cgan_core = os.path.join(_parent, 'abides_worldmodel_offline', 'abides-core')

if cgan_markets not in sys.path:
    sys.path.insert(0, cgan_markets)
if cgan_core not in sys.path:
    sys.path.insert(0, cgan_core)

# ============================================================================
# Mock helper
# ============================================================================

def _create_mock_module(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

# ============================================================================
# scripts.ganworldagent.config
# ============================================================================

config_mod = _create_mock_module("scripts.ganworldagent.config", {
    "ROOT_PATH": "/tmp/cgan/",
    "RAW_LOBSTER_DATA": "/tmp/cgan/raw/",
    "L3_DATA_PATH": "/tmp/cgan/converted/",
    "SYNTHETIC_DATA_PATH": "/tmp/cgan/generated/",
    "MODELS_PATH": "/tmp/cgan/models/",
    "REPORTS_PATH": "/tmp/cgan/reports/",
    "PAPER_PLOTS_PATH": "/tmp/cgan/plots/",
    "AVAILABLE_TICKERS": ["GOOG"],
    "AVAILABLE_DATES": [],
    "TRAINING_DATES": [],
    "NONTRAINING_DATES": [],
    "DEFAULT_SEEDS": [42],
    "TICKER": "GOOG",
    "AUTHOR": "",
    "TIME_TO_FILL_CANCEL_KEY": "level",
    "STYLIZED_CONFIG": {"plots": [], "ylims": {}},
    "setup_font": lambda plt_module: None,
    "check_existing_folders": lambda: None,
    "check_space_left_on_disk": lambda: None,
})

# ============================================================================
# scripts / scripts.ganworldagent packages — with __path__ for real submodules
# ============================================================================

scripts_pkg = _create_mock_module("scripts")
scripts_pkg.__path__ = [os.path.join(cgan_markets, 'scripts')]
ganworldagent_pkg = _create_mock_module("scripts.ganworldagent")
ganworldagent_pkg.__path__ = [os.path.join(cgan_markets, 'scripts', 'ganworldagent')]
ganworldagent_pkg.config = config_mod

# ============================================================================
# Stub modules (not needed, but imported transitively)
# ============================================================================

_create_mock_module("scripts.ganworldagent.generate_synthetic_data")
_create_mock_module("scripts.ganworldagent.impact_comparison")
_create_mock_module("abides_core.py_latex")
_create_mock_module("abides_core.py_latex.latex_lib")
_create_mock_module("abides_markets.visualization")
_create_mock_module("abides_markets.visualization.OB_plot")

# ============================================================================
# scripts.ganworldagent.worldmodel_evaluation — real utility functions needed
# by interrarival_time.py (which ganmodels.py imports)
# ============================================================================

from abides_core.utils import str_to_ns, ns_date, fmt_ts


def _ns_time(x):
    return x - ns_date(x)

def _is_ns_mkt_hours(ns, start_time, end_time):
    return (_ns_time(ns) >= str_to_ns(start_time)) & (_ns_time(ns) <= str_to_ns(end_time))

def _restrict_data_mkt_hours(data, start_time="09:30:00", end_time="15:15:00"):
    data = deepcopy(data)
    data["stream"]["stream_df"] = data["stream"]["stream_df"].reset_index(drop=True)
    indices = data["stream"]["stream_df"][
        data["stream"]["stream_df"].timestamp.apply(
            lambda x: _is_ns_mkt_hours(x, start_time, end_time)
        )
    ].index
    data["stream"]["stream_df"] = (
        data["stream"]["stream_df"].loc[indices].reset_index(drop=True)
    )
    data["stream"]["stream_df"]["time"] = data["stream"]["stream_df"].timestamp.apply(fmt_ts)
    data["original_L3"] = (
        data["original_L3"].reset_index(drop=True).loc[indices].reset_index(drop=True)
    )
    data["L1"]["best_bids"] = np.take(data["L1"]["best_bids"], indices, axis=0)
    data["L1"]["best_asks"] = np.take(data["L1"]["best_asks"], indices, axis=0)
    data["L2"]["bids"] = np.take(data["L2"]["bids"], indices, axis=0)
    data["L2"]["asks"] = np.take(data["L2"]["asks"], indices, axis=0)
    return data

def _apply_dict(d, func, *kargs, **kvargs):
    return dict((k, func(v)) for k, v in d.items())

def _concatenate_dict_df(d, key_col_name):
    dfs = []
    for k, dfi in d.items():
        dfi = dfi.copy()
        dfi[key_col_name] = k
        dfs.append(dfi)
    df = pd.concat(dfs)
    return df[[key_col_name] + df.columns[:-1].tolist()]

def _L1_to_mid(L1):
    df = pd.DataFrame(
        np.array([L1["best_bids"][:, 0], L1["best_bids"][:, 1], L1["best_asks"][:, 1]]).T,
        columns=["time", "best_bid", "best_ask"],
    )
    df["mid"] = (df["best_bid"] + df["best_ask"]) / 2
    return df[["time", "mid", "best_bid", "best_ask"]]

def _uniformize_time_minute(df, timecol="time", timestep_size_ns=60 * 1e9):
    grp = df.groupby(
        df[timecol].apply(lambda x: timestep_size_ns * int((x - ns_date(x)) / timestep_size_ns)),
    )
    sub = grp.first()
    sub.index.name = "timestep"
    sub = sub.reindex([
        timestep_size_ns * k
        for k in range(
            int(np.floor(str_to_ns("09:30:00") / timestep_size_ns)),
            int(np.floor(str_to_ns("16:00:00") / timestep_size_ns)),
        )
    ]).ffill()
    return sub

_create_mock_module("scripts.ganworldagent.worldmodel_evaluation", {
    "restrict_data_mkt_hours": _restrict_data_mkt_hours,
    "apply_dict": _apply_dict,
    "concatenate_dict_df": _concatenate_dict_df,
    "L1_to_mid": _L1_to_mid,
    "uniformize_time_minute": _uniformize_time_minute,
    "is_ns_mkt_hours": _is_ns_mkt_hours,
    "ns_time": _ns_time,
})

# scripts.ganworldagent.utils — real module, no mock needed (__path__ handles it)
