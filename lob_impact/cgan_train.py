#!/usr/bin/env python
"""
Train CGAN (Coletta) model on LOBSTER data converted to ABIDES format.

Handles all ABIDES dependency setup (module mocking for scripts.ganworldagent.*
that have side effects) and adds WandB logging.

Prerequisites:
    1. Install pytorch-lightning and wandb in the container
    2. Convert LOBSTER CSV to ABIDES pickle:
       python lob_impact/cgan_convert_lobster.py \
           --lobster_dir /home/myuser/data/rawLOBSTER/GOOG/JAN2023/ \
           --save_dir /home/myuser/data/cgan/GOOG/converted/ \
           --ticker GOOG --dates 20230103 20230104 ...

    3. Run this training script:
       python lob_impact/cgan_train.py \
           --input_datadir /home/myuser/data/cgan/GOOG/converted/ \
           --model_base_path /home/myuser/data/cgan/GOOG/models/ \
           --ticker GOOG \
           --epochs 40 --lookback_window 100 --gpus 1

Output:
    {model_base_path}/{TICKER}/NEW_SET_lb{lookback}_{dates}_v2_41/
        checkpoints/model_{epoch}.ckpt
        data__scalers.pickle
        interarrival_times
"""

import argparse
import os
import sys
import types
import warnings

# ============================================================================
# Step 1: Set up PYTHONPATH for ABIDES modules
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(script_dir)

# abides-markets contains: abides_markets/, scripts/ganworldagent/
cgan_markets = os.path.join(parent_folder_path, 'abides_worldmodel_offline', 'abides-markets')
# abides-core contains: abides_core/
cgan_core = os.path.join(parent_folder_path, 'abides_worldmodel_offline', 'abides-core')

sys.path.insert(0, cgan_markets)
sys.path.insert(0, cgan_core)

# ============================================================================
# Step 2: Mock problematic modules BEFORE importing ganmodels
#
# config.py → tries to create dirs at /data1/sascha/ and check disk space
# worldmodel_evaluation.py → imports py_latex, OB_plot, generate_synthetic_data
# These are not needed for training but are imported transitively.
# ============================================================================

def _create_mock_module(name, attrs=None):
    """Create a mock module and register it in sys.modules."""
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

# --- Mock scripts.ganworldagent.config ---
import pandas as pd
import numpy as np
from copy import deepcopy

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

# --- scripts.ganworldagent package ---
scripts_pkg = _create_mock_module("scripts")
scripts_pkg.__path__ = [os.path.join(cgan_markets, 'scripts')]
ganworldagent_pkg = _create_mock_module("scripts.ganworldagent")
ganworldagent_pkg.__path__ = [os.path.join(cgan_markets, 'scripts', 'ganworldagent')]
ganworldagent_pkg.config = config_mod

# --- Mock generate_synthetic_data, impact_comparison (not needed for training) ---
_create_mock_module("scripts.ganworldagent.generate_synthetic_data")
_create_mock_module("scripts.ganworldagent.impact_comparison")

# --- Mock abides_core.py_latex (may not exist or not needed) ---
_create_mock_module("abides_core.py_latex")
_create_mock_module("abides_core.py_latex.latex_lib")

# --- Mock abides_markets.visualization (not needed for training) ---
_create_mock_module("abides_markets.visualization")
_create_mock_module("abides_markets.visualization.OB_plot")

# --- Mock worldmodel_evaluation with the utility functions that ARE needed ---
# interrarival_time.py uses: restrict_data_mkt_hours, apply_dict, concatenate_dict_df
# These need abides_core.utils which is available (real import)
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

wm_eval_mod = _create_mock_module("scripts.ganworldagent.worldmodel_evaluation", {
    "restrict_data_mkt_hours": _restrict_data_mkt_hours,
    "apply_dict": _apply_dict,
    "concatenate_dict_df": _concatenate_dict_df,
    "L1_to_mid": _L1_to_mid,
    "uniformize_time_minute": _uniformize_time_minute,
    "is_ns_mkt_hours": _is_ns_mkt_hours,
    "ns_time": _ns_time,
})

# --- scripts.ganworldagent.utils: real module (only depends on pandas/numpy) ---
# No mock needed — __path__ on ganworldagent_pkg lets Python find the real utils.py

# ============================================================================
# Step 3: Now safe to import CGAN modules
# ============================================================================

from abides_markets.agents.gan.v2_41 import gan_utils, ganmodels
from argparse import Namespace

# ============================================================================
# Step 4: Training with WandB
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train CGAN (Coletta) model on LOBSTER/ABIDES data"
    )
    parser.add_argument(
        '--input_datadir', type=str, required=True,
        help='Path to ABIDES pickle files (output of cgan_convert_lobster.py)'
    )
    parser.add_argument(
        '--model_base_path', type=str, required=True,
        help='Path where models, scalers, and logs are saved'
    )
    parser.add_argument(
        '--ticker', type=str, default='GOOG',
        help='Stock ticker'
    )
    parser.add_argument(
        '--dates', type=str, nargs='+', default=None,
        help='Training dates (YYYYMMDD). If not specified, uses all dates in input_datadir.'
    )
    parser.add_argument(
        '--epochs', type=int, default=40,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Training batch size'
    )
    parser.add_argument(
        '--lookback_window', type=int, default=100,
        help='Lookback window size for state history'
    )
    parser.add_argument(
        '--gpus', type=int, default=1,
        help='Number of GPUs to use (0 for CPU)'
    )
    parser.add_argument(
        '--wandb_project', type=str, default='cgan-lob',
        help='WandB project name'
    )
    parser.add_argument(
        '--wandb_run_name', type=str, default=None,
        help='WandB run name (default: auto-generated)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug printing'
    )

    args = parser.parse_args()

    # Auto-discover dates if not specified
    if args.dates is None:
        args.dates = sorted([
            d for d in os.listdir(args.input_datadir)
            if os.path.isdir(os.path.join(args.input_datadir, d))
            and os.path.exists(os.path.join(args.input_datadir, d, f"{args.ticker}.pickle"))
        ])
        print(f"Auto-discovered dates: {args.dates}")

    if len(args.dates) == 0:
        print(f"ERROR: No data found for {args.ticker} in {args.input_datadir}")
        sys.exit(1)

    if args.wandb_run_name is None:
        args.wandb_run_name = f"cgan_{args.ticker}_lb{args.lookback_window}_{len(args.dates)}d_ep{args.epochs}"

    print(f"CGAN Training Configuration:")
    print(f"  Ticker:          {args.ticker}")
    print(f"  Input data:      {args.input_datadir}")
    print(f"  Model output:    {args.model_base_path}")
    print(f"  Training dates:  {args.dates}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Lookback window: {args.lookback_window}")
    print(f"  GPUs:            {args.gpus}")
    print(f"  WandB project:   {args.wandb_project}")
    print(f"  WandB run:       {args.wandb_run_name}")

    # Create the LOBGAN model (same as ganmodels.main but with WandB)
    ganmodels.reproducibility()

    model = ganmodels.LOBGAN(
        featureset=gan_utils.FeatureSet.NEW_SET,
        lookback_window=args.lookback_window,
        input_datadir=args.input_datadir,
        model_base_path=args.model_base_path,
        ticker=args.ticker,
        dates=args.dates,
        batch_size=args.batch_size,
        debug=args.debug,
        gpus=args.gpus if args.gpus > 0 else None,
    )

    # Set up WandB logger
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "ticker": args.ticker,
            "dates": args.dates,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lookback_window": args.lookback_window,
            "n_training_dates": len(args.dates),
            "featureset": "NEW_SET",
            "model": "CGAN_Coletta_v2_41",
        },
    )

    # Monkey-patch __log_results_tb: original expects TensorBoard, we use WandB
    import wandb as _wandb
    _orig_log_results_tb = ganmodels.LOBGAN._LOBGAN__log_results_tb
    def _log_results_wandb(self, trades_df, epoch):
        try:
            for col in trades_df.columns:
                if int(epoch) > 0 and "real" in col:
                    continue
                _wandb.log({f"hist/{col}": _wandb.Histogram(trades_df[col].dropna().values)},
                           step=int(epoch))
        except Exception:
            pass  # non-critical logging
    ganmodels.LOBGAN._LOBGAN__log_results_tb = _log_results_wandb

    # Monkey-patch DataLoaders: add num_workers + pin_memory for speed
    from torch.utils.data import DataLoader as _DataLoader
    _num_workers = int(os.environ.get('CGAN_NUM_WORKERS', 64))
    _orig_train_dl = ganmodels.LOBGAN.train_dataloader
    _orig_val_dl = ganmodels.LOBGAN.val_dataloader
    def _fast_train_dl(self):
        return _DataLoader(
            self.train_data_loader, batch_size=self.batch_size,
            drop_last=True, shuffle=True,
            num_workers=_num_workers, pin_memory=True, persistent_workers=True,
        )
    def _fast_val_dl(self):
        return _DataLoader(
            self.val_data_loader, batch_size=self.batch_size,
            drop_last=True, shuffle=False,
            num_workers=_num_workers, pin_memory=True, persistent_workers=True,
        )
    ganmodels.LOBGAN.train_dataloader = _fast_train_dl
    ganmodels.LOBGAN.val_dataloader = _fast_val_dl

    # Set up checkpoint callback
    epoch_checkpoint_callback = ganmodels.MyPeriodicCheckpoint(every=1)

    # Create trainer
    from pytorch_lightning.trainer import Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 'auto',
        callbacks=[epoch_checkpoint_callback],
        logger=wandb_logger,
    )

    print(f"\nStarting training...")
    trainer.fit(model)

    # Finish WandB
    import wandb
    wandb.finish()

    # Print output paths
    model_dir = model.model_path
    print(f"\nTraining complete!")
    print(f"Model directory: {model_dir}")
    print(f"  Checkpoints: {model_dir}checkpoints/")
    print(f"  Scalers:     {model_dir}data__scalers.pickle")
    print(f"  Interarrival: {model_dir}interarrival_times")


if __name__ == "__main__":
    main()
