#!/usr/bin/env python
"""
Null Baseline with Continuous Hidden State

This script implements a null baseline simulation with:
- Autoregressive generation using S5 model
- Continuous hidden state (same as aggressive scenario)
- Single GPU execution (no sharding)
- ZERO aggressive order injections

Purpose: measure baseline drift by running the S5 model from
conditioning windows without any intervention. This provides
the counterfactual for market impact analysis.
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from functools import partial
from typing import Any, Dict, Optional, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import jax
import jax.numpy as jnp
import numpy as onp
from tqdm import tqdm

# Add parent folder to path (using __file__ to get correct path)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_folder_path)

# Add AlphaTrade submodule (mounted at /AlphaTrade in container)
# Also try relative path for local runs
if os.path.exists('/AlphaTrade'):
    sys.path.insert(0, '/AlphaTrade')
else:
    # For local execution, use relative path
    sys.path.insert(0, os.path.join(parent_folder_path, 'Alphatrade'))

from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob.encoding import Vocab, Message_Tokenizer
from lob_impact.core import inference_w_insertions as inference
import lob.validation_helpers as valh
import lob.encoding as encoding
import preproc
from lob.init_train import init_train_state, load_checkpoint, load_metadata

# Indices for DECODED message fields (from inference_no_errcorr.py)
ORDER_ID_i = 0
EVENT_TYPE_i = 1
DIRECTION_i = 2
PRICE_ABS_i = 3
PRICE_i = 4
SIZE_i = 5
DTs_i = 6
DTns_i = 7
TIMEs_i = 8
TIMEns_i = 9
PRICE_REF_i = 10
SIZE_REF_i = 11
TIMEs_REF_i = 12
TIMEns_REF_i = 13

AGGRESSIVE_ORDER_ID = 77777777


class TeeLogger:
    """Duplicates output to both console and log file."""
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', buffering=1)  # line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def setup_logging(save_folder: Path) -> TeeLogger:
    """Set up logging to both console and file."""
    log_file = save_folder / 'experiment.log'
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee


def create_experiment_folder(base_dir: str) -> Path:
    """Create experiment folder with sequential number and timestamp."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Find max experiment index
    max_idx = 0
    for entry in base.iterdir():
        if entry.is_dir() and entry.name.startswith("exp_"):
            parts = entry.name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))

    next_idx = max_idx + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder = base / f"exp_{next_idx}_{timestamp}"
    new_folder.mkdir(parents=True, exist_ok=False)

    return new_folder


def sample_null_baseline(
    cfg: Dict[str, Any],
    save_folder: Path,
):
    """
    Null baseline: generate messages without any injection.

    Same as aggressive scenario but with zero insertions.
    Feed n_cond_msgs conditioning messages, then generate n_gen_msgs freely.
    Record mid-price trajectory for drift analysis.
    """
    # Unpack config
    n_gen_msgs = cfg['n_gen_msgs']
    n_cond_msgs = cfg['n_cond_msgs']
    n_samples = cfg['n_samples']
    batch_size = cfg['batch_size']
    rng_seed = cfg['rng_seed']
    stock = cfg['stock']
    data_dir = cfg['data_dir']
    ckpt_path = cfg['ckpt_path']
    tick_size = cfg['tick_size']
    sample_top_n = cfg['sample_top_n']
    n_vol_series = cfg['n_vol_series']
    book_dim = cfg['book_dim']
    test_split = cfg['test_split']
    n_eval_msgs_dataset = cfg.get('n_eval_msgs_dataset', 500)
    checkpoint_step = cfg.get('checkpoint_step', None)
    chunk_size = cfg.get('chunk_size', 1)

    # Derived parameters - no insertions, just pure generation
    n_msg_todo_total = n_gen_msgs
    cond_seq_len = n_cond_msgs * Message_Tokenizer.MSG_LEN

    # Initialize
    v = Vocab()
    n_classes = len(v)
    rng = jax.random.key(rng_seed)

    # Load model
    print(f"Loading model from {ckpt_path}")
    args = load_metadata(ckpt_path)
    args.num_devices = 1
    args.bsz = 1

    new_train_state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=cond_seq_len,
        book_dim=book_dim,
        book_seq_len=n_cond_msgs,
    )

    ckpt = load_checkpoint(new_train_state, ckpt_path, step=checkpoint_step, train=False)
    print(f"Loaded checkpoint step: {ckpt['step']}")
    train_state = ckpt['model']
    model = model_cls(training=False, step_rescale=1.0)

    # Load dataset
    print(f"Loading dataset from {data_dir}")
    ds = inference.get_dataset(
        data_dir,
        n_cond_msgs,
        n_eval_msgs_dataset,
        test_split=test_split,
    )
    print(f"Dataset length: {len(ds)}")

    # Create output folders
    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_real').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    # Sample indices
    assert n_samples % batch_size == 0, f'n_samples ({n_samples}) must be divisible by batch_size ({batch_size})'
    rng, _ = jax.random.split(rng)
    rng, rng_ = jax.random.split(rng)
    sample_i = jax.random.choice(
        rng_,
        jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size),
        replace=False
    ).tolist()

    # Initialize hidden state template
    init_hidden = model.initialize_carry(
        1,
        hidden_size=(args.ssm_size_base // pow(2, int(args.conj_sym))),
        n_message_layers=args.n_message_layers,
        n_book_pre_layers=args.n_book_pre_layers,
        n_book_post_layers=args.n_book_post_layers,
        n_fused_layers=args.n_layers,
        h_size_ema=args.ssm_size_base
    )

    # Replicate for batch
    init_hidden_batched = jax.tree_util.tree_map(
        lambda x: jnp.resize(x, (batch_size,) + x.shape),
        init_hidden
    )

    # Initialize simulator
    sim = OrderBook(cfg=JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))

    # Transform function for book states
    transform_L2_state_batch = jax.jit(
        jax.vmap(preproc.transform_L2_state_gpu, in_axes=(0, None, None)),
        static_argnums=(1, 2)
    )

    # AOT compile generate function (once)
    print("Compiling generate function (this happens once)...")
    generate_compiled = None

    # Split RNG before loop
    rng, rng_ = jax.random.split(rng)

    # Process batches
    for batch_idx, batch_i in enumerate(tqdm(sample_i, desc="Batches")):
        print(f'\n=== BATCH {batch_idx}: samples {batch_i} ===')

        # Load data and put on GPU
        gpu_device = jax.devices('gpu')[0]
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        m_seq = jax.device_put(jnp.array(m_seq), gpu_device)
        b_seq_pv = jax.device_put(jnp.array(b_seq_pv), gpu_device)
        msg_seq_raw = jax.device_put(jnp.array(msg_seq_raw), gpu_device)
        book_l2_init = jax.device_put(jnp.array(book_l2_init), gpu_device)

        # Transform book to volume representation
        b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)
        init_time_batched = b_seq_pv[:, 0, 1:3]

        # Split conditioning and evaluation
        m_seq_cond = m_seq[:, :cond_seq_len + 1]
        b_seq_cond = b_seq[:, :n_cond_msgs + 1]
        m_seq_raw_cond = msg_seq_raw[:, :n_cond_msgs]
        b_seq_pv_cond = onp.array(b_seq_pv[:, :n_cond_msgs + 1, 3:])

        # Initialize simulators
        sim_states = inference.get_sims_vmap(
            book_l2_init,
            m_seq_raw_cond,
            init_time_batched,
            sim,
        )

        # NO insertion schedule - pure generation
        # Pass empty insertion schedule (all zeros = no insertions)
        insertion_schedule = jnp.zeros((n_msg_todo_total, 9), dtype=jnp.int32)
        insertion_schedule_batched = jnp.tile(insertion_schedule[None, :, :], (batch_size, 1, 1))

        # Use rng_ from pre-loop split
        rng_batch = jax.random.split(rng_, batch_size)

        # AOT compile on first call
        if generate_compiled is None:
            print("  AOT compiling generate function...")
            generate_traced = inference.generate_batched.trace(
                sim,
                train_state,
                model,
                args.batchnorm,
                v.ENCODING,
                sample_top_n,
                tick_size,
                m_seq_cond,
                b_seq_cond,
                n_msg_todo_total,
                sim_states,
                rng_batch,
                init_hidden_batched,
                True,  # conditional
                init_time_batched,
                False,  # debug_book
                None,   # b_seq_real
                insertion_schedule_batched,
                chunk_size,
            )
            generate_lowered = generate_traced.lower()
            generate_compiled = generate_lowered.compile()

        # Generate ALL messages in single call (no insertions)
        all_msgs, all_books, num_errors, msgs_tokens = generate_compiled(
            train_state,
            v.ENCODING,
            m_seq_cond,
            b_seq_cond,
            sim_states,
            rng_batch,
            init_hidden_batched,
            init_time_batched,
            None,
            insertion_schedule_batched,
        )

        # Filter out zero-filled placeholder rows
        valid_mask = (all_msgs[:, :, ORDER_ID_i] != 0) | (all_msgs[:, :, EVENT_TYPE_i] != 0)
        n_valid_per_batch = valid_mask.sum(axis=1)
        print(f"  Generated {all_msgs.shape[1]} raw slots, {n_valid_per_batch[0]} valid messages, errors: {num_errors.sum()}")

        # Save results
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)

            # Conditioning data
            inference.msg_to_lobster_format(m_seq_raw_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_message_real_id_{sample_idx}.csv',
                index=False, header=False
            )
            inference.book_to_lobster_format(b_seq_pv_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_orderbook_real_id_{sample_idx}.csv',
                index=False, header=False
            )

            # Filter valid messages for this batch item
            valid_msgs_i = all_msgs[i][valid_mask[i]]
            valid_books_i = all_books[i][valid_mask[i]]

            # Generated data
            inference.msg_to_lobster_format(valid_msgs_i).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_message_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )
            inference.book_to_lobster_format(valid_books_i).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_orderbook_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )

        # Split RNG for next iteration
        rng, rng_ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="Null Baseline: S5 generation without aggressive injections")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='lob_impact/scenarios/0.null_baseline_config.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument('--n_gen_msgs', type=int, default=None, help='Override n_gen_msgs from config')
    return parser.parse_args()


def main():
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")

    args = parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.n_gen_msgs is not None:
        cfg['n_gen_msgs'] = args.n_gen_msgs

    print(f"Configuration: {cfg}")

    # Create experiment folder
    save_folder = create_experiment_folder(cfg['save_dir'])
    print(f"Experiment folder: {save_folder}")

    # Set up logging
    logger = setup_logging(save_folder)
    print(f"\n{'='*60}")
    print(f"Null Baseline Experiment (No Injections)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Configuration: {cfg}")
    print(f"Experiment folder: {save_folder}")

    # Save config
    with open(save_folder / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    try:
        sample_null_baseline(cfg, save_folder)
        print(f"\n{'='*60}")
        print(f"Experiment completed!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {save_folder}")
        print(f"{'='*60}")
    finally:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"Log saved to: {save_folder / 'experiment.log'}")


if __name__ == "__main__":
    main()
