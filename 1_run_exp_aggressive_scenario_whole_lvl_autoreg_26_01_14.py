#!/usr/bin/env python
import os
import sys
import argparse
import yaml

# prevent XLA pre-allocation if using TPU/GPU backends
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Parse --gpu argument BEFORE importing JAX to set CUDA_VISIBLE_DEVICES early
def _get_gpu_arg():
    """Extract --gpu argument before full argparse to set CUDA_VISIBLE_DEVICES."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--gpu="):
            return arg.split("=")[1]
    return None

_gpu_arg = _get_gpu_arg()
if _gpu_arg is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg
    print(f"[Early init] Restricting to GPU {_gpu_arg} (CUDA_VISIBLE_DEVICES={_gpu_arg})")

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import wandb
import jax
from pathlib import Path
from datetime import datetime

from lob.encoding import Vocab, Message_Tokenizer
# Use autoregressive inference module
from lob import inference_no_errcorr_autoreg as inference
from lob.init_train import init_train_state, load_checkpoint, load_metadata
from lob.lob_seq_model import FullLobPredModel, PaddedLobPredModel
from argparse import Namespace

import os
import numpy as np  # or use onp if preferred

from datetime import datetime
import functools
from glob import glob
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.nn import one_hot
import flax.linen as nn
from flax.training.train_state import TrainState
from lob import train_helpers
from flax.jax_utils import replicate as jax_replicate
import numpy as onp
import os
import sys
import pandas as pd
import pickle
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
from utils import debug, info
import os

# Use autoregressive validation helpers
import lob.validation_helpers_autoreg as valh
import lob.evaluation as eval
import preproc as preproc
from preproc import transform_L2_state, transform_L2_state_gpu
import lob.encoding as encoding
from lob.encoding import Message_Tokenizer, Vocab
from lob.lobster_dataloader import LOBSTER_Dataset
from jax import lax
import json


# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob.init_train import init_train_state, load_checkpoint, load_metadata, load_args_from_checkpoint


def add_time_to_book_data(b_seq_pv: jax.Array, msg_seq_raw: jax.Array) -> jax.Array:
    """
    Add time information to book data for compatibility with transform_L2_state_gpu.

    The checkpoint was trained with book data format: [delta_p_mid, time_s, time_ns, volumes...]
    But our dataset only has: [delta_p_mid, volumes...]

    This function extracts time from messages and inserts it into the book data.

    Args:
        b_seq_pv: Raw book data (batch, n_msgs, 1 + n_price_volume_pairs)
                  First column is delta_p_mid, rest is price-volume pairs
        msg_seq_raw: Raw message data (batch, n_msgs, msg_dim)
                     time_s is at index 8, time_ns at index 9

    Returns:
        Book data with time: (batch, n_msgs, 3 + n_price_volume_pairs)
        Format: [delta_p_mid, time_s, time_ns, price1, vol1, price2, vol2, ...]
    """
    # Extract time from messages
    time_s = msg_seq_raw[:, :, 8:9]    # (batch, n_msgs, 1)
    time_ns = msg_seq_raw[:, :, 9:10]  # (batch, n_msgs, 1)

    # Split book data: delta_p_mid and the rest
    delta_p_mid = b_seq_pv[:, :, :1]   # (batch, n_msgs, 1)
    volumes = b_seq_pv[:, :, 1:]       # (batch, n_msgs, n_price_volume_pairs)

    # Concatenate: [delta_p_mid, time_s, time_ns, volumes...]
    b_seq_with_time = jnp.concatenate([delta_p_mid, time_s, time_ns, volumes], axis=-1)

    return b_seq_with_time


def track_midprices_during_messages(
        m_seq_raw_inp: jax.Array,
        book_l2_init: jax.Array,
        tick_size: int,
        step_size: int,
    ) -> jax.Array:
    """
    JIT‑friendly, loop‑free version that computes mid‑prices every
    ``step_size`` messages using `jax.lax.scan`.

    Args:
        m_seq_raw_inp:  (batch, T, msg_dim) – raw decoded messages
        book_l2_init:   (batch, book_dim) – initial L2 state of the orderbook
        tick_size:      Tick size of the instrument
        step_size:      Interval (in number of messages) between successive mid‑price samples

    Returns
    -------
    midprices : jax.Array
        Shape (num_steps, batch).  mid‑price after each `step_size` messages.
    """
    # Number of *complete* chunks of length `step_size`
    num_steps = m_seq_raw_inp.shape[1] // step_size
    if num_steps == 0:
        return jnp.empty((0, m_seq_raw_inp.shape[0]), dtype=jnp.float32)

    # Reshape messages into (num_steps, batch, step_size, msg_dim)
    msgs = m_seq_raw_inp[:, : num_steps * step_size, :]
    msgs = msgs.reshape(
        m_seq_raw_inp.shape[0],          # batch
        num_steps,
        step_size,
        m_seq_raw_inp.shape[2],          # msg_dim
    )
    msgs = jnp.swapaxes(msgs, 0, 1)      # → (num_steps, batch, step_size, msg_dim)

    def scan_step(carry, msg_chunk):
        # carry: current book L2 state  (batch, book_dim)
        sim_init, sim_states = inference.get_sims_vmap(carry, msg_chunk)
        mid_price = inference.batched_get_safe_mid_price(
            sim_init, sim_states, tick_size
        )
        # Extract fresh L2 state to feed next step
        full_l2_state = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(
            sim_states, carry.shape[1]
        )
        # keep only the leading slice so the carry's shape matches the input
        new_l2_state = full_l2_state[:, : carry.shape[1]]
        return new_l2_state, mid_price

    _, midprices = lax.scan(scan_step, book_l2_init, msgs)  # (num_steps, batch)
    return midprices



# def insert_custom_end(
#         m_seq_gen_doubled,
#         b_seq_gen_doubled,
#         msgs_decoded_doubled,
#         l2_book_states_halved,
#         encoder,
#         mid_price,
#         tick_size=100,
#         EVENT_TYPE_i=4,
#         DIRECTION_i=0,
#         order_volume=75,
#         use_relative_volume: bool = False,
#         order_volume_ratio: float = 1.0,
#     ):
    
#     ORDER_ID_i = 77777777
#     sim_init, sim_states_init = inference.get_sims_vmap(l2_book_states_halved[:,-2], msgs_decoded_doubled[:,-1:])
#     if DIRECTION_i == 0:
#         PRICE_i = jax.vmap(sim_init.get_best_ask)(sim_states_init)
#     if DIRECTION_i == 1:
#         PRICE_i = jax.vmap(sim_init.get_best_bid)(sim_states_init)

#     PRICE_i = jnp.expand_dims(PRICE_i, axis=-1)
#     mid_price = jnp.expand_dims(mid_price, axis=-1)

#     TIMEs_i = msgs_decoded_doubled[:, -1:, 8].astype(jnp.int32)
#     TIMEns_i = msgs_decoded_doubled[:, -1:, 9].astype(jnp.int32)

#     batch_size = TIMEns_i.shape[0]

#     best_bid_ask = jax.vmap(sim_init.get_best_bid_and_ask_inclQuants)(sim_states_init)
#     # Insert custom logic for order sizing
#     if use_relative_volume:
#         if DIRECTION_i == 0:
#             SIZE_i = (best_bid_ask[1][:, 1] * order_volume_ratio).astype(jnp.int32)
#         else:
#             SIZE_i = (best_bid_ask[0][:, 1] * order_volume_ratio).astype(jnp.int32)
#     else:
#         if DIRECTION_i == 0:
#             SIZE_i = jnp.minimum(order_volume, best_bid_ask[1][:, 1])
#         else:
#             SIZE_i = jnp.minimum(order_volume, best_bid_ask[0][:, 1])
#     batched_quantity = SIZE_i

#     batched_new_order_id = jnp.array([ORDER_ID_i] * batch_size, dtype=jnp.int32)
#     batched_EVENT_TYPE = jnp.array([EVENT_TYPE_i] * batch_size, dtype=jnp.int32)
#     batched_side = jnp.array([DIRECTION_i] * batch_size, dtype=jnp.int32)
#     batched_p_abs = PRICE_i.squeeze(-1)    
#     batched_time_s = TIMEs_i.squeeze(-1)
#     batched_time_ns = TIMEns_i.squeeze(-1)

#     batched_construct_sim_msg = jax.vmap(inference.construct_sim_msg)
#     batched_sim_msg = batched_construct_sim_msg(
#         batched_EVENT_TYPE,
#         batched_side,
#         batched_quantity,
#         batched_p_abs,
#         batched_new_order_id,
#         batched_time_s,
#         batched_time_ns,
#     )

#     new_sim_state = jax.vmap(sim_init.process_order_array)(sim_states_init, batched_sim_msg)
#     p_mid_new = inference.batched_get_safe_mid_price(sim_init, new_sim_state, tick_size)
#     p_mid_new = p_mid_new[:, None]
#     p_change = ((p_mid_new - mid_price) // tick_size).astype(jnp.int32)
#     book_l2 = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(new_sim_state, 20)
#     new_l2_book_states_halved = jnp.concatenate([l2_book_states_halved, book_l2[:, None, :]], axis=1)
#     new_book_raw = jnp.concatenate([p_change, book_l2], axis=1)
#     new_book_raw = new_book_raw[:, None, :]

#     transform_L2_state_batch = jax.jit(jax.vmap(inference.transform_L2_state, in_axes=(0, None, None)), static_argnums=(1, 2))
#     new_book = transform_L2_state_batch(new_book_raw, 500, 100)

#     b_seq_gen_doubled = jnp.concatenate([b_seq_gen_doubled, new_book], axis=1)

#     ins_msg = jnp.concatenate([
#         batched_new_order_id.reshape(-1, 1),
#         batched_EVENT_TYPE.reshape(-1, 1),
#         batched_side.reshape(-1, 1),
#         batched_p_abs.reshape(-1, 1),
#         jnp.full((batch_size, 1), 1, dtype=jnp.int32),
#         batched_quantity.reshape(-1, 1),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         batched_time_s.reshape(-1, 1),
#         batched_time_ns.reshape(-1, 1),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#         jnp.full((batch_size, 1), 0, dtype=jnp.int32),
#     ], axis=1)
 
#     new_batched_sim_msg = ins_msg[:, None, :]
#     UPDATED_msgs_decoded_doubled = jnp.concatenate([msgs_decoded_doubled, new_batched_sim_msg], axis=1)

#     msg_encoded = jax.vmap(lambda m: encoding.encode_msg(m, encoder))(ins_msg)
#     UPDATED_m_seq_gen_doubled = jnp.concatenate([m_seq_gen_doubled, msg_encoded], axis=1)

#     return UPDATED_m_seq_gen_doubled, b_seq_gen_doubled, UPDATED_msgs_decoded_doubled, new_l2_book_states_halved, p_mid_new

def insert_custom_end(
        m_seq_gen_doubled,
        b_seq_gen_doubled,
        msgs_decoded_doubled,
        l2_book_states_halved,
        encoder,
        mid_price,
        tick_size=100,
        EVENT_TYPE_i=4,
        DIRECTION_i=0,          # 0 = buy (hit ask), 1 = sell (hit bid)
        order_volume=75,        # kept for compat, not used by sizing rule
        use_relative_volume=False,
        order_volume_ratio=1.0,
    ):
    ORDER_ID_i = 77777777
    sim_init, sim_states_init = inference.get_sims_vmap(
        l2_book_states_halved[:, -2], msgs_decoded_doubled[:, -1:]
    )

    # best price on the side we will aggress
    if DIRECTION_i == 0:
        PRICE_i = jax.vmap(sim_init.get_best_ask)(sim_states_init)
    else:
        PRICE_i = jax.vmap(sim_init.get_best_bid)(sim_states_init)

    PRICE_i   = jnp.expand_dims(PRICE_i, axis=-1)
    mid_price = jnp.expand_dims(mid_price, axis=-1)

    TIMEs_i  = msgs_decoded_doubled[:, -1:, 8].astype(jnp.int32)
    TIMEns_i = msgs_decoded_doubled[:, -1:, 9].astype(jnp.int32)
    batch_size = TIMEns_i.shape[0]

    # === volumes at best levels ===
    # best_bid_ask structure from your code: [0] = bid info, [1] = ask info; [:, 1] = quantity
    best_bid_ask = jax.vmap(sim_init.get_best_bid_and_ask_inclQuants)(sim_states_init)

    # pick available volume on the level we will consume
    avail = jnp.where(
        DIRECTION_i == 0,
        best_bid_ask[1][:, 1],  # ask vol for a buy
        best_bid_ask[0][:, 1],  # bid vol for a sell
    ).astype(jnp.int32)

    # === SIZING RULE (CONFIG-DRIVEN) ===
    # If use_relative_volume: consume a fraction of available L1 volume.
    #   SIZE = floor(avail * order_volume_ratio); if avail>0 and result==0 -> 1
    # Else: fixed-size capped by avail (do not cross levels here).
    if use_relative_volume:
        ratio = jnp.clip(jnp.float32(order_volume_ratio), 0.0, 1.0)
        SIZE_i = jnp.floor(ratio * jnp.float32(avail)).astype(jnp.int32)
        # if there's some liquidity but rounding gave 0, take 1; if avail==0, keep 0
        SIZE_i = jnp.where((avail > 0) & (SIZE_i == 0), 1, SIZE_i)
    else:
        SIZE_i = jnp.minimum(jnp.int32(order_volume), avail)

    batched_quantity   = SIZE_i  # sized by relative fraction or fixed cap
    batched_new_order_id = jnp.full((batch_size,), ORDER_ID_i, dtype=jnp.int32)
    batched_EVENT_TYPE   = jnp.full((batch_size,), EVENT_TYPE_i, dtype=jnp.int32)
    batched_side         = jnp.full((batch_size,), DIRECTION_i, dtype=jnp.int32)
    batched_p_abs        = PRICE_i.squeeze(-1)
    batched_time_s       = TIMEs_i.squeeze(-1)
    batched_time_ns      = TIMEns_i.squeeze(-1)

    batched_construct_sim_msg = jax.vmap(inference.construct_sim_msg)
    batched_sim_msg = batched_construct_sim_msg(
        batched_EVENT_TYPE,
        batched_side,
        batched_quantity,
        batched_p_abs,
        batched_new_order_id,
        batched_time_s,
        batched_time_ns,
    )

    new_sim_state = jax.vmap(sim_init.process_order_array)(sim_states_init, batched_sim_msg)
    p_mid_new = inference.batched_get_safe_mid_price(sim_init, new_sim_state, tick_size)
    p_mid_new = p_mid_new[:, None]
    p_change = ((p_mid_new - mid_price) // tick_size).astype(jnp.int32)

    # Use 10 levels (40 values) to match l2_state_n in autoregressive inference
    book_l2 = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(new_sim_state, 10)
    new_l2_book_states_halved = jnp.concatenate([l2_book_states_halved, book_l2[:, None, :]], axis=1)

    # Include time for transform_L2_state_gpu which expects: [delta_p_mid, time_s, time_ns, book_data...]
    # p_change shape: (batch, 1), batched_time_s/ns shape: (batch,)
    # Keep all data as int32 - transform_L2_state_gpu will handle type conversions internally
    time_s_col = batched_time_s.reshape(-1, 1).astype(jnp.int32)
    time_ns_col = batched_time_ns.reshape(-1, 1).astype(jnp.int32)
    # Create (batch, 43) array for transform: [p_change, time_s, time_ns, book_l2...]
    # book_l2 contains [bid_p1, bid_v1, ask_p1, ask_v1, ...] - must remain int for indexing
    new_book_raw = jnp.concatenate([p_change.astype(jnp.int32), time_s_col, time_ns_col, book_l2.astype(jnp.int32)], axis=1)
    # Shape is (batch, 43) - transform_L2_state_gpu expects 1D input per batch element

    # Use transform_L2_state_gpu to get 503-element output (with time normalization)
    transform_L2_state_batch = jax.jit(
        jax.vmap(preproc.transform_L2_state_gpu, in_axes=(0, None, None)), static_argnums=(1, 2)
    )
    new_book = transform_L2_state_batch(new_book_raw, 500, 100)  # Output: (batch, 503)
    new_book = new_book[:, None, :]  # Add time step dimension: (batch, 1, 503)
    b_seq_gen_doubled = jnp.concatenate([b_seq_gen_doubled, new_book], axis=1)

    ins_msg = jnp.concatenate([
        batched_new_order_id.reshape(-1, 1),
        batched_EVENT_TYPE.reshape(-1, 1),
        batched_side.reshape(-1, 1),
        batched_p_abs.reshape(-1, 1),
        jnp.full((batch_size, 1), 1, dtype=jnp.int32),       # visible flag
        batched_quantity.reshape(-1, 1),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        batched_time_s.reshape(-1, 1),
        batched_time_ns.reshape(-1, 1),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
        jnp.full((batch_size, 1), 0, dtype=jnp.int32),
    ], axis=1)

    new_batched_sim_msg     = ins_msg[:, None, :]
    UPDATED_msgs_decoded_doubled = jnp.concatenate([msgs_decoded_doubled, new_batched_sim_msg], axis=1)

    msg_encoded = jax.vmap(lambda m: encoding.encode_msg(m, encoder))(ins_msg)
    UPDATED_m_seq_gen_doubled = jnp.concatenate([m_seq_gen_doubled, msg_encoded], axis=1)

    return UPDATED_m_seq_gen_doubled, b_seq_gen_doubled, UPDATED_msgs_decoded_doubled, new_l2_book_states_halved, p_mid_new


# Helper to get init time from last message in conditioning sequence
def get_init_time_from_msgs(m_seq_raw_inp: jax.Array) -> jax.Array:
    """
    Extract time from the last message in conditioning sequence.

    Args:
        m_seq_raw_inp: (batch, n_msgs, 14) raw decoded messages

    Returns:
        init_time: (batch, 2) array of [time_s, time_ns]
    """
    # TIMEs_i = 8, TIMEns_i = 9 in decoded message format
    time_s = m_seq_raw_inp[:, -1, 8].astype(jnp.int32)
    time_ns = m_seq_raw_inp[:, -1, 9].astype(jnp.int32)
    return jnp.stack([time_s, time_ns], axis=-1)


def run_generation_scenario(
        n_samples: int,
        batch_size: int,
        ds: LOBSTER_Dataset,
        rng: jax.dtypes.prng_key,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        stock_symbol: str,
        n_vol_series: int = 500,
        save_folder: str = './data_saved/',
        tick_size: int = 100,
        sample_top_n: int = -1,
        sample_all: bool = False,
        num_insertions: int = 2,
        num_coolings: int = 2,
        midprice_step_size=100,
        EVENT_TYPE_i=4,
        DIRECTION_i=0,
        order_volume=75,
        use_relative_volume: bool = False,
        order_volume_ratio: float = 1.0,
        use_sample_file: bool = False,
        sample_file_path: Optional[str] = None,
        start_batch: int = 0,
        end_batch: int = -1,
        # New parameters for autoregressive generation
        args_ckpt: Optional[Namespace] = None,
        token_mode: int = 24,
        model_unbatched: Optional[nn.Module] = None,  # Unbatched model for generate_batched
        num_devices: int = 1,  # Number of GPUs for pmap parallelization
    ):

    # rng, rng_ = jax.random.split(rng)
    # if sample_all:
    #     sample_i = jnp.arange(
    #         len(ds) // batch_size * batch_size,
    #         dtype=jnp.int32
    #     ).reshape(-1, batch_size).tolist()
    # else:
    #     assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'

    #     sample_i = jax.random.choice(
    #         rng_,
    #         jnp.arange(len(ds), dtype=jnp.int32),
    #         shape=(n_samples // batch_size, batch_size),
    #         replace=False
    #     ).tolist()
    # rng, rng_ = jax.random.split(rng)

    
    rng, rng_ = jax.random.split(rng)

    if use_sample_file:
        assert sample_file_path is not None, "Path to sample file not provided"
        with open(sample_file_path, "r") as f:
            sample_i_full = json.load(f)
        # режем по start/end
        sample_i = sample_i_full[start_batch: end_batch if end_batch != -1 else None]
        # If batch_size is smaller than file batches, truncate each batch
        file_batch_size = len(sample_i[0]) if sample_i else 0
        if file_batch_size > batch_size:
            print(f"Truncating batches from {file_batch_size} to {batch_size} samples")
            sample_i = [batch[:batch_size] for batch in sample_i]
        # проверяем, что каждый батч нужного размера
        for i, batch in enumerate(sample_i):
            assert len(batch) == batch_size, (
                f"Batch {i} has incorrect size {len(batch)}, expected {batch_size}"
            )
    else:
        if sample_all:
            sample_i = jnp.arange(
                len(ds) // batch_size * batch_size,
                dtype=jnp.int32
            ).reshape(-1, batch_size).tolist()
        else:
            assert n_samples % batch_size == 0, 'n_samples must be divisible by batch_size'
            sample_i = jax.random.choice(
                rng_,
                jnp.arange(len(ds), dtype=jnp.int32),
                shape=(n_samples // batch_size, batch_size),
                replace=False
            ).tolist()

    rng, rng_ = jax.random.split(rng)

    save_folder = Path(save_folder)

    # Align saved folders with heuristic_historical_scenario_run_quantile_fixing.py
    save_folder.joinpath('msgs_decoded_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('b_seq_gen_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('mid_price').mkdir(exist_ok=True, parents=True)

    base_save_folder = save_folder

    # Use transform_L2_state_gpu which expects time info (3 leading elements: delta_p_mid, time_s, time_ns)
    # The vmap is over the first dimension (batch), then the function JITs internally
    transform_L2_state_batch_gpu = jax.jit(
        jax.vmap(
            jax.vmap(transform_L2_state_gpu, in_axes=(0, None, None)),  # over n_msgs
            in_axes=(0, None, None)  # over batch
        ),
        static_argnums=(1, 2)
    )

    num_iterations = num_insertions + num_coolings

    # === Initialize hidden state for autoregressive generation ===
    if args_ckpt is None:
        raise ValueError("args_ckpt must be provided for autoregressive generation")

    conj_sym = getattr(args_ckpt, 'conj_sym', True)
    hidden_size = args_ckpt.ssm_size_base // (2 if conj_sym else 1)

    print(f"Initializing hidden state: batch_size={batch_size}, hidden_size={hidden_size}")
    print(f"  n_message_layers={getattr(args_ckpt, 'n_message_layers', 2)}")
    print(f"  n_book_pre_layers={getattr(args_ckpt, 'n_book_pre_layers', 1)}")
    print(f"  n_book_post_layers={getattr(args_ckpt, 'n_book_post_layers', 1)}")
    print(f"  n_fused_layers={args_ckpt.n_layers}")
    print(f"  merging={getattr(args_ckpt, 'merging', 'projected')}")

    # Use correct model class for initialize_carry based on merging method
    merging = getattr(args_ckpt, 'merging', 'projected')
    model_class_for_carry = PaddedLobPredModel if merging == 'padded' else FullLobPredModel

    init_hidden = model_class_for_carry.initialize_carry(
        batch_size=batch_size,
        hidden_size=hidden_size,
        n_message_layers=getattr(args_ckpt, 'n_message_layers', 2),
        n_book_pre_layers=getattr(args_ckpt, 'n_book_pre_layers', 1),
        n_book_post_layers=getattr(args_ckpt, 'n_book_post_layers', 1),
        n_fused_layers=args_ckpt.n_layers,
        h_size_ema=args_ckpt.ssm_size_base,
    )

    # === Setup multi-GPU pmap if num_devices > 1 ===
    use_pmap = num_devices > 1 and batch_size % num_devices == 0
    if use_pmap:
        print(f"Using PMAP with {num_devices} devices for parallel generation")
        generate_pmap = inference.create_generate_batched_pmap(num_devices)
        samples_per_device = batch_size // num_devices
        # Replicate train_state to all devices for parallel execution
        train_state_replicated = jax_replicate(train_state)
        print(f"Replicated train_state to {num_devices} devices")
        # Verify device placement
        first_param = jax.tree_util.tree_leaves(train_state_replicated.params)[0]
        print(f"  train_state params shape: {first_param.shape} (first dim = num_devices)")
    else:
        print(f"Using single-GPU VMAP (num_devices={num_devices}, batch_size={batch_size})")
        generate_pmap = None
        train_state_replicated = None

    print('sample_i:', sample_i)

    for batch_i in tqdm(sample_i):
        print('BATCH', batch_i)
        proc_msgs_numb = -n_msgs

        # Reset hidden state for each new batch
        hidden_state = init_hidden

        for iteration in range(1,num_iterations+1):
            print('\nITERATION ', iteration)
            midprices = []

            if iteration == 1:
                # First iteration: load data and use conditional=True
                is_first_iteration = True

                m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
                m_seq = jnp.array(m_seq)
                b_seq_pv = jnp.array(b_seq_pv)
                msg_seq_raw = jnp.array(msg_seq_raw)
                book_l2_init = jnp.array(book_l2_init)

                # Add time info to book data for compatibility with checkpoint (expects book_dim=503)
                b_seq_pv_with_time = add_time_to_book_data(b_seq_pv, msg_seq_raw)
                # Transform using GPU version that expects 3 leading elements (delta_p_mid, time_s, time_ns)
                b_seq = transform_L2_state_batch_gpu(b_seq_pv_with_time, n_vol_series, tick_size)

                m_seq_inp = m_seq[:, : seq_len]
                b_seq_inp = b_seq[: , : n_msgs]
                m_seq_raw_inp = msg_seq_raw[:, : n_msgs]

                m_seq_np = np.array(jax.device_get(m_seq_inp))
                b_seq_np = np.array(jax.device_get(b_seq_inp))
                msgs_decoded_np = np.array(jax.device_get(m_seq_raw_inp))
                l2_book_states_np = np.array(jax.device_get(book_l2_init))

                # Do not save intermediate inputs or initial states

                sim_init, sim_states_init = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp)

                midprices_batch = track_midprices_during_messages(
                    m_seq_raw_inp,
                    book_l2_init,
                    tick_size,
                    midprice_step_size,
                )
                midprices = list(midprices_batch)            # start new history
                proc_msgs_numb += m_seq_raw_inp.shape[1]     # advance counter

                # Reset hidden state for first iteration of each batch
                hidden_state = init_hidden

            else:
                # For iterations after the first, we continue from previous generation
                # Use conditional=False to skip conditioning sequence processing
                m_seq = m_seq_gen_doubled
                b_seq = b_seq_gen_doubled
                msg_seq_raw = msgs_decoded_doubled
                m_seq_raw_inp = msg_seq_raw

                # For non-conditional mode, we just need the last token and last book state
                # m_seq_inp should be shape (batch, 1) - just the last token
                m_seq_inp = m_seq[:, -1:]  # Last token only
                b_seq_inp = b_seq[:, -1:]  # Last book state only

                sim_init, sim_states_init = inference.get_sims_vmap(l2_book_states_halved[:,-2], m_seq_raw_inp[:,-1:])
                # Use hidden_state from previous iteration (already set)
                is_first_iteration = False

            # === AUTOREGRESSIVE GENERATION ===
            print(f"\n=== AUTOREGRESSIVE GENERATION (batch_size={batch_size}, conditional={is_first_iteration}, use_pmap={use_pmap}) ===")

            # Create RNG keys for the batch
            rng, rng_ = jax.random.split(rng)
            rng_keys = jax.random.split(rng_, batch_size)  # Shape: (batch_size, 2)

            # Get init_time from last message
            init_time = get_init_time_from_msgs(m_seq_raw_inp)
            print(f"  init_time shape: {init_time.shape}")
            print(f"  m_seq_inp shape: {m_seq_inp.shape}")
            print(f"  b_seq_inp shape: {b_seq_inp.shape}")

            # Call autoregressive generation
            # Returns: msgs_decoded, l2_book_states, num_errors, msgs_tokens, hidden_state
            if use_pmap:
                # Reshape inputs for pmap: (batch, ...) -> (num_devices, per_device, ...)
                m_seq_inp_pmap = inference.reshape_for_pmap(m_seq_inp, num_devices)
                b_seq_inp_pmap = inference.reshape_for_pmap(b_seq_inp, num_devices)
                rng_keys_pmap = inference.reshape_for_pmap(rng_keys, num_devices)
                init_time_pmap = inference.reshape_for_pmap(init_time, num_devices)
                hidden_state_pmap = inference.reshape_hidden_for_pmap(hidden_state, num_devices)

                # Reshape sim_states_init - it's a LobState pytree
                sim_states_init_pmap = jax.tree.map(
                    lambda x: inference.reshape_for_pmap(x, num_devices) if hasattr(x, 'shape') else x,
                    sim_states_init
                )

                # Call pmap version
                msgs_decoded_pmap, l2_book_states_pmap, num_errors_pmap, msgs_tokens_pmap, hidden_state_pmap = generate_pmap(
                    sim_init,           # OrderBook instance (replicated)
                    train_state_replicated,  # model params (replicated to all devices!)
                    model_unbatched,    # unbatched model (replicated)
                    batchnorm,          # bool (replicated)
                    encoder,            # dict (replicated)
                    sample_top_n,       # int (replicated)
                    tick_size,          # int (replicated)
                    m_seq_inp_pmap,     # (num_devices, per_device, seq_len) tokens
                    b_seq_inp_pmap,     # (num_devices, per_device, n_msgs, book_dim)
                    n_gen_msgs,         # int (replicated)
                    sim_states_init_pmap, # LobState (device-split)
                    rng_keys_pmap,      # (num_devices, per_device, 2) random keys
                    hidden_state_pmap,  # hidden state tuple (device-split)
                    is_first_iteration, # conditional (replicated)
                    init_time_pmap,     # (num_devices, per_device, 2) init time
                    False,              # debug_book=False (replicated)
                    None,               # b_seq_real=None (replicated)
                    token_mode,         # int (replicated)
                )

                # Reshape outputs back: (num_devices, per_device, ...) -> (batch, ...)
                msgs_decoded = inference.reshape_from_pmap(msgs_decoded_pmap, batch_size)
                l2_book_states = inference.reshape_from_pmap(l2_book_states_pmap, batch_size)
                num_errors = num_errors_pmap.sum()  # Sum errors across devices
                msgs_tokens = inference.reshape_from_pmap(msgs_tokens_pmap, batch_size)
                hidden_state = inference.reshape_hidden_from_pmap(hidden_state_pmap, batch_size)
            else:
                # Use single-GPU vmap version
                msgs_decoded, l2_book_states, num_errors, msgs_tokens, hidden_state = inference.generate_batched(
                    sim_init,           # OrderBook instance
                    train_state,        # model params
                    model_unbatched,    # unbatched model - vmap handles batching
                    batchnorm,          # bool
                    encoder,            # dict (not modified for batching - generate_batched handles it)
                    sample_top_n,       # int
                    tick_size,          # int
                    m_seq_inp,          # (batch, seq_len) tokens - full seq for first iter, 1 token for rest
                    b_seq_inp,          # (batch, n_msgs, book_dim) - full seq for first iter, 1 for rest
                    n_gen_msgs,         # int - messages to generate
                    sim_states_init,    # LobState
                    rng_keys,           # (batch, 2) random keys
                    hidden_state,       # hidden state tuple (batched)
                    is_first_iteration, # conditional - True for first iter, False for subsequent
                    init_time,          # (batch, 2) init time
                    False,              # debug_book=False
                    None,               # b_seq_real=None
                    token_mode,         # int
                )

            # Convert msgs_tokens to m_seq_gen format (flatten tokens)
            m_seq_gen = msgs_tokens.reshape(batch_size, -1)  # (batch, n_gen_msgs * MSG_LEN)

            # For b_seq_gen, we need to reconstruct from l2_book_states
            # l2_book_states shape: (batch, n_gen_msgs, 40) - raw L2 states
            # Transform to volume series representation
            # Note: msgs_decoded contains the decoded messages
            b_seq_gen = None  # Will be computed if needed

            m_seq_gen_doubled = m_seq_gen
            b_seq_gen_doubled = b_seq_inp  # Keep input book seq for now, will be updated
            msgs_decoded_doubled = msgs_decoded
            l2_book_states_halved = l2_book_states[:, -1, :40]

            print(f'\n\nsuccessfully generated iteration no. {iteration}')
            
            midprices_batch = track_midprices_during_messages(
                msgs_decoded_doubled,
                l2_book_states[:, 0, :40],
                tick_size,
                midprice_step_size,
            )
            midprices.extend(list(midprices_batch))
            proc_msgs_numb += msgs_decoded_doubled.shape[1]

 
            if iteration <= num_insertions:
                print(">> INSERTING CUSTOM ORDER")
                m_seq_gen_doubled, b_seq_gen_doubled, msgs_decoded_doubled, l2_book_states_halved, p_mid_new = insert_custom_end(
                    m_seq_gen_doubled,
                    b_seq_gen_doubled,
                    msgs_decoded_doubled,
                    l2_book_states,
                    encoder,
                    midprices[-1],
                    tick_size,
                    EVENT_TYPE_i,
                    DIRECTION_i,
                    order_volume,
                    use_relative_volume,
                    order_volume_ratio,
                )
                midprices.append(jnp.squeeze(p_mid_new, axis=-1))
                proc_msgs_numb += 1
                
                m_seq_np = np.array(jax.device_get(m_seq_gen_doubled))
                b_seq_np = np.array(jax.device_get(b_seq_gen_doubled))
                msgs_decoded_np = np.array(jax.device_get(msgs_decoded_doubled))
                mid_price_np = np.array(jax.device_get(midprices))
                l2_book_states_np = np.array(jax.device_get(l2_book_states_halved))
        

                m_seq_gen_doubled = m_seq_gen_doubled[:, Message_Tokenizer.MSG_LEN:]
                msgs_decoded_doubled = msgs_decoded_doubled[:, 1:, :]
                b_seq_gen_doubled = b_seq_gen_doubled[:, 1:, :]

            if iteration > num_insertions:
                print(f'\nGENERATE FORWARD WITHOUT AGGRESSIVE ORDER - {iteration}\n\n')
                l2_book_states_halved = l2_book_states

                m_seq_np = np.array(jax.device_get(m_seq_gen_doubled))
                b_seq_np = np.array(jax.device_get(b_seq_gen_doubled))
                msgs_decoded_np = np.array(jax.device_get(msgs_decoded_doubled))
                l2_book_states_np = np.array(jax.device_get(l2_book_states_halved))
                mid_price_np = np.array(jax.device_get(midprices))

            # Save only the same artifacts as heuristic_historical_scenario_run_quantile_fixing.py
            np.save(os.path.join(base_save_folder, 'msgs_decoded_doubled', f'msgs_decoded_doubled_batch_{batch_i}_iter_{iteration}.npy'), msgs_decoded_np)
            np.save(os.path.join(base_save_folder, 'mid_price', f'mid_price_batch_{batch_i}_iter_{iteration}.npy'), mid_price_np)
            np.save(os.path.join(base_save_folder, 'b_seq_gen_doubled', f'b_seq_gen_doubled_batch_{batch_i}_iter_{iteration}.npy'), b_seq_np)

def create_next_experiment_folder(save_folder: str) -> Path:
    base = Path(save_folder)
    if not base.exists():
        raise FileNotFoundError(f"Directory {save_folder!r} does not exist")
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

# def parse_args():
#     p = argparse.ArgumentParser(description="Run LOB inference scenario")
#     p.add_argument(
#         "--config", "-c", 
#         type=str, 
#         default="/app/1_run_exp_aggresive_scenario.yaml",
#         help="Path to your YAML config file"
#     )
#     return p.parse_args()

# def main():
#     print(f"JAX backend platform: {jax.lib.xla_bridge.get_backend().platform}")
    
#     args = parse_args()
#     # load YAML config
#     with open(args.config, "r") as f:
#         cfg = yaml.safe_load(f)

def parse_args():
    p = argparse.ArgumentParser(description="Run LOB inference scenario")
    p.add_argument(
        "--config", "-c",
        type=str,
        default="1_run_exp_aggresive_scenario",
        help="Name of the config file (without .yaml extension) or full path to YAML config file"
    )
    p.add_argument(
        "--start-batch",
        type=int,
        default=None,
        help="Override start_batch from config (for parallel job submission)"
    )
    p.add_argument(
        "--end-batch",
        type=int,
        default=None,
        help="Override end_batch from config (for parallel job submission)"
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Use specific GPU index (0, 1, 2, ...). If set, forces single-GPU mode."
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size from config. When using sample_file, takes first N samples from each batch."
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Note: --gpu is handled at the top of the file BEFORE JAX import
    # to properly set CUDA_VISIBLE_DEVICES

    # Get backend platform (compatible with newer JAX versions)
    try:
        import jax.extend
        platform = jax.extend.backend.get_backend().platform
    except (ImportError, AttributeError):
        platform = jax.lib.xla_bridge.get_backend().platform
    print(f"JAX backend platform: {platform}")
    
    # Determine config file path
    if args.config.endswith('.yaml'):
        # Full path provided
        config_path = args.config
    else:
        # Just the name provided, construct path
        config_path = f"/app/{args.config}.yaml"
    
    print(f"Loading config from: {config_path}")
    
    # load YAML config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Check JAX devices and log to console and file
    jax_devices = jax.devices()
    accelerator_types = set(d.device_kind for d in jax_devices)
    device_summary = [f"{d.id}: {d.device_kind}" for d in jax_devices]
    device_message = f"JAX devices detected: {device_summary}"

    if platform == "gpu":
        log_message = f"✅ Running on GPU(s): {device_message}"
    else:
        log_message = f"⚠️ Running on CPU only: {device_message}"

    print(log_message)
    logger.info(log_message)

    # Initialize WandB
    wandb.init(
        project="Aggressive_Scenario",    # ← fill in
        entity="george-nigm",     # ← fill in
        config=cfg
    )
    

    # Multi-host / multi-node JAX setup (expects env vars WORLD_SIZE, RANK, COORD_ADDR)
    if "WORLD_SIZE" in os.environ:
        import jax.distributed as jdist
        jdist.initialize(
            coordinator_address=os.environ["COORD_ADDR"],
            num_processes=int(os.environ["WORLD_SIZE"]),
            process_id=int(os.environ["RANK"])
        )

    # unpack config
    save_folder       = cfg["save_folder"]
    batch_size        = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    if args.batch_size is not None:
        print(f"CLI override: batch_size={batch_size} (config had {cfg['batch_size']})")
    n_samples         = cfg["n_samples"]
    n_gen_msgs        = cfg["n_gen_msgs"]
    midprice_step_size= cfg["midprice_step_size"]
    num_insertions    = cfg["num_insertions"]
    num_coolings      = cfg["num_coolings"]
    EVENT_TYPE_i      = cfg["EVENT_TYPE_i"]
    DIRECTION_i       = cfg["DIRECTION_i"]
    order_volume      = cfg["order_volume"]
    bsz               = cfg["bsz"]
    # num_devices       = cfg["num_devices"]
    n_messages        = cfg["n_messages"]
    book_dim          = cfg["book_dim"]
    n_vol_series      = cfg["n_vol_series"]
    sample_top_n      = cfg["sample_top_n"]
    model_size        = cfg["model_size"]
    data_dir          = cfg["data_dir"]
    sample_all        = cfg["sample_all"]
    stock             = cfg["stock"]
    tick_size         = cfg["tick_size"]
    rng_seed          = cfg["rng_seed"]
    ckpt_path         = cfg["ckpt_path"]
    use_relative_volume = cfg["use_relative_volume"]
    order_volume_ratio = cfg["order_volume_ratio"]

    # num_devices  = jax.local_device_count() 
    # print(f'num_devices: ', num_devices)

    use_sample_file  = cfg["use_sample_file"]
    sample_file_path = cfg["sample_file_path"]
    # Allow CLI override for batch range (useful for parallel job submission on HPC)
    start_batch      = args.start_batch if args.start_batch is not None else cfg["start_batch"]
    end_batch        = args.end_batch if args.end_batch is not None else cfg["end_batch"]
    if args.start_batch is not None or args.end_batch is not None:
        print(f"CLI override: start_batch={start_batch}, end_batch={end_batch}")
    
    num_devices = jax.local_device_count()
    print(f'num_devices: {num_devices}')
    print(f"Using vmap-only generation (no pmap sharding required)")

    
    # Experiment folder
    exp_folder = create_next_experiment_folder(save_folder)
    print("Experiment dir:", exp_folder)
    wandb.run.summary["experiment_dir"] = str(exp_folder)

    with open(exp_folder / "used_config.yaml", "w") as f_out:
        yaml.dump(cfg, f_out)
    
    # Setup logging to experiment folder
    log_file_path = exp_folder / "job.log"
    print(f"Redirecting all output to: {log_file_path}")
    
    # Redirect stdout and stderr to log file
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    # Print initial info to log
    print(f"Experiment started at: {datetime.now()}")
    print(f"Experiment folder: {exp_folder}")
    print(f"Configuration: {cfg}")
    print("=" * 80)

    # Log config file as artifact
    artifact = wandb.Artifact(name="used_config", type="config")
    artifact.add_file(str(exp_folder / "used_config.yaml"))
    wandb.log_artifact(artifact)

    # Load metadata and model
    print("Loading metadata from", ckpt_path)
    args_ckpt = load_metadata(ckpt_path)

    # Get token_mode from checkpoint metadata (default to 22 - original vocab size)
    # The checkpoint was trained with token_mode=22 (vocab size 12012 with 4 special tokens)
    token_mode = getattr(args_ckpt, 'token_mode', 22)
    print(f"Using token_mode: {token_mode}")

    # Create Vocab with correct token_mode
    vocab = Vocab(token_mode=token_mode)
    n_classes = len(vocab)

    print("Initializing model...")
    train_state, model_cls, model_cls_unbatched = init_train_state(
        args_ckpt,
        n_classes=n_classes,
        seq_len=n_messages * Message_Tokenizer.MSG_LEN,
        book_dim=book_dim,
        book_seq_len=n_messages,
    )
    print("Loading checkpoint...")
    ckpt = load_checkpoint(train_state, ckpt_path, train=False)
    state = ckpt["model"]
    print(f"  Loaded state type: {type(state)}")
    print(f"  Params keys: {list(state.params.keys())[:3]}...")

    # Batched model for other operations
    model = model_cls(training=False, step_rescale=1.0)
    # Unbatched model for generate_batched (which uses vmap for batching)
    model_unbatched = model_cls_unbatched(training=False, step_rescale=1.0)

    # override devices & batch
    args_ckpt.bsz = bsz
    args_ckpt.num_devices = num_devices

    # prepare RNG
    rng = jax.random.PRNGKey(rng_seed)

    # data directory
    data_path = Path(data_dir) / stock
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_path} ({len(list(data_path.iterdir()))} files)")

    # get dataset
    ds = inference.get_dataset(data_path, n_messages, n_gen_msgs)
    wandb.log({"dataset_size": len(list(data_path.iterdir()))})

    # run generation
    results = run_generation_scenario(
        n_samples,
        batch_size,
        ds,
        rng,
        n_messages * Message_Tokenizer.MSG_LEN,
        n_messages,
        n_gen_msgs,
        state,
        model,
        args_ckpt.batchnorm,
        vocab.ENCODING,
        stock,
        n_vol_series,
        exp_folder,
        tick_size,
        sample_top_n,
        sample_all,
        num_insertions,
        num_coolings,
        midprice_step_size,
        EVENT_TYPE_i,
        DIRECTION_i,
        order_volume,
        use_relative_volume,
        order_volume_ratio,
        use_sample_file,
        sample_file_path,
        start_batch,
        end_batch,
        # New parameters for autoregressive generation
        args_ckpt=args_ckpt,
        token_mode=token_mode,
        model_unbatched=model_unbatched,  # Pass unbatched model for generate_batched
        num_devices=num_devices,  # Use all available GPUs for parallel generation
    )

    # log any returned metrics/artifacts
    wandb.log({"finished": True})
    wandb.save(str(exp_folder / "*"))
    
    # Close log file and restore stdout/stderr
    print(f"Experiment completed at: {datetime.now()}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Logs saved to: {log_file_path}")

if __name__ == "__main__":
    main()