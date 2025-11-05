#!/usr/bin/env python
"""
CST-based aggressive scenario generation script.
Identical logic to 1_run_exp_aggressive_scenario_whole_lvl_copy.py but uses
CST (Cont-Stoikov-Talreja) model instead of S5 for generation.
"""
import os
import sys
import argparse
import yaml

# prevent XLA pre-allocation if using TPU/GPU backends
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import wandb
import jax
from pathlib import Path
from datetime import datetime
import numpy as np
from functools import partial
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
import logging
import json
import pickle

# Add CST model imports
cst_model_path = os.path.join(os.path.dirname(__file__), 'lob_bench', 'cst_model')
sys.path.append(cst_model_path)
import cst
from cst import CSTParams, Book

# Import CST utilities
try:
    from param_estimation import load_params
except ImportError:
    # Try alternative import
    sys.path.append(cst_model_path)
    import param_estimation
    load_params = param_estimation.load_params

try:
    from lobster_conversion import (
        LobsterMessage, msg_to_jaxlob, update_oid,
        make_step_book_scannable as make_cst_step_scannable
    )
except ImportError:
    # Try alternative import
    import lobster_conversion
    LobsterMessage = lobster_conversion.LobsterMessage
    make_cst_step_scannable = lobster_conversion.make_step_book_scannable

# Standard LOB imports
from lob.encoding import Vocab, Message_Tokenizer
from lob import inference_no_errcorr as inference
from preproc import transform_L2_state
import lob.encoding as encoding
from lob.lobster_dataloader import LOBSTER_Dataset
from jax import lax
import jax.numpy as jnp

# Add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import Configuration as JaxLOBConfig
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob_bench.cst_model import lobster_conversion

logger = logging.getLogger(__name__)

START_OID = 100
SIM_CONFIG = JaxLOBConfig()


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


def convert_lobster_msg_to_raw(lobster_msg: LobsterMessage, mid_price: jax.Array, encoder) -> jax.Array:
    """
    Convert CST LobsterMessage to raw message format compatible with existing pipeline.
    
    Format: [order_id, event_type, direction, price_abs, visible, size, ...]
    """
    # Convert time to seconds and nanoseconds
    time_s = jnp.int32(lobster_msg.time)
    time_ns = jnp.int32((lobster_msg.time - time_s) * 1e9)
    
    # Calculate relative price
    price_rel = (lobster_msg.price - mid_price) // 100  # tick_size
    
    # Construct raw message in format expected by the pipeline
    # Format: [oid, event_type, direction, price_abs, visible, size, ...]
    raw_msg = jnp.array([
        lobster_msg.oid,           # 0: order_id
        lobster_msg.event_type,    # 1: event_type
        jnp.where(lobster_msg.direction == 1, 1, 0),  # 2: direction (1=bid, 0=ask)
        lobster_msg.price,         # 3: price_abs
        1,                         # 4: visible flag
        lobster_msg.size,          # 5: size
        price_rel,                 # 6: price (relative)
        0,                         # 7: delta_t_s (will be computed)
        time_s,                    # 8: time_s
        time_ns,                   # 9: time_ns
        0,                         # 10: price_ref (NA)
        0,                         # 11: size_ref (NA)
        0,                         # 12: time_s_ref (NA)
        0,                         # 13: time_ns_ref (NA)
    ], dtype=jnp.int32)
    
    return raw_msg


def convert_raw_to_lobster_format(raw_msgs: jax.Array) -> jax.Array:
    """
    Convert raw messages to format compatible with the pipeline.
    This ensures compatibility with existing insert_custom_end function.
    """
    # raw_msgs shape: (batch, T, 14)
    # Already in correct format from convert_lobster_msg_to_raw
    return raw_msgs


def generate_cst_batched(
        sim_init,
        books: Any,  # (batch,) CST Book objects (can be pytree)
        base_rates: jax.Array,
        params: CSTParams,
        sim_states: LobState,  # (batch, ...)
        rng_keys: jax.Array,  # (batch, 2)
        n_gen_msgs: int,
        n_levels: int = 20,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Generate messages using CST model for a batch of initial states.
    
    Returns:
        m_seq_gen: Encoded message sequences (batch, seq_len)
        b_seq_gen: Book sequences (batch, T, book_dim)
        msgs_decoded: Raw decoded messages (batch, T, msg_dim)
        l2_book_states: L2 book states (batch, T, n_levels*4)
        num_errors: Number of errors (batch,) - always 0 for CST
    """
    # Create scan function for CST generation
    cst_step = make_cst_step_scannable(n_levels)
    
    def generate_single(book, sim_state, rng_key):
        """Generate for a single sample in the batch."""
        # Create sim and initial state
        sim = OrderBook(SIM_CONFIG)
        
        # Get initial L2 state from book
        l2_init, init_time = cst.get_l2_book(book, params, n_levels)
        sim_state_init = sim.reset(l2_init.flatten())
        
        # Generate using scan; provide a dummy xs for iteration count
        final_carry, (l2_books, lobster_msgs) = jax.lax.scan(
            cst_step,
            (sim, sim_state_init, book, base_rates, params, START_OID, rng_key),
            xs=jnp.arange(n_gen_msgs, dtype=jnp.int32),
        )
        
        # Get mid prices for each step for encoding messages
        def get_mid_price(l2):
            l2_reshaped = l2.reshape(n_levels, 4)
            best_ask = l2_reshaped[0, 0]
            best_bid = l2_reshaped[0, 2]
            mid = jax.lax.cond(
                (best_ask > 0) & (best_bid > 0),
                lambda: (best_ask + best_bid) // 2,
                lambda: jax.lax.cond(
                    best_ask > 0,
                    lambda: best_ask - params.tick_size,
                    lambda: jax.lax.cond(
                        best_bid > 0,
                        lambda: best_bid + params.tick_size,
                        lambda: jnp.int32(0)
                    )
                )
            )
            return mid
        
        # l2_books shape: (T, n_levels*4)
        mid_prices = jax.vmap(get_mid_price)(l2_books.reshape(n_gen_msgs, n_levels * 4))
        
        # Convert lobster messages - they are LobsterMessage dataclass objects
        # Extract fields from dataclass structure
        # lobster_msgs is a pytree with fields: time, event_type, oid, size, price, direction
        def extract_and_convert(msg_struct, mid_p):
            time_total = msg_struct.time
            time_s = jnp.int32(time_total)
            time_ns = jnp.int32((time_total - time_s) * 1e9)
            price_rel = (msg_struct.price - mid_p) // params.tick_size
            direction_code = jnp.where(msg_struct.direction == 1, 1, 0)
            
            raw_msg = jnp.array([
                msg_struct.oid,
                msg_struct.event_type,
                direction_code,
                msg_struct.price,
                1,  # visible
                msg_struct.size,
                price_rel,
                0,  # delta_t_s
                time_s,
                time_ns,
                0,  # price_ref
                0,  # size_ref
                0,  # time_s_ref
                0,  # time_ns_ref
            ], dtype=jnp.int32)
            return raw_msg
        
        # Use vmap to convert all messages
        raw_msgs = jax.vmap(extract_and_convert)(lobster_msgs, mid_prices)
        
        # Convert L2 books to the shape expected (T, n_levels*4)
        l2_books_reshaped = l2_books.reshape(n_gen_msgs, n_levels * 4)
        
        # Create encoded sequence from raw messages
        encoder = Vocab().ENCODING
        encoded_msgs = jax.vmap(lambda m: encoding.encode_msg(m, encoder))(raw_msgs)
        # Flatten encoded messages to sequence: (T * MSG_LEN,)
        m_seq_gen = encoded_msgs.reshape(-1)
        
        # Create book sequence with p_change and transform
        # Get initial mid price
        l2_init_flat = l2_init.flatten() if len(l2_init.shape) > 1 else l2_init
        p_mid_init = get_mid_price(l2_init_flat)
        
        def make_book_seq(i, p_mid_prev):
            p_mid_curr = mid_prices[i]
            p_change = ((p_mid_curr - p_mid_prev) // params.tick_size).astype(jnp.int32)
            l2_book = l2_books_reshaped[i]
            book_raw = jnp.concatenate([jnp.array([p_change]), l2_book])
            book_seq = transform_L2_state(book_raw[None, :], 500, 100)[0]
            return book_seq, p_mid_curr
        
        # Use scan to build book sequences
        def book_scan_fn(carry, i):
            p_mid_prev = carry
            book_seq, p_mid_new = make_book_seq(i, p_mid_prev)
            return p_mid_new, book_seq
        
        _, book_seqs = jax.lax.scan(book_scan_fn, p_mid_init, jnp.arange(n_gen_msgs))
        b_seq_gen = book_seqs
        
        return m_seq_gen, b_seq_gen, raw_msgs, l2_books_reshaped, jnp.int32(0)
    
    # Handle batch processing
    # books is a pytree (Book dataclass), JAX vmap should handle it
    # Get batch size from rng_keys
    if len(rng_keys.shape) > 0:
        batch_size = rng_keys.shape[0]
    else:
        batch_size = 1
    
    # Use vmap - JAX should handle Book dataclass as pytree
    generate_vmap = jax.vmap(generate_single, in_axes=(0, 0, 0))
    m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = generate_vmap(
        books, sim_states, rng_keys
    )
    
    return m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors


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
        order_volume=75,
        use_relative_volume=False,
        order_volume_ratio=1.0,
    ):
    """
    Insert custom aggressive order at the end. Same logic as original.
    """
    ORDER_ID_i = 77777777
    sim_init, sim_states_init = inference.get_sims_vmap(
        l2_book_states_halved[:, -2] if l2_book_states_halved.shape[1] > 1 else l2_book_states_halved[:, 0],
        msgs_decoded_doubled[:, -1:]
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
    best_bid_ask = jax.vmap(sim_init.get_best_bid_and_ask_inclQuants)(sim_states_init)

    # pick available volume on the level we will consume
    avail = jnp.where(
        DIRECTION_i == 0,
        best_bid_ask[1][:, 1],  # ask vol for a buy
        best_bid_ask[0][:, 1],  # bid vol for a sell
    ).astype(jnp.int32)

    # === SIZING RULE ===
    if use_relative_volume:
        ratio = jnp.clip(jnp.float32(order_volume_ratio), 0.0, 1.0)
        SIZE_i = jnp.floor(ratio * jnp.float32(avail)).astype(jnp.int32)
        SIZE_i = jnp.where((avail > 0) & (SIZE_i == 0), 1, SIZE_i)
    else:
        SIZE_i = jnp.minimum(jnp.int32(order_volume), avail)

    batched_quantity   = SIZE_i
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

    # Use current L2 dimensionality to determine number of levels
    current_levels = l2_book_states_halved.shape[-1] // 4
    book_l2 = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(new_sim_state, current_levels)
    new_l2_book_states_halved = jnp.concatenate([l2_book_states_halved, book_l2[:, None, :]], axis=1)
    new_book_raw = jnp.concatenate([p_change, book_l2], axis=1)
    new_book_raw = new_book_raw[:, None, :]

    transform_L2_state_batch = jax.jit(
        jax.vmap(transform_L2_state, in_axes=(0, None, None)), static_argnums=(1, 2)
    )
    new_book = transform_L2_state_batch(new_book_raw, 500, 100)
    b_seq_gen_doubled = jnp.concatenate([b_seq_gen_doubled, new_book], axis=1)

    ins_msg = jnp.concatenate([
        batched_new_order_id.reshape(-1, 1),
        batched_EVENT_TYPE.reshape(-1, 1),
        batched_side.reshape(-1, 1),
        batched_p_abs.reshape(-1, 1),
        jnp.full((batch_size, 1), 1, dtype=jnp.int32),
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


def run_generation_scenario(
        n_samples: int,
        batch_size: int,
        ds: LOBSTER_Dataset,
        rng: jax.dtypes.prng_key,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        params: CSTParams,
        base_rates: jax.Array,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        stock_symbol: str,
        n_vol_series: int = 500,
        save_folder: str = './data_saved/',
        tick_size: int = 100,
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
        n_levels: int = 20,
    ):

    rng, rng_ = jax.random.split(rng)

    if use_sample_file:
        assert sample_file_path is not None, "Path to sample file not provided"
        with open(sample_file_path, "r") as f:
            sample_i_full = json.load(f)
        sample_i = sample_i_full[start_batch: end_batch if end_batch != -1 else None]
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

    transform_L2_state_batch = jax.jit(jax.vmap(transform_L2_state, in_axes=(0, None, None)), static_argnums=(1, 2))

    num_iterations = num_insertions + num_coolings

    print('sample_i:', sample_i)

    for batch_i in tqdm(sample_i):
        print('BATCH', batch_i)
        proc_msgs_numb = -n_msgs

        for iteration in range(1, num_iterations+1):
            print('\nITERATION ', iteration)
            midprices = []
            
            if iteration == 1:
                m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
                m_seq = jnp.array(m_seq)
                b_seq_pv = jnp.array(b_seq_pv)
                msg_seq_raw = jnp.array(msg_seq_raw)
                book_l2_init = jnp.array(book_l2_init)
                b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)

                m_seq_inp = m_seq[:, : seq_len]
                b_seq_inp = b_seq[: , : n_msgs]
                m_seq_raw_inp = msg_seq_raw[:, : n_msgs]

                m_seq_np = np.array(jax.device_get(m_seq_inp))
                b_seq_np = np.array(jax.device_get(b_seq_inp))
                msgs_decoded_np = np.array(jax.device_get(m_seq_raw_inp))
                l2_book_states_np = np.array(jax.device_get(book_l2_init))

                # Do not save intermediate inputs or initial states

                sim_init, sim_states_init = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp)

                # Initialize CST books from L2 states
                # Convert L2 book to CST Book format
                # book_l2_init shape: (batch, n_levels*4) - flattened L2
                def l2_to_cst_book(l2_state):
                    # Infer levels from input length to avoid shape mismatches
                    inferred_levels = l2_state.shape[0] // 4
                    # Reshape to (levels, 4) format: [ask_p, ask_q, bid_p, bid_q] per level
                    l2_reshaped = l2_state.reshape(inferred_levels, 4)
                    # Convert to flat array for cst.init_book: [ask_p_1, ask_q_1, bid_p_1, bid_q_1, ...]
                    l2_flat = l2_reshaped.flatten()
                    # Get time from last message
                    init_time = float(0.0)
                    return cst.init_book(
                        l2_flat,
                        params,
                        init_time
                    )
                
                cst_books = jax.vmap(l2_to_cst_book)(book_l2_init)

                midprices_batch = track_midprices_during_messages(
                    m_seq_raw_inp,
                    book_l2_init,
                    tick_size,
                    midprice_step_size,
                )
                midprices = list(midprices_batch)
                proc_msgs_numb += m_seq_raw_inp.shape[1]

            else: 
                # Use previous iteration's generated data
                m_seq = m_seq_gen_doubled
                b_seq = b_seq_gen_doubled
                msg_seq_raw = msgs_decoded_doubled
                m_seq_inp = m_seq
                b_seq_inp = b_seq
                m_seq_raw_inp = msg_seq_raw
                
                # Reconstruct CST books from last L2 state
                def l2_to_cst_book(l2_state):
                    # Infer levels from input length to avoid shape mismatches
                    inferred_levels = l2_state.shape[0] // 4
                    # Reshape to (levels, 4) format
                    l2_reshaped = l2_state.reshape(inferred_levels, 4)
                    # Convert to flat array for cst.init_book
                    l2_flat = l2_reshaped.flatten()
                    # Get time from last message
                    init_time = float(msgs_decoded_doubled[0, -1, 8] + msgs_decoded_doubled[0, -1, 9] / 1e9)
                    return cst.init_book(
                        l2_flat,
                        params,
                        init_time
                    )
                
                # Get last L2 state (handle shape)
                # l2_book_states_halved shape: (batch, n_levels*4) or (batch, T, n_levels*4)
                if len(l2_book_states_halved.shape) == 2:
                    last_l2 = l2_book_states_halved
                else:
                    last_l2 = l2_book_states_halved[:, -1]
                cst_books = jax.vmap(l2_to_cst_book)(last_l2)
                
                sim_init, sim_states_init = inference.get_sims_vmap(
                    last_l2,
                    m_seq_raw_inp[:, -1:]
                )
            
            # Create RNG keys for batch
            rng, rng_ = jax.random.split(rng)
            rng_keys = jax.random.split(rng_, batch_size)
            
            # base_rates should be the same for all samples in batch
            # Expand if needed (CST uses same rates across batch)
            if len(base_rates.shape) == 0 or base_rates.shape[0] == 1:
                # Broadcast to batch if needed
                base_rates_batch = base_rates
            else:
                base_rates_batch = base_rates

            # Generate using CST
            print("Generating with CST model...")
            # Determine levels for this batch from the L2 state shape
            if iteration == 1:
                levels_for_batch = book_l2_init.shape[1] // 4
            else:
                base_l2 = last_l2 if 'last_l2' in locals() else l2_book_states_halved
                levels_for_batch = base_l2.shape[-1] // 4
            print(f"  Batch size: {batch_size}, n_gen_msgs: {n_gen_msgs}, n_levels: {levels_for_batch}")
            try:
                m_seq_gen, b_seq_gen, msgs_decoded, l2_book_states, num_errors = generate_cst_batched(
                    sim_init,
                    cst_books,
                    base_rates_batch,
                    params,
                    sim_states_init,
                    rng_keys,
                    n_gen_msgs,
                    levels_for_batch,
                )
            except Exception as e:
                print(f"ERROR in generate_cst_batched: {e}")
                import traceback
                traceback.print_exc()
                raise

            # Ensure m_seq_gen is properly shaped (needed for compatibility)
            # m_seq_gen from CST is flattened, need to reshape to match expected format
            # Expected: (batch, seq_len) where seq_len = n_gen_msgs * MSG_LEN
            if len(m_seq_gen.shape) == 1:
                # Reshape to (batch, seq_len)
                msg_len = Message_Tokenizer.MSG_LEN
                expected_len = n_gen_msgs * msg_len
                if m_seq_gen.shape[0] == batch_size * expected_len:
                    m_seq_gen = m_seq_gen.reshape(batch_size, expected_len)
                else:
                    # Pad or truncate as needed
                    current_len = m_seq_gen.shape[0] // batch_size
                    if current_len < expected_len:
                        # Pad
                        pad_len = expected_len - current_len
                        pad_tokens = jnp.full((batch_size, pad_len), Vocab().NA_TOK, dtype=jnp.int32)
                        m_seq_gen = jnp.concatenate([m_seq_gen.reshape(batch_size, -1), pad_tokens], axis=1)
                    else:
                        m_seq_gen = m_seq_gen.reshape(batch_size, -1)[:, :expected_len]

            m_seq_gen_doubled = m_seq_gen
            b_seq_gen_doubled = b_seq_gen
            msgs_decoded_doubled = msgs_decoded
            # Keep only the active levels from the generated L2 (dynamic)
            gen_levels = (l2_book_states.shape[-1] // 4) if len(l2_book_states.shape) >= 2 else n_levels
            l2_book_states_halved = l2_book_states[:, -1, : gen_levels * 4] if len(l2_book_states.shape) == 3 else l2_book_states

            print(f'\n\nsuccessfully generated iteration no. {iteration}')
            
            # Get initial state for tracking midprices
            init_l2_for_tracking = l2_book_states[:, 0, : gen_levels * 4] if len(l2_book_states.shape) == 3 else l2_book_states[0]
            
            midprices_batch = track_midprices_during_messages(
                msgs_decoded_doubled,
                init_l2_for_tracking[None, :] if len(init_l2_for_tracking.shape) == 1 else init_l2_for_tracking,
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
                if len(l2_book_states.shape) == 3:
                    gen_levels = l2_book_states.shape[-1] // 4
                    l2_book_states_halved = l2_book_states[:, -1, : gen_levels * 4]
                else:
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


def parse_args():
    p = argparse.ArgumentParser(description="Run LOB inference scenario with CST model")
    p.add_argument(
        "--config", "-c", 
        type=str, 
        default="1_run_exp_aggresive_scenario",
        help="Name of the config file (without .yaml extension) or full path to YAML config file"
    )
    return p.parse_args()


def main():
    print(f"JAX backend platform: {jax.lib.xla_bridge.get_backend().platform}")

    args = parse_args()
    
    # Determine config file path
    if args.config.endswith('.yaml'):
        config_path = args.config
    else:
        config_path = f"/app/{args.config}.yaml"
    
    print(f"Loading config from: {config_path}")
    
    # load YAML config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Check JAX devices
    jax_devices = jax.devices()
    accelerator_types = set(d.device_kind for d in jax_devices)
    device_summary = [f"{d.id}: {d.device_kind}" for d in jax_devices]
    device_message = f"JAX devices detected: {device_summary}"

    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        log_message = f"✅ Running on GPU(s): {device_message}"
    else:
        log_message = f"⚠️ Running on CPU only: {device_message}"

    print(log_message)
    logger.info(log_message)

    # Initialize WandB
    wandb.init(
        project="Aggressive_Scenario_CST",
        entity="george-nigm",
        config=cfg
    )

    # unpack config
    save_folder       = cfg["save_folder"]
    batch_size        = cfg["batch_size"]
    n_samples         = cfg["n_samples"]
    n_gen_msgs        = cfg["n_gen_msgs"]
    midprice_step_size= cfg["midprice_step_size"]
    num_insertions    = cfg["num_insertions"]
    num_coolings      = cfg["num_coolings"]
    EVENT_TYPE_i      = cfg["EVENT_TYPE_i"]
    DIRECTION_i       = cfg["DIRECTION_i"]
    order_volume      = cfg["order_volume"]
    n_messages        = cfg["n_messages"]
    book_dim          = cfg["book_dim"]
    n_vol_series      = cfg["n_vol_series"]
    data_dir          = cfg["data_dir"]
    sample_all        = cfg["sample_all"]
    stock             = cfg["stock"]
    tick_size         = cfg["tick_size"]
    rng_seed          = cfg["rng_seed"]
    use_relative_volume = cfg["use_relative_volume"]
    order_volume_ratio = cfg["order_volume_ratio"]

    # CST-specific config
    cst_params_file   = cfg.get("cst_params_file", "lob_bench/cst_model/params/model_params.pkl")
    n_levels          = cfg.get("n_levels", 20)

    use_sample_file  = cfg["use_sample_file"]
    sample_file_path = cfg["sample_file_path"]
    start_batch      = cfg["start_batch"]
    end_batch        = cfg["end_batch"]
    
    num_devices = jax.local_device_count()
    print(f'num_devices: ', num_devices)
    
    # Validate batch_size
    if batch_size % num_devices != 0:
        print(f"Warning: batch_size ({batch_size}) is not divisible by num_devices ({num_devices})")
        raise ValueError(f"batch_size ({batch_size}) must be divisible by num_devices ({num_devices})")
    
    print(f"✅ batch_size ({batch_size}) is compatible with num_devices ({num_devices})")
    print(f"   Each GPU will process {batch_size // num_devices} samples")

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

    # Load CST parameters instead of S5 checkpoint
    cst_params_full_path = os.path.join(os.path.dirname(__file__), cst_params_file)
    if not os.path.exists(cst_params_full_path):
        # Try absolute path or relative from current directory
        if os.path.isabs(cst_params_file):
            cst_params_full_path = cst_params_file
        else:
            # Try in current directory
            alt_path = os.path.join(os.getcwd(), cst_params_file)
            if os.path.exists(alt_path):
                cst_params_full_path = alt_path
            else:
                # Try in app directory (Docker)
                alt_path = os.path.join("/app", cst_params_file)
                if os.path.exists(alt_path):
                    cst_params_full_path = alt_path
    
    if os.path.exists(cst_params_full_path):
        print("Loading CST parameters from", cst_params_full_path)
        aggr_params_dict = load_params(cst_params_full_path)
        params, lo_lambda, co_theta = cst.init_params(aggr_params_dict)
        print(f"CST Parameters loaded: tick_size={params.tick_size}, num_ticks={params.num_ticks}")
    else:
        print(f"WARNING: CST params file not found at {cst_params_full_path}")
        print("Please ensure the CST params file exists. The script will attempt to continue with defaults.")
        raise FileNotFoundError(
            f"CST parameters file not found: {cst_params_full_path}. "
            f"Please create the parameters file or update the path in config."
        )
    
    # Get base rates for CST
    base_rates = cst.get_event_base_rates(
        params,
        cancel_rates=co_theta,
        lo_rates=lo_lambda,
    )
    print(f"Base rates computed, shape: {base_rates.shape}")

    # prepare RNG
    rng = jax.random.PRNGKey(rng_seed)

    # data directory
    data_path = Path(data_dir) / stock
    data_path.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_path} ({len(list(data_path.iterdir()))} files)")

    # get dataset (same as S5 version)
    ds = inference.get_dataset(data_path, n_messages, n_gen_msgs)
    wandb.log({"dataset_size": len(list(data_path.iterdir()))})

    # run generation with CST
    results = run_generation_scenario(
        n_samples,
        batch_size,
        ds,
        rng,
        n_messages * Message_Tokenizer.MSG_LEN,
        n_messages,
        n_gen_msgs,
        params,
        base_rates,
        Vocab().ENCODING,
        stock,
        n_vol_series,
        exp_folder,
        tick_size,
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
        n_levels,
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

