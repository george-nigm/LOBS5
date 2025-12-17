#!/usr/bin/env python
"""
RWKV-based aggressive scenario generation script.
Identical insert-and-cooldown logic to 1_run_exp_aggressive_scenario_whole_lvl_cst.py
but uses an RWKV checkpoint for generative modeling of message streams.

Outputs both decoded messages and order book states.
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
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
import logging
import json

from jax import lax
import jax.numpy as jnp

# LOB utils (same pipeline pieces as CST script)
from lob.encoding import Vocab, Message_Tokenizer
from lob import inference_no_errcorr as inference
from preproc import transform_L2_state
import lob.encoding as encoding
from lob.lobster_dataloader import LOBSTER_Dataset

# Add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import Configuration as JaxLOBConfig

logger = logging.getLogger(__name__)

SIM_CONFIG = JaxLOBConfig()

# --- RWKV imports from lobgen (direct, without auto/tokenizers deps) ---
LOBGEN_DIR = os.path.join(os.path.dirname(__file__), "lobgen")
sys.path.append(LOBGEN_DIR)
from lobgen.jax_rwkv.jax_rwkv import base_rwkv as rwkv6_base

# parsing utilities from lobgen for generated strings
from lobgen.constants import MESSAGE_TOKEN_TYPES, MESSAGE_TOKEN_DTYPE_MAP
from lobgen.data_loading import _df_to_str


def process_long_seq(forward_jit, params, tokens, state, length, padding=128):
    full_instruction_length = tokens.shape[-1]
    instruction_len = length
    pad_needed = (padding - (full_instruction_length % padding)) % padding
    tokens_padded = jnp.pad(tokens, (0, pad_needed), constant_values=0)
    x = (jnp.zeros_like(params['emb']['weight'][:, 0]), state)

    def inner_loop(x, i):
        (true_out, state) = x
        cur_len = jnp.minimum(padding, instruction_len - i)
        out, new_state = forward_jit(jax.lax.dynamic_slice_in_dim(tokens_padded, i, padding), state, params, cur_len)
        state = jax.lax.cond(cur_len <= 0, lambda: state, lambda: new_state)
        true_out = jax.lax.cond(cur_len <= 0, lambda: true_out, lambda: out[jnp.maximum(cur_len - 1, 0)])
        return (true_out, state), 0

    (true_out, state), _ = jax.lax.scan(inner_loop, x, jnp.arange(0, tokens_padded.shape[-1], padding))
    return true_out, state


def softmax_temperature(logits: jax.Array, temperature: float) -> jax.Array:
    logits = logits / jnp.maximum(temperature, 1e-6)
    logits = logits - jnp.max(logits)
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits))


def sample_token(key: jax.Array, logits: jax.Array, temperature: float = 1.0) -> int:
    probs = softmax_temperature(logits, temperature)
    return int(jax.random.categorical(key, jnp.log(probs)))


def encode_context_from_msgs(msgs_decoded: jax.Array, n_msgs_context: int) -> str:
    """Build tokenizer input string from raw decoded messages using lobgen formatting."""
    if n_msgs_context <= 0 or msgs_decoded.shape[1] == 0:
        return ""
    # msgs_decoded shape: (batch, T, msg_dim). We take first element of batch.
    import pandas as pd
    arr = np.array(jax.device_get(msgs_decoded[0, :n_msgs_context]))
    df = pd.DataFrame(arr, columns=[
        '<order_id>', '<event_type>', '<direction>', '<price>', '<visible>', '<size>',
        '<price_rel>', '<delta_t_s>', '<time_s>', '<time_ns>', '<price_ref>',
        '<size_ref>', '<time_s_ref>', '<time_ns_ref>'
    ])
    # keep only known message token columns intersection with MESSAGE_TOKEN_TYPES
    cols = [c for c in MESSAGE_TOKEN_TYPES if c in df.columns]
    df = df[cols]
    batches = _df_to_str(df, n_msgs=n_msgs_context)
    return batches[0] if batches else ""


def parse_generated_to_arrays(s: str) -> Tuple[jax.Array, jax.Array]:
    import pandas as pd, re, numpy as _np
    # local lightweight parser to avoid importing transformers via lobgen.run_inference
    s = s.replace(",", "").replace(" ", "").replace("Ġ", "").replace("..", ".").replace("Ċ", "").replace("[PAD]", "").replace("[UNK]", "")
    df_gen_cols = MESSAGE_TOKEN_TYPES
    cols = "".join([f"{token}|" for token in df_gen_cols])
    pattern = rf"({cols})([^<]+)?"
    matches = re.findall(pattern, s)
    df_gen_cols = [col for col in df_gen_cols if col in set(map(lambda x: x[0], matches))]
    rows = []
    current_row = {}
    for col, value in matches:
        if col in current_row:
            rows.append(current_row)
            current_row = {}
        current_row[col] = value if value is not None else _np.nan
    if current_row:
        rows.append(current_row)
    df_gen = pd.DataFrame(rows, columns=df_gen_cols)
    # ensure all expected message columns exist
    for col in MESSAGE_TOKEN_TYPES:
        if col not in df_gen.columns:
            df_gen[col] = _np.nan
    df_gen = df_gen[MESSAGE_TOKEN_TYPES]
    dtype_map = {**{col: int for col in MESSAGE_TOKEN_TYPES}, **MESSAGE_TOKEN_DTYPE_MAP}
    df_gen = df_gen.astype(dtype_map, errors="ignore")
    df_gen = df_gen.dropna()
    # build raw message arrays in the order expected by pipeline
    cols = [
        '<order_id>', '<event_type>', '<direction>', '<price>', '<visible>', '<size>',
        '<price_rel>', '<delta_t_s>', '<time_s>', '<time_ns>', '<price_ref>',
        '<size_ref>', '<time_s_ref>', '<time_ns_ref>'
    ]
    for c in cols:
        if c not in df_gen.columns:
            df_gen[c] = 0
    df_gen = df_gen[cols].astype({c: int for c in cols})
    raw_msgs = jnp.array(df_gen.values, dtype=jnp.int32)[None, :, :]  # (1, T, 14)

    # encode messages to token ids used by LOB pipeline
    encoder = Vocab().ENCODING
    msg_encoded = jax.vmap(lambda m: encoding.encode_msg(m, encoder))(raw_msgs[0])
    m_seq_gen = msg_encoded.reshape(1, -1)
    return m_seq_gen, raw_msgs


def reconstruct_books_from_msgs(raw_msgs: jax.Array, tick_size: int, n_vol_series: int) -> Tuple[jax.Array, jax.Array]:
    """Simulate order book to get L2 states and book sequences compatible with the pipeline."""
    if raw_msgs.shape[1] == 0:
        # Avoid calling transform when empty; known output width for (500,100) settings is 501
        out_dim = 501
        return (
            jnp.zeros((raw_msgs.shape[0], 0, n_vol_series * 4), dtype=jnp.int32),
            jnp.zeros((raw_msgs.shape[0], 0, out_dim), dtype=jnp.int32),
        )
    # initialize simulator from the first message state by building a trivial L2 from the first message
    # Use pipeline helper to get sims and states from messages (requires an initial L2, use zeros to start)
    batch = raw_msgs.shape[0]
    # Build zero L2 state with minimal levels inferred later during transform
    # Here we simulate step-by-step
    sim_init = OrderBook(SIM_CONFIG)

    def roll_msgs(msg_seq):
        # fallback: ensure at least one message
        if msg_seq.shape[0] == 0:
            msg_seq = jnp.zeros((1, 14), dtype=jnp.int32)
        # Start from empty book
        state = sim_init.reset(jnp.zeros((n_vol_series * 4,), dtype=jnp.int32))
        l2_list = []
        for i in range(msg_seq.shape[0]):
            s_msg = inference.construct_sim_msg(
                msg_seq[i, 1],  # event
                msg_seq[i, 2],  # side
                msg_seq[i, 5],  # quantity
                msg_seq[i, 3],  # price abs
                msg_seq[i, 0],  # oid
                msg_seq[i, 8],  # time_s
                msg_seq[i, 9],  # time_ns
            )
            state = sim_init.process_order_array(state, s_msg)
            l2 = sim_init.get_L2_state(state, n_vol_series)
            l2_list.append(l2)
        return jnp.stack(l2_list, axis=0)

    l2_books = jax.vmap(roll_msgs)(raw_msgs)  # (batch, T, n_vol_series*4)

    # build book sequences (p_change + transformed L2)
    def get_mid_from_l2(l2):
        l2_reshaped = l2.reshape(n_vol_series, 4)
        best_ask = l2_reshaped[0, 0]
        best_bid = l2_reshaped[0, 2]
        mid = jax.lax.cond(
            (best_ask > 0) & (best_bid > 0),
            lambda: (best_ask + best_bid) // 2,
            lambda: jax.lax.cond(
                best_ask > 0,
                lambda: best_ask - tick_size,
                lambda: jax.lax.cond(best_bid > 0, lambda: best_bid + tick_size, lambda: jnp.int32(0)),
            ),
        )
        return mid

    mids = jax.vmap(jax.vmap(get_mid_from_l2))(l2_books)

    def make_book_seq_for_batch(l2_b, mids_b):
        p_changes = []
        prev = mids_b[0]
        seqs = []
        for i in range(l2_b.shape[0]):
            cur = mids_b[i]
            p_change = ((cur - prev) // tick_size).astype(jnp.int32)
            book_vec = jnp.concatenate([jnp.atleast_1d(p_change), jnp.atleast_1d(l2_b[i])])
            seq = transform_L2_state(book_vec, 500, 100)
            seqs.append(seq)
            prev = cur
        return jnp.stack(seqs, axis=0)

    b_seq = jax.vmap(make_book_seq_for_batch)(l2_books, mids)
    return l2_books, b_seq


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
        # Return empty array with same dtype as midprices (int32 for prices)
        return jnp.empty((0, m_seq_raw_inp.shape[0]), dtype=jnp.int32)

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
        # book_dim = n_vol_series * 4, so n_levels = carry.shape[1] // 4
        n_levels = carry.shape[1] // 4
        sim_init, sim_states = inference.get_sims_vmap(carry, msg_chunk)
        mid_price = inference.batched_get_safe_mid_price(
            sim_init, sim_states, tick_size
        )
        # Extract fresh L2 state to feed next step
        full_l2_state = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(
            sim_states, n_levels
        )
        # keep only the leading slice so the carry's shape matches the input
        new_l2_state = full_l2_state[:, : carry.shape[1]]
        return new_l2_state, mid_price

    _, midprices = lax.scan(scan_step, book_l2_init, msgs)  # (num_steps, batch)
    return midprices


def generate_rwkv_batch(
    tokenizer,
    RWKV,
    params,
    context_str: str,
    n_gen_msgs: int,
    temperature: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    # Initialize state with context
    forward_jit = jax.jit(RWKV.forward)
    state = RWKV.default_state(params)
    if not context_str:
        # build a minimal prompt with two lines of LOB tokens and zero values
        def _one_msg():
            return "".join([f"{tok}0" for tok in MESSAGE_TOKEN_TYPES])
        context_str = "\n".join([_one_msg(), _one_msg()])
    input_ids = jnp.array(tokenizer.encode(context_str), dtype=jnp.int32)
    if input_ids.size == 0:
        out0, state = forward_jit(jnp.array([0], dtype=jnp.int32), state, params, 1)
        last_logits = out0[0]
    else:
        last_logits, state = process_long_seq(forward_jit, params, input_ids, state, len(input_ids))

    # Generate until we can parse at least n_gen_msgs
    generated_tokens = []
    key = jax.random.key(0)
    max_steps = 20000  # safety cap
    decoded_str = ""
    # lightweight counter for parsed rows without importing extra modules each time
    import re as _re, numpy as _np, pandas as _pd

    def _count_rows(s_in: str) -> int:
        s_in = s_in.replace(" ", "").replace("Ġ", "").replace("..", ".").replace("Ċ", "").replace("[PAD]", "").replace("[UNK]", "")
        cols_local = MESSAGE_TOKEN_TYPES
        patt = rf"({''.join([f'{t}|' for t in cols_local])})([^<]+)?"
        matches = _re.findall(patt, s_in)
        if not matches:
            return 0
        rows = []
        current_row = {}
        for col, value in matches:
            if col in current_row:
                rows.append(current_row)
                current_row = {}
            current_row[col] = value if value is not None else _np.nan
        if current_row:
            rows.append(current_row)
        # minimal frame just to drop empty rows
        df_tmp = _pd.DataFrame(rows)
        return len(df_tmp.dropna())

    for step in range(max_steps):
        # sample from last logits
        key, _key = jax.random.split(key)
        token = sample_token(_key, last_logits, temperature)
        generated_tokens.append(token)
        # forward one step to get next logits
        out_step, state = forward_jit(jnp.array([token], dtype=jnp.int32), state, params, 1)
        last_logits = out_step[0]
        # decode incrementally every 64 tokens to check stop condition
        if len(generated_tokens) % 64 == 0:
            decoded_str = tokenizer.decode(generated_tokens)
            if _count_rows(decoded_str) >= n_gen_msgs:
                break

    if not decoded_str:
        decoded_str = tokenizer.decode(generated_tokens)

    # Parse and convert to arrays
    m_seq_gen, raw_msgs = parse_generated_to_arrays(decoded_str)
    # Truncate to exactly n_gen_msgs
    raw_msgs = raw_msgs[:, :n_gen_msgs]
    msg_len = Message_Tokenizer.MSG_LEN
    m_seq_gen = m_seq_gen[:, : n_gen_msgs * msg_len]
    return m_seq_gen, raw_msgs


def insert_custom_end(
        m_seq_gen_doubled,
        b_seq_gen_doubled,
        msgs_decoded_doubled,
        l2_book_states_halved,
        encoder,
        mid_price,
        tick_size=100,
        EVENT_TYPE_i=4,
        DIRECTION_i=0,
        order_volume=75,
        use_relative_volume=False,
        order_volume_ratio=1.0,
    ):
    """Same insertion logic as CST script."""
    ORDER_ID_i = 77777777
    # Get actual batch_size from msgs_decoded_doubled (define once at the start)
    actual_batch_size = msgs_decoded_doubled.shape[0]
    
    # Ensure l2_book_states_halved has shape (batch, levels*4)
    if len(l2_book_states_halved.shape) == 2:
        base_l2 = l2_book_states_halved
    elif len(l2_book_states_halved.shape) == 3:
        base_l2 = l2_book_states_halved[:, -1]
    else:
        # Fallback: if shape is (levels*4,), expand to (batch_size, levels*4)
        if len(l2_book_states_halved.shape) == 1:
            base_l2 = jnp.broadcast_to(l2_book_states_halved[None, :], (actual_batch_size, l2_book_states_halved.shape[0]))
        else:
            raise ValueError(f"Unexpected l2_book_states_halved shape: {l2_book_states_halved.shape}")
    
    print(f"[DEBUG insert_custom_end] base_l2.shape before fix: {base_l2.shape}")
    print(f"[DEBUG insert_custom_end] msgs_decoded_doubled.shape: {msgs_decoded_doubled.shape}")
    print(f"[DEBUG insert_custom_end] actual_batch_size: {actual_batch_size}")
    
    # Ensure base_l2 has correct batch dimension matching msgs_decoded_doubled
    if base_l2.shape[0] != actual_batch_size:
        if base_l2.shape[0] == 1:
            base_l2 = jnp.broadcast_to(base_l2, (actual_batch_size, base_l2.shape[1]))
        else:
            raise ValueError(f"base_l2 batch dimension mismatch: {base_l2.shape[0]} vs {actual_batch_size}")
    
    print(f"[DEBUG insert_custom_end] base_l2.shape after fix: {base_l2.shape}")
    
    sim_init, sim_states_init = inference.get_sims_vmap(
        base_l2,
        msgs_decoded_doubled[:, -1:] if msgs_decoded_doubled.shape[1] > 0 else msgs_decoded_doubled
    )
    print(f"[DEBUG insert_custom_end] sim_states_init type: {type(sim_states_init)}")
    # Check if sim_states_init is a structured object (LobState)
    if hasattr(sim_states_init, 'asks'):
        print(f"[DEBUG insert_custom_end] sim_states_init.asks.shape: {sim_states_init.asks.shape}")
        print(f"[DEBUG insert_custom_end] sim_states_init.bids.shape: {sim_states_init.bids.shape}")

    # Define batch_size early for use in mid_price processing
    batch_size = actual_batch_size

    if DIRECTION_i == 0:
        PRICE_i = jax.vmap(sim_init.get_best_ask)(sim_states_init)
    else:
        PRICE_i = jax.vmap(sim_init.get_best_bid)(sim_states_init)

    PRICE_i = jnp.expand_dims(PRICE_i, axis=-1)
    
    # Ensure mid_price has correct batch_size and shape
    print(f"[DEBUG insert_custom_end] mid_price.shape before fix: {mid_price.shape}, batch_size: {batch_size}")
    if len(mid_price.shape) == 1:
        if mid_price.shape[0] == 1 and batch_size > 1:
            # Broadcast from (1,) to (batch_size,)
            mid_price = jnp.broadcast_to(mid_price, (batch_size,))
        mid_price = jnp.expand_dims(mid_price, axis=-1)
    elif len(mid_price.shape) == 2:
        if mid_price.shape[0] == 1 and batch_size > 1:
            # Broadcast from (1, 1) to (batch_size, 1)
            mid_price = jnp.broadcast_to(mid_price, (batch_size, mid_price.shape[1]))
        elif mid_price.shape[1] == 1:
            # Already has (batch, 1) shape, keep as is
            pass
        else:
            # Reshape if needed
            mid_price = mid_price[:, None] if mid_price.shape[0] == batch_size else mid_price
    else:
        # Already has correct shape or needs expansion
        if len(mid_price.shape) == 0:
            mid_price = jnp.broadcast_to(jnp.array([mid_price]), (batch_size, 1))
    
    print(f"[DEBUG insert_custom_end] mid_price.shape after fix: {mid_price.shape}")

    # Get time info - handle case when there are no messages
    if msgs_decoded_doubled.shape[1] > 0:
        TIMEs_i  = msgs_decoded_doubled[:, -1:, 8].astype(jnp.int32)
        TIMEns_i = msgs_decoded_doubled[:, -1:, 9].astype(jnp.int32)
    else:
        TIMEs_i = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        TIMEns_i = jnp.zeros((batch_size, 1), dtype=jnp.int32)

    best_bid_ask = jax.vmap(sim_init.get_best_bid_and_ask_inclQuants)(sim_states_init)

    avail = jnp.where(
        DIRECTION_i == 0,
        best_bid_ask[1][:, 1],
        best_bid_ask[0][:, 1],
    ).astype(jnp.int32)

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

    print(f"[DEBUG insert_custom_end] batched_sim_msg.shape: {batched_sim_msg.shape}")
    new_sim_state = jax.vmap(sim_init.process_order_array)(sim_states_init, batched_sim_msg)
    print(f"[DEBUG insert_custom_end] new_sim_state type: {type(new_sim_state)}")
    if hasattr(new_sim_state, 'asks'):
        print(f"[DEBUG insert_custom_end] new_sim_state.asks.shape: {new_sim_state.asks.shape}")
        print(f"[DEBUG insert_custom_end] new_sim_state.bids.shape: {new_sim_state.bids.shape}")
    p_mid_new = inference.batched_get_safe_mid_price(sim_init, new_sim_state, tick_size)
    print(f"[DEBUG insert_custom_end] p_mid_new.shape: {p_mid_new.shape}")
    p_mid_new = p_mid_new[:, None]
    p_change = ((p_mid_new - mid_price) // tick_size).astype(jnp.int32)

    current_levels = l2_book_states_halved.shape[-1] // 4
    # Debug: check shapes
    print(f"[DEBUG insert_custom_end] batch_size: {batch_size}")
    print(f"[DEBUG insert_custom_end] current_levels: {current_levels}")
    print(f"[DEBUG insert_custom_end] l2_book_states_halved.shape: {l2_book_states_halved.shape}")
    print(f"[DEBUG insert_custom_end] p_change.shape: {p_change.shape}")
    
    # Get L2 state - ensure it's properly batched
    book_l2 = jax.vmap(sim_init.get_L2_state, in_axes=(0, None))(new_sim_state, current_levels)
    print(f"[DEBUG insert_custom_end] book_l2.shape before fix: {book_l2.shape}")
    print(f"[DEBUG insert_custom_end] batch_size: {batch_size}, p_change.shape: {p_change.shape}")
    
    # Ensure book_l2 has correct shape (batch, levels*4) matching p_change
    if book_l2.shape[0] != batch_size:
        if book_l2.shape[0] == 1:
            # Broadcast from (1, levels*4) to (batch_size, levels*4)
            book_l2 = jnp.broadcast_to(book_l2, (batch_size, book_l2.shape[1]))
        elif len(book_l2.shape) == 1:
            # If somehow we got (levels*4,), reshape to (batch_size, levels*4)
            book_l2 = jnp.broadcast_to(book_l2[None, :], (batch_size, book_l2.shape[0]))
        else:
            raise ValueError(f"book_l2 has unexpected shape: {book_l2.shape}, expected batch_size={batch_size}")
    
    print(f"[DEBUG insert_custom_end] book_l2.shape after fix: {book_l2.shape}")
    
    new_l2_book_states_halved = book_l2  # keep as (batch, levels)
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
        ds,
        rng: jax.dtypes.prng_key,
        seq_len: int,
        n_msgs: int,
        n_gen_msgs: int,
        tokenizer,
        RWKV,
        params,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        stock_symbol: str,
        n_vol_series: int = 20,
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
            if ds is None:
                # fallback indices: dummy indices for batches
                one_batch = list(range(batch_size))
                sample_i = [one_batch for _ in range(n_samples // batch_size)]
            else:
                sample_i = jax.random.choice(
                    rng_,
                    jnp.arange(len(ds), dtype=jnp.int32),
                    shape=(n_samples // batch_size, batch_size),
                    replace=False
                ).tolist()

    rng, rng_ = jax.random.split(rng)

    save_folder = Path(save_folder)
    save_folder.joinpath('msgs_decoded_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('b_seq_gen_doubled').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('l2_book_states').mkdir(exist_ok=True, parents=True)
    save_folder.joinpath('mid_price').mkdir(exist_ok=True, parents=True)

    base_save_folder = save_folder

    transform_L2_state_batch = jax.jit(jax.vmap(transform_L2_state, in_axes=(0, None, None)), static_argnums=(1, 2))

    num_iterations = num_insertions + num_coolings

    print('sample_i:', sample_i)

    for batch_i in tqdm(sample_i):
        print('BATCH', batch_i)
        proc_msgs_numb = -n_msgs
        midprices = []  # Initialize midprices list once per batch, accumulate across iterations

        for iteration in range(1, num_iterations+1):
            print('\nITERATION ', iteration)
            # Initialize prev_iter_final_l2 for tracking on subsequent iterations
            prev_iter_final_l2 = None

            if iteration == 1:
                if ds is not None:
                    m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
                    m_seq = jnp.array(m_seq)
                    b_seq_pv = jnp.array(b_seq_pv)
                    msg_seq_raw = jnp.array(msg_seq_raw)
                    book_l2_init = jnp.array(book_l2_init)
                    b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)

                    m_seq_inp = m_seq[:, : seq_len]
                    b_seq_inp = b_seq[: , : n_msgs]
                    # For context generation, use only n_msgs (50)
                    m_seq_raw_inp_context = msg_seq_raw[:, : n_msgs]
                    # For tracking midprices, use ALL historical messages (500)
                    # msg_seq_raw contains all historical messages from dataset
                    m_seq_raw_inp_full = msg_seq_raw  # Use all historical messages

                    sim_init, sim_states_init = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp_context)

                    context_str = encode_context_from_msgs(m_seq_raw_inp_context, n_msgs)
                    
                    # Track midprices from ALL initial messages (historical) - use full sequence
                    print(f"[DEBUG] Tracking midprices from full historical sequence: m_seq_raw_inp_full.shape = {m_seq_raw_inp_full.shape}")
                    midprices_batch = track_midprices_during_messages(
                        m_seq_raw_inp_full,
                        book_l2_init,
                        tick_size,
                        midprice_step_size,
                    )
                    # Convert to list and add to midprices (accumulate across iterations)
                    # midprices_batch has shape (num_steps, batch), convert to list of arrays
                    print(f"[DEBUG] Historical midprices_batch.shape = {midprices_batch.shape}")
                    print(f"[DEBUG] Historical m_seq_raw_inp_full.shape = {m_seq_raw_inp_full.shape}")
                    print(f"[DEBUG] Historical m_seq_raw_inp_context.shape = {m_seq_raw_inp_context.shape} (for context only)")
                    midprices.extend([midprices_batch[i] for i in range(midprices_batch.shape[0])])
                    proc_msgs_numb += m_seq_raw_inp_full.shape[1]  # Use full sequence length
                    print(f"[DEBUG] After historical messages: midprices length = {len(midprices)}")
                    print(f"[DEBUG] Expected historical steps: {m_seq_raw_inp_full.shape[1] // midprice_step_size}")
                    
                    # Get final L2 state after ALL historical messages for tracking generated messages
                    # This will be used as init_l2_for_tracking after generation
                    if m_seq_raw_inp_full.shape[1] > 0:
                        # Get final state after processing all historical messages (500)
                        sim_init_hist, sim_states_hist = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp_full)
                        final_l2_after_hist = jax.vmap(sim_init_hist.get_L2_state, in_axes=(0, None))(
                            sim_states_hist, book_l2_init.shape[1] // 4
                        )
                        final_l2_after_hist = final_l2_after_hist[:, : book_l2_init.shape[1]]
                    else:
                        final_l2_after_hist = book_l2_init
                else:
                    # fallback: empty context and zero L2 state
                    m_seq_inp = jnp.zeros((batch_size, 0), dtype=jnp.int32)
                    b_seq_inp = jnp.zeros((batch_size, 0, 1 + n_vol_series * 4), dtype=jnp.int32)
                    m_seq_raw_inp = jnp.zeros((batch_size, 0, 14), dtype=jnp.int32)
                    book_l2_init = jnp.zeros((batch_size, n_vol_series * 4), dtype=jnp.int32)
                    sim_init, sim_states_init = inference.get_sims_vmap(book_l2_init, m_seq_raw_inp)
                    context_str = ""
                    # Initialize final_l2_after_hist for fallback case
                    final_l2_after_hist = book_l2_init
                    # No historical midprices in fallback case (midprices already initialized as [] above)
            else:
                # Use previous iteration's generated data
                m_seq_inp = m_seq_gen_doubled
                b_seq_inp = b_seq_gen_doubled
                m_seq_raw_inp = msgs_decoded_doubled

                # Rebuild context from last n_msgs messages
                take_msgs = min(n_msgs, m_seq_raw_inp.shape[1])
                context_str = encode_context_from_msgs(m_seq_raw_inp[:, -take_msgs:, :], take_msgs)

                # Reconstruct simulator state from last L2
                last_l2 = l2_book_states_halved if len(l2_book_states_halved.shape) == 2 else l2_book_states_halved[:, -1]
                sim_init, sim_states_init = inference.get_sims_vmap(
                    last_l2,
                    m_seq_raw_inp[:, -1:]
                )
                # Store last_l2 for use as init_l2_for_tracking after generation
                prev_iter_final_l2 = last_l2

            # Generate using RWKV
            print("Generating with RWKV model...")
            m_seq_gen, msgs_decoded = generate_rwkv_batch(
                tokenizer,
                RWKV,
                params,
                context_str,
                n_gen_msgs,
            )

            # Reconstruct L2 and book sequences from generated messages
            l2_book_states, b_seq_gen = reconstruct_books_from_msgs(msgs_decoded, tick_size, n_vol_series)

            # Expand to batch_size if needed (generate_rwkv_batch returns batch_size=1)
            if m_seq_gen.shape[0] == 1 and batch_size > 1:
                # Broadcast to batch_size
                m_seq_gen = jnp.broadcast_to(m_seq_gen, (batch_size, m_seq_gen.shape[1]))
            if b_seq_gen.shape[0] == 1 and batch_size > 1:
                b_seq_gen = jnp.broadcast_to(b_seq_gen, (batch_size, b_seq_gen.shape[1], b_seq_gen.shape[2]))
            if msgs_decoded.shape[0] == 1 and batch_size > 1:
                msgs_decoded = jnp.broadcast_to(msgs_decoded, (batch_size, msgs_decoded.shape[1], msgs_decoded.shape[2]))
            if l2_book_states.shape[0] == 1 and batch_size > 1:
                if len(l2_book_states.shape) == 3:
                    l2_book_states = jnp.broadcast_to(l2_book_states, (batch_size, l2_book_states.shape[1], l2_book_states.shape[2]))
                elif len(l2_book_states.shape) == 2:
                    l2_book_states = jnp.broadcast_to(l2_book_states, (batch_size, l2_book_states.shape[1]))

            m_seq_gen_doubled = m_seq_gen
            b_seq_gen_doubled = b_seq_gen
            msgs_decoded_doubled = msgs_decoded
            gen_levels = (l2_book_states.shape[-1] // 4) if len(l2_book_states.shape) >= 2 else n_vol_series
            if len(l2_book_states.shape) == 3 and l2_book_states.shape[1] > 0:
                l2_book_states_halved = l2_book_states[:, -1, : gen_levels * 4]
            elif len(l2_book_states.shape) == 3 and l2_book_states.shape[1] == 0:
                l2_book_states_halved = jnp.zeros((l2_book_states.shape[0], gen_levels * 4), dtype=jnp.int32)
            else:
                l2_book_states_halved = l2_book_states

            print(f'\n\nsuccessfully generated iteration no. {iteration}')

            # Track midprices on the fly via simulator
            # On first iteration, use final state after historical messages
            # On subsequent iterations, use last state from previous iteration (from l2_book_states_halved)
            if iteration == 1:
                if ds is not None:
                    # Use final state after historical messages (already computed above)
                    init_l2_for_tracking = final_l2_after_hist
                else:
                    # Fallback: use first state from generated books
                    if len(l2_book_states.shape) == 3 and l2_book_states.shape[1] > 0:
                        init_l2_for_tracking = l2_book_states[:, 0, : gen_levels * 4]
                    elif len(l2_book_states.shape) == 3 and l2_book_states.shape[1] == 0:
                        init_l2_for_tracking = jnp.zeros((l2_book_states.shape[0], gen_levels * 4), dtype=jnp.int32)
                    else:
                        init_l2_for_tracking = l2_book_states[0]
            else:
                # On subsequent iterations, use last state from previous iteration
                # This is stored in prev_iter_final_l2 (computed above before generation)
                if prev_iter_final_l2 is not None:
                    init_l2_for_tracking = prev_iter_final_l2
                else:
                    # Fallback: use first state from current generated books
                    if len(l2_book_states.shape) == 3 and l2_book_states.shape[1] > 0:
                        init_l2_for_tracking = l2_book_states[:, 0, : gen_levels * 4]
                    elif len(l2_book_states.shape) == 3 and l2_book_states.shape[1] == 0:
                        init_l2_for_tracking = jnp.zeros((l2_book_states.shape[0], gen_levels * 4), dtype=jnp.int32)
                    else:
                        init_l2_for_tracking = l2_book_states[0]
            
            # Track midprices for all generated messages using step_size
            print(f"[DEBUG] Before tracking: msgs_decoded_doubled.shape = {msgs_decoded_doubled.shape}, init_l2_for_tracking.shape = {init_l2_for_tracking.shape}")
            if msgs_decoded_doubled.shape[1] > 0:
                midprices_batch = track_midprices_during_messages(
                    msgs_decoded_doubled,
                    init_l2_for_tracking,
                    tick_size,
                    midprice_step_size,
                )
                print(f"[DEBUG] midprices_batch.shape = {midprices_batch.shape}")
                # Convert to list and extend midprices list (accumulate across iterations)
                # midprices_batch has shape (num_steps, batch), convert to list of arrays
                midprices.extend([midprices_batch[i] for i in range(midprices_batch.shape[0])])
                proc_msgs_numb += msgs_decoded_doubled.shape[1]
            else:
                print(f"[DEBUG] No messages to track midprices for (msgs_decoded_doubled.shape[1] = 0)")
            print(f"[DEBUG] After iteration {iteration} generation: midprices length = {len(midprices)}")

            if iteration <= num_insertions:
                print(">> INSERTING CUSTOM ORDER")
                # Use last midprice if available, otherwise use zeros
                last_midprice = midprices[-1] if len(midprices) > 0 else jnp.zeros((batch_size,), dtype=jnp.int32)
                m_seq_gen_doubled, b_seq_gen_doubled, msgs_decoded_doubled, l2_book_states_halved, p_mid_new = insert_custom_end(
                    m_seq_gen_doubled,
                    b_seq_gen_doubled,
                    msgs_decoded_doubled,
                    l2_book_states_halved,
                    encoder,
                    last_midprice,
                    tick_size,
                    EVENT_TYPE_i,
                    DIRECTION_i,
                    order_volume,
                    use_relative_volume,
                    order_volume_ratio,
                )
                # Add the new midprice after insertion
                midprices.append(jnp.squeeze(p_mid_new, axis=-1) if len(p_mid_new.shape) > 1 else p_mid_new)
                proc_msgs_numb += 1
                print(f"[DEBUG] After insertion: midprices length = {len(midprices)}")

            # Save artifacts
            m_seq_np = np.array(jax.device_get(m_seq_gen_doubled))
            b_seq_np = np.array(jax.device_get(b_seq_gen_doubled))
            msgs_decoded_np = np.array(jax.device_get(msgs_decoded_doubled))
            l2_book_states_np = np.array(jax.device_get(l2_book_states_halved))
            # mid-price: convert list of arrays to numpy array
            # midprices is a list of arrays, each with shape (batch,)
            # np.array(midprices) will create shape (len(midprices), batch)
            try:
                if len(midprices) > 0:
                    # Convert each JAX array to numpy and stack
                    print(f"[DEBUG] Before saving: len(midprices) = {len(midprices)}")
                    midprices_np_list = [np.array(jax.device_get(mp)) for mp in midprices]
                    print(f"[DEBUG] After conversion: len(midprices_np_list) = {len(midprices_np_list)}")
                    mid_price_np = np.array(midprices_np_list)  # Shape: (num_steps, batch)
                    print(f"[DEBUG] Final mid_price_np.shape = {mid_price_np.shape}")
                else:
                    mid_price_np = np.zeros((0, batch_size), dtype=np.int32)
            except Exception as e:
                print(f"Error saving midprices: {e}")
                import traceback
                traceback.print_exc()
                mid_price_np = np.zeros((0, batch_size), dtype=np.int32)

            # Print shapes for verification
            print(f"\n[SAVE] Saving artifacts for batch {batch_i}, iteration {iteration}:")
            print(f"  msgs_decoded_np.shape: {msgs_decoded_np.shape}")
            print(f"  b_seq_np.shape: {b_seq_np.shape}")
            print(f"  l2_book_states_np.shape: {l2_book_states_np.shape}")
            print(f"  mid_price_np.shape: {mid_price_np.shape}")
            print(f"  mid_price_np.ndim: {mid_price_np.ndim}")
            print(f"  Total midprices accumulated so far: {len(midprices)}")
            if msgs_decoded_np.size > 0:
                print(f"  msgs_decoded_np sample (first msg, first batch): {msgs_decoded_np[0, 0, :]}")
            if b_seq_np.size > 0:
                print(f"  b_seq_np sample (first book, first batch): {b_seq_np[0, 0, :10]}...")
            if mid_price_np.size > 0:
                print(f"  mid_price_np sample (first few): {mid_price_np[:min(3, mid_price_np.shape[0]), 0]}")

            np.save(os.path.join(base_save_folder, 'msgs_decoded_doubled', f'msgs_decoded_doubled_batch_{batch_i}_iter_{iteration}.npy'), msgs_decoded_np)
            np.save(os.path.join(base_save_folder, 'b_seq_gen_doubled', f'b_seq_gen_doubled_batch_{batch_i}_iter_{iteration}.npy'), b_seq_np)
            np.save(os.path.join(base_save_folder, 'l2_book_states', f'l2_book_states_batch_{batch_i}_iter_{iteration}.npy'), l2_book_states_np)
            np.save(os.path.join(base_save_folder, 'mid_price', f'mid_price_batch_{batch_i}_iter_{iteration}.npy'), mid_price_np)


def create_next_experiment_folder(save_folder: str) -> Path:
    base = Path(save_folder)
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
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
    p = argparse.ArgumentParser(description="Run LOB inference scenario with RWKV model")
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

    # Initialize WandB
    wandb.init(
        project="Aggressive_Scenario_RWKV",
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

    use_sample_file  = cfg["use_sample_file"]
    sample_file_path = cfg["sample_file_path"]
    start_batch      = cfg["start_batch"]
    end_batch        = cfg["end_batch"]

    num_devices = jax.local_device_count()
    if batch_size % num_devices != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by num_devices ({num_devices})")

    # Experiment folder
    exp_folder = create_next_experiment_folder(save_folder)
    print("Experiment dir:", exp_folder)
    wandb.run.summary["experiment_dir"] = str(exp_folder)

    with open(exp_folder / "used_config.yaml", "w") as f_out:
        yaml.dump(cfg, f_out)

    # Setup logging to experiment folder
    log_file_path = exp_folder / "job.log"
    print(f"Redirecting all output to: {log_file_path}")
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"Experiment started at: {datetime.now()}")
    print(f"Experiment folder: {exp_folder}")
    print(f"Configuration: {cfg}")
    print("=" * 80)

    # Load RWKV checkpoint and tokenizer
    ckpt_dir = "/app/checkpoints/goog2022_rwkv_6g0.1B/final"
    class _IdentityTokenizer:
        def encode(self, s: str):
            return []
        def decode(self, tokens):
            return ""
    tokenizer = _IdentityTokenizer()

    # Initialize RWKV-6 Scan variant and load parameters from pickle
    RWKV = rwkv6_base.ScanRWKV
    import pickle as _pkl
    with open(os.path.join(ckpt_dir, "params.model"), "rb") as _f:
        params = _pkl.load(_f)

    # prepare RNG
    rng = jax.random.PRNGKey(rng_seed)

    # data directory
    data_path = Path(data_dir) / stock
    data_path.mkdir(parents=True, exist_ok=True)

    # get dataset if available, otherwise None
    ds = None
    try:
        if any(True for _ in data_path.iterdir()):
            ds = inference.get_dataset(data_path, n_messages, n_gen_msgs)
            wandb.log({"dataset_size": len(list(data_path.iterdir()))})
    except Exception:
        ds = None

    # run generation with RWKV
    run_generation_scenario(
        n_samples,
        batch_size,
        ds,
        rng,
        n_messages * Message_Tokenizer.MSG_LEN,
        n_messages,
        n_gen_msgs,
        tokenizer,
        RWKV,
        params,
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
    )

    wandb.log({"finished": True})
    wandb.save(str(exp_folder / "*"))

    print(f"Experiment completed at: {datetime.now()}")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f"Logs saved to: {log_file_path}")


if __name__ == "__main__":
    main()


