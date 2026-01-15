"""
Autoregressive inference module for LOB generation with hidden state support.

This module provides inference functions that maintain and pass hidden state
through the generation process, enabling true autoregressive generation
where the hidden state accumulates across messages and iterations.

Key differences from inference_no_errcorr.py:
1. _generate_token takes and returns hidden state
2. _generate_msg propagates hidden state through token generation
3. generate returns final hidden state
4. generate_batched is vmapped to handle batched hidden states
"""
from jax import config
config.update("jax_disable_jit", False)

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
import math
import time

# Import autoreg validation helpers
from lob import validation_helpers_autoreg as valh
# Also import some functions from original validation helpers
from lob import validation_helpers as valh_orig
from lob import evaluation as eval
import preproc as preproc
from lob import encoding
from lob.encoding import Message_Tokenizer, Vocab
from lob.lobster_dataloader import LOBSTER_Dataset
import chex

# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(
    os.path.split(os.path.abspath(__file__))[0])
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import Configuration as JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

# Field indices for decoded messages (same as in inference_no_errcorr.py)
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

# REF_LEN: Number of tokens for reference fields
REF_LEN = Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN

l2_state_n = 10

# ENCODED TOKEN INDICES
# time tokens aren't generated but calculated using delta_t
# hence, skip generation from TIME_START_I (inclusive) to TIME_END_I (exclusive)
TIME_START_I, _ = valh.get_idx_from_field('time_s')
_, TIME_END_I = valh.get_idx_from_field('time_ns')


# ============================================================================
# Helper functions (imported/adapted from original module)
# ============================================================================

@jax.jit
def construct_sim_msg(
        event_type: int,
        side: int,
        quantity: int,
        price: int,
        order_id: int,
        time_s: int,
        time_ns: int,
    ) -> jax.Array:
    """Construct a message array for the JAX LOB simulator."""
    return jnp.array([
        event_type,
        (side * 2) - 1,  # convert {0, 1} to {-1, 1}
        quantity,
        price,
        0,  # trader_id
        order_id,
        time_s,
        time_ns,
    ], dtype=jnp.int32)


@jax.jit
def construct_dummy_sim_msg() -> jax.Array:
    """Construct a dummy/no-op message for the simulator."""
    return jnp.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


@jax.jit
def add_times(
        a_s: jax.Array,
        a_ns: jax.Array,
        b_s: jax.Array,
        b_ns: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
    """Add two timestamps given as seconds and nanoseconds."""
    a_ns = b_ns + a_ns
    extra_s = a_ns // 1000000000
    a_ns = a_ns % 1000000000
    a_s = a_s + b_s + extra_s
    return a_s, a_ns


def _get_safe_mid_price(
        sim: OrderBook,
        sim_state: LobState,
        tick_size: int,
    ) -> int:
    """Get mid price from simulator, handling edge cases."""
    ask = sim.get_best_ask(sim_state)
    bid = sim.get_best_bid(sim_state)

    case_i = (ask <= 0) * 1 + (bid <= 0) * 2

    p_mid = jax.lax.switch(
        case_i,
        (
            lambda ask, bid: (ask + bid) // 2,
            lambda ask, bid: bid + tick_size,
            lambda ask, bid: ask - tick_size,
            lambda ask, bid: 0,
        ),
        ask, bid
    )
    p_mid = (p_mid // tick_size) * tick_size
    return p_mid


@partial(jax.jit, static_argnums=(0,))
def _get_new_mid_price(
        sim: OrderBook,
        sim_state: LobState,
        p_mid_old: jax.Array,
        tick_size: int,
    ) -> jax.Array:
    """Get new mid price, falling back to old if invalid."""
    ask = sim.get_best_ask(sim_state)
    bid = sim.get_best_bid(sim_state)
    mid = ((((ask + bid) // 2) // tick_size) * tick_size)
    return jax.lax.cond(
        (ask <= 0) | (bid <= 0),
        lambda new, old: old,
        lambda new, old: new,
        mid, p_mid_old
    )


def _add_time_tokens(
        tok_seq_A: jax.Array,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        time_init_s: int,
        time_init_ns: int,
        delta_t_s_start_i: int,
        delta_t_s_end_i: int,
        delta_t_ns_start_i: int,
        delta_t_ns_end_i: int,
    ):
    """Add time tokens to a generated token sequence."""
    delta_t_s_toks = tok_seq_A[delta_t_s_start_i: delta_t_s_end_i]
    delta_t_ns_toks = tok_seq_A[delta_t_ns_start_i: delta_t_ns_end_i]
    delta_t_s = encoding.decode(delta_t_s_toks, *encoder['time'])
    delta_t_s = encoding.combine_field(delta_t_s, 3)
    delta_t_ns = encoding.decode(delta_t_ns_toks, *encoder['time'])
    delta_t_ns = encoding.combine_field(delta_t_ns, 3)

    time_s_ret, time_ns_ret = add_times(time_init_s, time_init_ns, delta_t_s, delta_t_ns)

    time_s = encoding.split_field(time_s_ret, 2, 3)
    time_s_toks = encoding.encode(time_s, *encoder['time'])
    time_ns = encoding.split_field(time_ns_ret, 3, 3)
    time_ns_toks = encoding.encode(time_ns, *encoder['time'])

    time_tokens = jnp.hstack([time_s_toks, time_ns_toks])
    return time_tokens, time_s_ret, time_ns_ret


def get_sim_msg(
        pred_msg_enc: jax.Array,
        sim: OrderBook,
        sim_state: LobState,
        mid_price: int,
        new_order_id: int,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        token_mode: int = 24,
    ) -> Tuple[jax.Array, jax.Array]:
    """Convert encoded predicted message to simulator message."""
    msg_decoded = encoding.decode_msg(pred_msg_enc, encoder, token_mode=token_mode)
    new_part = msg_decoded[: Message_Tokenizer.N_NEW_FIELDS]

    event_type = msg_decoded[EVENT_TYPE_i]
    quantity = msg_decoded[SIZE_i]
    side = msg_decoded[DIRECTION_i]
    rel_price = msg_decoded[PRICE_i]
    time_s = msg_decoded[TIMEs_i]
    time_ns = msg_decoded[TIMEns_i]
    time_s_ref = msg_decoded[TIMEs_REF_i]
    time_ns_ref = msg_decoded[TIMEns_REF_i]

    p_abs = mid_price + rel_price * tick_size

    orig_order = sim.get_order_at_time(sim_state, side, time_s_ref, time_ns_ref)
    order_id_ref = orig_order[2]

    order_id = jax.lax.cond(
        (event_type == 2) | (event_type == 3),
        lambda new_id, ref_id: ref_id,
        lambda new_id, ref_id: new_id,
        new_order_id, order_id_ref
    )

    sim_msg = construct_sim_msg(
        event_type,
        side,
        quantity,
        p_abs,
        order_id,
        time_s,
        time_ns,
    )

    msg_decoded = msg_decoded.at[PRICE_ABS_i].set(p_abs) \
                             .at[ORDER_ID_i].set(order_id)

    return jax.lax.cond(
        jnp.isnan(new_part).any(),
        lambda sim_msg, msg_decoded: (construct_dummy_sim_msg(), msg_decoded),
        lambda sim_msg, msg_decoded: (sim_msg, msg_decoded),
        sim_msg, msg_decoded
    )


# ============================================================================
# Autoregressive generation functions with hidden state
# ============================================================================

def _generate_token(
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        valid_mask_array: jax.Array,
        sample_top_n: int,
        m_tok: jax.Array,
        b_tok: jax.Array,
        hidden: Tuple,
        token_index: int,
        rng,
    ):
    """
    Generate a single token using the model with hidden state.

    Args:
        train_state: Model parameters
        model: Flax model
        batchnorm: Whether to use batchnorm
        valid_mask_array: Syntax validation matrix
        sample_top_n: Number of top tokens to sample from
        m_tok: Current message token(s)
        b_tok: Current book state
        hidden: Current hidden state tuple
        token_index: Index of token being generated in message
        rng: Random key

    Returns:
        Tuple of (predicted_token, updated_hidden, next_index, new_rng)
    """
    # Get valid tokens for current position
    valid_mask = valh.get_valid_mask(valid_mask_array, token_index)

    # Apply model with hidden state
    hidden, logits = valh.apply_model(
        hidden,
        m_tok,
        b_tok,
        train_state,
        model,
        batchnorm,
        False  # shift_start
    )

    logits = logits[0]

    # Filter invalid tokens
    if valid_mask is not None:
        logits = valh.filter_valid_pred(logits, valid_mask)

    # Sample token
    rng, rng_ = jax.random.split(rng)
    m_tok = valh.fill_predicted_tok(logits, sample_top_n, jnp.array([rng_]))

    return m_tok, hidden, token_index + 1, rng


def _make_generate_token_scannable(
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        valid_mask_array: jax.Array,
        sample_top_n: int,
    ):
    """Create a scannable version of _generate_token."""
    __generate_token = jax.jit(functools.partial(
        _generate_token, train_state, model, batchnorm, valid_mask_array, sample_top_n
    ))

    def _generate_token_scannable(carry, xs):
        m_tok, hidden, mask_i, rng = __generate_token(*carry)
        return (m_tok, carry[1], hidden, mask_i, rng), m_tok

    return _generate_token_scannable


def _generate_msg(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        valid_mask_array: jax.Array,
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,
        token_mode: int,
        m_init: jax.Array,
        b_init: jax.Array,
        n_msg_todo: int,
        p_mid: jax.Array,
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        hidden: Tuple,
        time_i: jax.Array,
        b_seq_real: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, LobState, jax.Array, jax.Array, jax.Array, jax.Array, int, Tuple, jax.Array]:
    """
    Generate a single message with hidden state propagation.

    Returns:
        Tuple of (msg_decoded, sim_state, m_final, tok_seq_gen, b_final,
                  book_l2, p_mid_new, n_msg_todo, hidden, time_f)
    """
    rng, rng_ = jax.random.split(rng)

    with jax.ensure_compile_time_eval():
        l = Message_Tokenizer.MSG_LEN
        time_s_start_i, time_s_end_i = valh.get_idx_from_field('time_s')
        time_ns_start_i, time_ns_end_i = valh.get_idx_from_field('time_ns')
        delta_t_s_start_i, delta_t_s_end_i = valh.get_idx_from_field('delta_t_s')
        delta_t_ns_start_i, delta_t_ns_end_i = valh.get_idx_from_field('delta_t_ns')

    time_init_s = time_i[0]
    time_init_ns = time_i[1]

    generate_token_scannable = _make_generate_token_scannable(
        train_state, model, batchnorm, valid_mask_array, sample_top_n
    )

    if debug_book:
        b_init = jnp.expand_dims(b_seq_real, 0)

    # Generate tokens until time field
    token_idx = 0
    gen_token_carry = (m_init, b_init, hidden, token_idx, rng_)
    (m_inter, b_inter, hidden, token_idx, rng_), tok_seq_A = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=time_s_start_i
    )
    tok_seq_A = jnp.squeeze(tok_seq_A)

    # Fill time tokens
    tok_seq_T, time_s, time_ns = _add_time_tokens(
        tok_seq_A,
        encoder,
        time_init_s,
        time_init_ns,
        delta_t_s_start_i,
        delta_t_s_end_i,
        delta_t_ns_start_i,
        delta_t_ns_end_i,
    )
    time_f = jnp.array([time_s, time_ns])

    # Roll time tokens through hidden state
    tok_seq_roll_thru_hidden = jnp.concatenate([tok_seq_A[-1:], tok_seq_T[:-1]])
    hidden, _ = valh.apply_model(
        hidden,
        tok_seq_roll_thru_hidden,
        b_init,
        train_state,
        model,
        batchnorm,
        False
    )

    # Update index and finish message generation
    token_idx = time_ns_end_i
    gen_token_carry = (tok_seq_T[-1:], b_init, hidden, token_idx, rng_)

    (m_final, b_final, hidden, token_idx, rng_), tok_seq_B = jax.lax.scan(
        generate_token_scannable,
        gen_token_carry,
        xs=None,
        length=l - time_ns_end_i
    )
    tok_seq_B = jnp.squeeze(tok_seq_B)

    # Complete generated message
    tok_seq_gen = jnp.concatenate([tok_seq_A, tok_seq_T, tok_seq_B])
    order_id = n_msg_todo

    sim_msg, msg_decoded = get_sim_msg(
        tok_seq_gen,
        sim,
        sim_state,
        mid_price=p_mid,
        new_order_id=order_id,
        tick_size=tick_size,
        encoder=encoder,
        token_mode=token_mode,
    )

    # Process message in simulator
    sim_state = sim.process_order_array(sim_state, sim_msg)

    # Get new mid price
    p_mid_new = _get_new_mid_price(sim, sim_state, p_mid, tick_size)
    p_change = ((p_mid_new - p_mid) // tick_size)

    # Get new book state
    book_l2 = sim.get_L2_state(sim_state, l2_state_n)

    # Use original transform_L2_state (without time in input) to create the 501-element book
    # Then add normalized time to match model's expected input shape (503 = 3 + 500)
    new_book_raw = jnp.concatenate([jnp.array([p_change]), book_l2[0:40]]).reshape(1, -1)
    b_base = preproc.transform_L2_state(new_book_raw, 500, 100)  # Shape: (1, 501)
    # Normalize time: time_s to fraction of day (34200s = 9.5h market open, 23400s = 6.5h trading)
    # time_ns to fraction of second
    time_normalized = jnp.array([[(time_f[0] - 34200) / 23400, time_f[1] / 1e9]])  # Shape: (1, 2)
    # Insert time after delta_p_mid: [delta_p_mid, time_s_norm, time_ns_norm, ...volumes...]
    b_final = jnp.concatenate([b_base[:, :1], time_normalized, b_base[:, 1:]], axis=1)  # Shape (1, 503)

    n_msg_todo -= 1

    return msg_decoded, sim_state, m_final, tok_seq_gen, b_final, book_l2, p_mid_new, n_msg_todo, hidden, time_f


def _make_generate_msg_scannable(
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        valid_mask_array: jax.Array,
        sample_top_n: int,
        tick_size: int,
        debug_book: bool,
        token_mode: int,
    ):
    """Create a scannable version of _generate_msg."""
    # NOTE: Don't specify device= here to allow pmap to place computation on the correct device
    # When using pmap, the device is determined by the outer pmap, not this inner jit
    __generate_msg = jax.jit(functools.partial(
        _generate_msg, sim, train_state, model, batchnorm,
        encoder, valid_mask_array, sample_top_n, tick_size, debug_book, token_mode
    ))

    def _generate_msg_scannable(gen_state, input):
        b_seq_real = input
        m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng, hidden, time = gen_state
        rng, rng_ = jax.random.split(rng)

        msg_decoded, sim_state, m_seq, msg_token, b_seq, book_l2, p_mid, n_msg_todo, hidden, time = __generate_msg(
            m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng_, hidden, time, b_seq_real
        )
        return (m_seq, b_seq, n_msg_todo, p_mid, sim_state, rng, hidden, time), (msg_decoded, book_l2, msg_token)

    return _generate_msg_scannable


@partial(jax.jit, static_argnums=(0, 2, 3, 5, 6, 9, 13, 15, 17), backend='gpu')
def generate(
        sim: OrderBook,  # static
        train_state: TrainState,
        model: nn.Module,  # static
        batchnorm: bool,  # static
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        sample_top_n: int,  # static
        tick_size: int,  # static
        m_seq_cond: jax.Array,
        b_seq_cond: jax.Array,
        n_msg_todo: int,  # static
        sim_state: LobState,
        rng: jax.dtypes.prng_key,
        init_hidden: Tuple,
        conditional: bool,  # static
        init_time: jax.Array,
        debug_book: bool = False,  # static
        b_seq_real: Optional[jax.Array] = None,
        token_mode: int = 24,  # static
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, Tuple]:
    """
    Generate messages autoregressively with hidden state accumulation.

    Args:
        sim: OrderBook simulator (static)
        train_state: Model parameters
        model: Flax model (static)
        batchnorm: Whether to use batchnorm (static)
        encoder: Token encoder dictionary
        sample_top_n: Number of top tokens to sample from (static)
        tick_size: Price tick size (static)
        m_seq_cond: Conditioning message sequence
        b_seq_cond: Conditioning book sequence
        n_msg_todo: Number of messages to generate (static)
        sim_state: Initial simulator state
        rng: Random key
        init_hidden: Initial hidden state
        conditional: Whether to condition on input sequence (static)
        init_time: Initial time array
        debug_book: Whether to use real book for debugging (static)
        b_seq_real: Real book sequence for debugging
        token_mode: Token vocabulary mode (static)

    Returns:
        Tuple of (msgs_decoded, l2_book_states, num_errors, msgs_tokens, hidden_state_final)
    """
    # Note: This function is JIT-compiled. Print statements cause retracing.
    # Remove or use jax.debug.print for debugging.

    with jax.ensure_compile_time_eval():
        v = Vocab(token_mode=token_mode)
        valid_mask_array = valh.syntax_validation_matrix(v)

    if not debug_book:
        b_seq_real = None

    if conditional:
        def roll_hidden_scan(carry, xs):
            m_seq, b_seq = xs
            h = carry
            h, log = valh.apply_model(
                h,
                m_seq,
                b_seq,
                train_state,
                model,
                batchnorm,
                True
            )
            carry = h
            return carry, None

        # Process the conditioning sequence to "warm up" the hidden state
        # For padded merging, we need n_tokens to be divisible by MSG_LEN to align with books
        # Each book state corresponds to MSG_LEN message tokens
        from lob.encoding import Message_Tokenizer
        MSG_LEN = Message_Tokenizer.MSG_LEN

        # Calculate number of complete messages in conditioning sequence
        n_tokens = m_seq_cond.shape[0]
        n_msgs_cond = n_tokens // MSG_LEN

        # Process all messages except the last one to warm up hidden state
        # The last message's tokens will be processed during generation
        n_msgs_to_process = n_msgs_cond - 1
        n_tokens_to_process = n_msgs_to_process * MSG_LEN

        m_seq_to_process = m_seq_cond[:n_tokens_to_process]
        b_seq_to_process = b_seq_cond[:n_msgs_to_process]

        # Process conditioning tokens through the model
        hidden_state, _ = valh.apply_model(
            init_hidden,
            m_seq_to_process,
            b_seq_to_process,
            train_state,
            model,
            batchnorm,
            True
        )

        # Now process the last message's tokens (all but the very last token)
        # to continue warming up hidden state
        last_msg_tokens = m_seq_cond[n_tokens_to_process:-1]  # Last message minus final token
        last_book = b_seq_cond[-1:]  # Last book state

        if last_msg_tokens.shape[0] > 0:
            hidden_state, _ = valh.apply_model(
                hidden_state,
                last_msg_tokens,
                last_book,
                train_state,
                model,
                batchnorm,
                True
            )

        # Final token starts generation
        init_token = m_seq_cond[-1:]  # Very last token
        init_book = b_seq_cond[-1:]
        init_time = jnp.asarray(valh.get_first_time(m_seq_cond, encoder))
    else:
        hidden_state = init_hidden
        assert (m_seq_cond.ndim == 1) & (m_seq_cond.shape[0] == 1), "m_seq_cond needs to be a scalar (start tok?)"
        init_token = m_seq_cond
        init_book = b_seq_cond

    # Get initial mid price
    p_mid = _get_safe_mid_price(sim, sim_state, tick_size)

    generate_msg_scannable = _make_generate_msg_scannable(
        sim, train_state, model, batchnorm,
        encoder, valid_mask_array, sample_top_n, tick_size, debug_book, token_mode,
    )

    gen_state, (msgs_decoded, l2_book_states, msgs_tokens) = jax.lax.scan(
        generate_msg_scannable,
        (init_token, init_book, n_msg_todo, p_mid, sim_state, rng, hidden_state, init_time),
        length=n_msg_todo,
        xs=b_seq_real,
    )
    (final_token, final_book, n_msg_todo, p_mid, sim_state, rng, hidden_state, final_time) = gen_state

    # Count errors
    num_errors = (l2_book_states[1:] == l2_book_states[:-1]).all(axis=1).sum()

    # Return including final hidden state
    return msgs_decoded, l2_book_states, num_errors, msgs_tokens, hidden_state


# Vmapped and jitted version of generate for batch processing (single GPU)
generate_batched = jax.jit(
    jax.vmap(
        generate,
        in_axes=(
            None, None, None, None, None,  # sim, train_state, model, batchnorm, encoder
            None, None,    0,    0, None,  # sample_top_n, tick_size, m_seq_cond, b_seq_cond, n_msg_todo
            0,       0,    0, None,    0,  # sim_state, rng, init_hidden, conditional, init_time
            None,    0, None,              # debug_book, b_seq_real, token_mode
        )
    ),
    static_argnums=(0, 2, 3, 5, 6, 9, 13, 15, 17),
    backend='gpu'
)


# ============================================================================
# Multi-GPU pmap version for parallel execution across devices
# ============================================================================

def create_generate_batched_pmap(num_devices: int = 2):
    """
    Create a pmap+vmap version of generate for multi-GPU execution.

    This splits the batch across multiple GPUs for parallel execution.
    Each GPU processes batch_size // num_devices samples.

    Args:
        num_devices: Number of GPUs to use (default: 2)

    Returns:
        A function that takes inputs with leading device dimension:
        - Input shape: (num_devices, per_device_batch, ...)
        - Output shape: (num_devices, per_device_batch, ...)

    Usage:
        generate_pmap = create_generate_batched_pmap(num_devices=2)
        # Reshape inputs: (batch, ...) -> (num_devices, per_device, ...)
        results = generate_pmap(sim, train_state, model, ...)
        # Reshape outputs: (num_devices, per_device, ...) -> (batch, ...)
    """
    # Inner vmap handles samples within each device
    vmapped_generate = jax.vmap(
        generate,
        in_axes=(
            None, None, None, None, None,  # sim, train_state, model, batchnorm, encoder
            None, None,    0,    0, None,  # sample_top_n, tick_size, m_seq_cond, b_seq_cond, n_msg_todo
            0,       0,    0, None,    0,  # sim_state, rng, init_hidden, conditional, init_time
            None,    0, None,              # debug_book, b_seq_real, token_mode
        )
    )

    # Outer pmap distributes across GPUs
    # Note: pmap in_axes specifies which args have device dimension
    # - 0 means arg has leading device dimension
    # - None means arg is replicated across devices
    pmapped_generate = jax.pmap(
        vmapped_generate,
        in_axes=(
            None,    0, None, None, None,  # sim, train_state (replicated!), model, batchnorm, encoder
            None, None,    0,    0, None,  # sample_top_n, tick_size, m_seq_cond, b_seq_cond, n_msg_todo
            0,       0,    0, None,    0,  # sim_state, rng, init_hidden, conditional, init_time
            None,    0, None,              # debug_book, b_seq_real, token_mode
        ),
        static_broadcasted_argnums=(0, 2, 3, 5, 6, 9, 13, 15, 17),
        devices=jax.devices()[:num_devices] if len(jax.devices()) >= num_devices else jax.devices(),
    )

    return pmapped_generate


def reshape_for_pmap(arr, num_devices):
    """Reshape array from (batch, ...) to (num_devices, per_device, ...)."""
    if arr is None:
        return None
    if not hasattr(arr, 'shape'):
        # Not an array, return as-is
        return arr
    batch_size = arr.shape[0]
    per_device = batch_size // num_devices
    new_shape = (num_devices, per_device) + arr.shape[1:]
    return arr.reshape(new_shape)


def reshape_from_pmap(arr, batch_size):
    """Reshape array from (num_devices, per_device, ...) to (batch, ...)."""
    if arr is None:
        return None
    if not hasattr(arr, 'shape'):
        # Not an array, return as-is
        return arr
    new_shape = (batch_size,) + arr.shape[2:]
    return arr.reshape(new_shape)


def reshape_hidden_for_pmap(hidden_state, num_devices):
    """
    Reshape hidden state pytree for pmap.

    Hidden state can be a nested structure (tuple of lists of arrays, etc.).
    Uses jax.tree_map to handle arbitrary nesting.
    """
    return jax.tree.map(
        lambda x: reshape_for_pmap(x, num_devices),
        hidden_state
    )


def reshape_hidden_from_pmap(hidden_state, batch_size):
    """
    Reshape hidden state pytree back from pmap.
    """
    return jax.tree.map(
        lambda x: reshape_from_pmap(x, batch_size),
        hidden_state
    )


# ============================================================================
# Additional helper functions - re-export from original module for compatibility
# ============================================================================

# Note: get_sim and get_sims_vmap have the same signature as the original module
# to maintain backwards compatibility with the scenario scripts
from lob.inference_no_errcorr import (
    get_sim,
    get_sims_vmap,
    batched_get_safe_mid_price,
)


# Re-export useful functions from original module
from lob.inference_no_errcorr import (
    get_dataset,
    msg_to_lobster_format,
    book_to_lobster_format,
    msg_to_jnp,
)
