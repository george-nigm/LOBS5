#!/usr/bin/env python
"""
RWKV Aggressive Scenario v2: Market Impact Simulation

Clean rewrite using S5's data pipeline (inference.get_dataset, get_sims_vmap).
Replicates 1.aggressive_scenario_s5.py but with RWKV model for generation.

Output format matches S5/CST/CGAN scenarios for existing analysis notebooks.
"""

import argparse
import os
import sys
import time as time_mod
import yaml
from datetime import datetime
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
from tqdm import tqdm

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(script_dir)
sys.path.insert(0, parent_folder_path)

# AlphaTrade submodule
if os.path.exists('/AlphaTrade'):
    sys.path.insert(0, '/AlphaTrade')
else:
    sys.path.insert(0, os.path.join(parent_folder_path, 'Alphatrade'))

from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

# S5 infrastructure (same as 1.aggressive_scenario_s5.py)
from lob.encoding import Vocab, Message_Tokenizer
import lob.inference_no_errcorr_w_insertions as inference

# ============================================================================
# Constants
# ============================================================================

# Decoded message field indices (14 fields, matches S5 output)
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

# RWKV BPE special tokens
NEWLINE_TOKEN = 36
PAD_TOKEN = 0
ORDERBOOK_START_TOKEN = 775

# LOBSTER field tags (from lobgen/constants.py)
TIME_COL = "<time>"
EVENT_TYPE_COL = "<event_type>"
ORDER_ID_COL = "<order_id>"
SIZE_COL = "<size>"
PRICE_COL = "<price>"
DIRECTION_COL = "<direction>"

SIM_CONFIG = JAXLOB_Configuration()


# ============================================================================
# Logging (same as S5)
# ============================================================================

class TeeLogger:
    """Duplicates output to both console and log file."""
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', buffering=1)

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
    log_file = save_folder / 'experiment.log'
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee


def create_experiment_folder(base_dir: str) -> Path:
    base = Path(base_dir)
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


# ============================================================================
# msg_seq_raw -> RWKV BPE tokens (bridge between S5 data and RWKV model)
# ============================================================================

def msg_seq_raw_to_rwkv_tokens(
    msg_seq_raw_np: onp.ndarray,
    tokenizer,
) -> Tuple[List[onp.ndarray], List[int]]:
    """Convert S5-format msg_seq_raw to RWKV BPE tokens for conditioning.

    Direction convention:
      msg_seq_raw: 0=sell side, 1=buy side (from preprocessing: (lobster+1)/2)
      RWKV text:   -1=sell side, 1=buy side (raw LOBSTER format)
      Conversion:  rwkv_dir = msg_seq_raw_dir * 2 - 1

    Time convention:
      msg_seq_raw has TIMEs (absolute seconds) and TIMEns (absolute nanoseconds fraction)
      RWKV text has absolute timestamp in nanoseconds: TIMEs * 1e9 + TIMEns

    Args:
        msg_seq_raw_np: (batch, n_msgs, 14) numpy array
        tokenizer: PreTrainedTokenizerFast BPE tokenizer

    Returns:
        all_tokens: list of 1D int32 arrays (one per batch item)
        all_lengths: list of token counts
    """
    batch_size = msg_seq_raw_np.shape[0]
    all_tokens = []
    all_lengths = []

    for b in range(batch_size):
        msgs = msg_seq_raw_np[b]
        rows = []
        for m_idx in range(len(msgs)):
            msg = msgs[m_idx]
            # Absolute timestamp in nanoseconds (model was trained on absolute times)
            delta_ns = int(msg[TIMEs_i]) * 1_000_000_000 + int(msg[TIMEns_i])
            # Convert direction: msg_seq_raw {0=sell,1=buy} -> LOBSTER {-1=sell,1=buy}
            direction_raw = int(msg[DIRECTION_i]) * 2 - 1
            row = (
                f"{TIME_COL},{delta_ns},"
                f"{EVENT_TYPE_COL},{int(msg[EVENT_TYPE_i])},"
                f"{ORDER_ID_COL},{int(msg[ORDER_ID_i])},"
                f"{SIZE_COL},{int(msg[SIZE_i])},"
                f"{PRICE_COL},{int(msg[PRICE_ABS_i])},"
                f"{DIRECTION_COL},{direction_raw}\n"
            )
            rows.append(row)

        tokenized = tokenizer(
            rows, return_attention_mask=False, return_tensors="np"
        )["input_ids"]
        tokens = onp.concatenate(tokenized).astype(onp.int32)
        all_tokens.append(tokens)
        all_lengths.append(len(tokens))

    return all_tokens, all_lengths


# ============================================================================
# Token <-> message bridge (RWKV-specific)
# ============================================================================

def build_vocab_lookup(tokenizer, vocab_size: int = None) -> dict:
    """Pre-build {token_id: str} dict (~50x faster than tokenizer.decode per token)."""
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    lookup = {}
    for i in range(vocab_size):
        try:
            lookup[i] = tokenizer.decode(i).strip()
        except Exception:
            lookup[i] = ""
    return lookup


def parse_block_tokens(
    tokenizer, tokens_1d, starting_time_ns: int = 0, vocab_lookup: dict = None,
) -> Tuple[List[dict], int]:
    """
    Parse 1D BPE token array into list of message dicts.

    Accumulates field strings and parses at newline boundaries (handles
    multi-token BPE numbers correctly).

    Direction in output: LOBSTER convention (1=buy side, -1=sell side).

    Returns (messages, final_time_ns).
    """
    messages = []
    cumulative_ns = starting_time_ns
    field_strings = {}
    num_tags = 0
    current_tag = None
    orderbook_active = False

    for t in tokens_1d:
        t = int(t)
        if t == PAD_TOKEN:
            continue
        if t == ORDERBOOK_START_TOKEN:
            orderbook_active = True
            continue
        if t == NEWLINE_TOKEN:
            if orderbook_active:
                orderbook_active = False
                continue
            # End of message — parse fields
            if num_tags == 6:
                try:
                    time_str = field_strings.get('time', '0').replace(',', '').strip()
                    abs_ns = int(time_str)
                    if abs_ns > 0:
                        cumulative_ns = abs_ns  # SET absolute (not accumulate delta)
                    msg = {
                        'time_delta_ns': abs_ns,
                        'event_type': int(field_strings.get('event_type', '0').replace(',', '')),
                        'order_id': int(field_strings.get('order_id', '0').replace(',', '')),
                        'size': int(field_strings.get('size', '0').replace(',', '')),
                        'price': int(field_strings.get('price', '0').replace(',', '')),
                        'direction': int(field_strings.get('direction', '0').replace(',', '')),
                        'abs_time_ns': cumulative_ns,
                    }
                    INT32_MAX = 2_147_483_647
                    if (msg['event_type'] in (1, 2, 3, 4)
                            and msg['size'] > 0 and msg['price'] > 0
                            and abs(msg['price']) < INT32_MAX
                            and abs(msg['order_id']) < INT32_MAX
                            and abs(msg['size']) < INT32_MAX):
                        messages.append(msg)
                except (ValueError, OverflowError):
                    pass
            field_strings = {}
            num_tags = 0
            current_tag = None
            continue
        if orderbook_active:
            continue

        decoded = vocab_lookup[t] if vocab_lookup is not None else tokenizer.decode(t).strip()
        is_tag = ("<" in decoded)
        if is_tag:
            num_tags += 1
            if "time" in decoded:
                current_tag = 'time'
            elif "event_type" in decoded:
                current_tag = 'event_type'
            elif "order_id" in decoded:
                current_tag = 'order_id'
            elif "size" in decoded:
                current_tag = 'size'
            elif "price" in decoded:
                current_tag = 'price'
            elif "direction" in decoded:
                current_tag = 'direction'
            else:
                current_tag = None
        else:
            if current_tag:
                field_strings[current_tag] = field_strings.get(current_tag, '') + decoded

    return messages, cumulative_ns


def format_aggressive_as_rwkv_text(
    time_delta_ns: int, event_type: int, order_id: int,
    size: int, price: int, direction_lobster: int,
) -> str:
    """Format aggressive order as RWKV text for tokenization.

    direction_lobster: LOBSTER convention (-1=sell side, 1=buy side).
    """
    return (
        f"{TIME_COL},{time_delta_ns},"
        f"{EVENT_TYPE_COL},{event_type},"
        f"{ORDER_ID_COL},{order_id},"
        f"{SIZE_COL},{size},"
        f"{PRICE_COL},{price},"
        f"{DIRECTION_COL},{direction_lobster}\n"
    )


# ============================================================================
# Sim message conversion
# ============================================================================

def msg_to_sim_array(msg: dict) -> jax.Array:
    """Convert parsed RWKV message dict to 8-field JAX-LOB sim message.

    RWKV direction (LOBSTER): 1=buy side, -1=sell side
    msg_seq_raw convention:   1=buy side, 0=sell side
    construct_sim_msg does:   (side*2)-1 to get simulator format
    """
    d = msg['direction']  # LOBSTER: 1 or -1
    side_01 = (d + 1) // 2  # 1->1 (buy), -1->0 (sell)
    abs_ns = msg['abs_time_ns']
    return inference.construct_sim_msg(
        jnp.int32(msg['event_type']),
        jnp.int32(side_01),
        jnp.int32(msg['size']),
        jnp.int32(msg['price']),
        jnp.int32(msg['order_id']),
        jnp.int32(abs_ns // 1_000_000_000),
        jnp.int32(abs_ns % 1_000_000_000),
    )


def msg_to_decoded_14(msg: dict, sim: OrderBook, sim_state: LobState, tick_size: int) -> jax.Array:
    """Convert parsed message dict to 14-field decoded array (LOBSTER output format).

    Call BEFORE applying msg to sim (so mid price is pre-message).
    """
    d = msg['direction']  # LOBSTER: 1 or -1
    side_01 = (d + 1) // 2  # 1->1, -1->0 (msg_seq_raw convention)
    abs_ns = msg['abs_time_ns']
    time_s = abs_ns // 1_000_000_000
    time_ns = abs_ns % 1_000_000_000

    best_ask = sim.get_best_ask(sim_state)
    best_bid = sim.get_best_bid(sim_state)
    mid_price = (best_ask + best_bid) // 2
    mid_price = (mid_price // tick_size) * tick_size
    rel_price = jax.lax.cond(
        mid_price > 0,
        lambda: (jnp.int32(msg['price']) - mid_price) // tick_size,
        lambda: jnp.int32(0),
    )

    return jnp.array([
        msg['order_id'], msg['event_type'], side_01, msg['price'],
        rel_price, msg['size'], 0, 0, time_s, time_ns,
        0, 0, 0, 0,
    ], dtype=jnp.int32)


# ============================================================================
# OID update for cancel/execution messages
# ============================================================================

def update_oid(msg: jax.Array, rng: jax.Array, sim: OrderBook, sim_state: LobState):
    """Assign correct order ID for cancel/execution messages using JAX-LOB state.

    sim_msg format from construct_sim_msg:
      [event_type, side, quantity, price, order_id, -88, time_s, time_ns]
       idx 0       1     2        3      4         5    6       7
    """
    OID_IDX = 4  # order_id position in sim message array

    def _leave_oid(msg, rng, *args):
        return msg, rng

    def _get_random_cancel_msg(msg, rng, sim, sim_state):
        side = msg[1]
        side_array = jax.lax.cond(
            side == 1, lambda a, b: b, lambda a, b: a,
            sim_state.asks, sim_state.bids
        )
        rng, _rng = jax.random.split(rng)
        msg_dict = {"quantity": msg[2], "price": msg[3]}
        idx = job.get_random_id_match(SIM_CONFIG, _rng, side_array, msg_dict)
        cancelled_oid = side_array[idx, 2]
        return msg.at[OID_IDX].set(cancelled_oid), rng

    def _get_oid_from_active(msg, rng, sim, sim_state):
        side = msg[1]
        def _top_bid_oid(ss):
            idx = job._get_top_bid_order_idx(SIM_CONFIG, ss.bids).squeeze()
            return ss.bids[idx, 2]
        def _top_ask_oid(ss):
            idx = job._get_top_ask_order_idx(SIM_CONFIG, ss.asks).squeeze()
            return ss.asks[idx, 2]
        oid = jax.lax.cond(side == 1, _top_bid_oid, _top_ask_oid, sim_state)
        return msg.at[OID_IDX].set(oid), rng

    msg, rng = jax.lax.switch(
        ((msg[0] == 2) | (msg[0] == 3)) + 2 * (msg[0] == 4),
        (_leave_oid, _get_random_cancel_msg, _get_oid_from_active),
        msg, rng, sim, sim_state,
    )
    return msg, rng


# ============================================================================
# RWKV batched generation (Python loop + vmapped forward)
# ============================================================================

def make_batched_generate_fn(rwkv_forward, simple_sampler_fn, temperature: float):
    """
    Create batched generation function using Python for-loop.

    Avoids XLA compilation explosion from nested scan+vmap by using
    a Python loop with vmapped forward+sampler at each step.
    """
    v_forward_1 = jax.jit(jax.vmap(
        lambda tokens, state, params: rwkv_forward(tokens, state, params, 1),
        in_axes=(0, 0, None)
    ))
    v_sampler = jax.jit(jax.vmap(
        lambda key, out: simple_sampler_fn(key, out, temperature),
        in_axes=(0, 0)
    ))

    def batched_generate_block(states, outs, rng, params, max_tokens, target_msgs):
        """Generate tokens until each sample has target_msgs newlines.

        Returns (states, outs, rng, tokens_array: (batch, max_tokens) int32).
        """
        batch_size = outs.shape[0]
        all_tokens = onp.zeros((batch_size, max_tokens), dtype=onp.int32)
        msg_counts = jnp.zeros(batch_size, dtype=jnp.int32)
        active = jnp.ones(batch_size, dtype=jnp.bool_)

        for t in range(max_tokens):
            if not jnp.any(active):
                break

            rng, rng_ = jax.random.split(rng)
            step_keys = jax.random.split(rng_, batch_size)

            sampled = v_sampler(step_keys, outs)
            tokens_t = jnp.where(active, sampled.astype(jnp.int32), PAD_TOKEN)
            all_tokens[:, t] = onp.array(tokens_t)

            token_batch = tokens_t[:, None]
            outs_new, states_new = v_forward_1(token_batch, states, params)

            def cond_update(old, new):
                mask_shape = (active.shape[0],) + (1,) * (old.ndim - 1)
                return jnp.where(active.reshape(mask_shape), new, old)

            states = jax.tree_util.tree_map(cond_update, states, states_new)
            outs = jnp.where(active[:, None], outs_new[:, 0], outs)

            is_newline = (tokens_t == NEWLINE_TOKEN)
            msg_counts = msg_counts + is_newline.astype(jnp.int32) * active.astype(jnp.int32)
            active = msg_counts < target_msgs

        return states, outs, rng, jnp.array(all_tokens)

    return batched_generate_block


# ============================================================================
# Aggressive order creation (same logic as S5)
# ============================================================================

def create_aggressive_order(
    sim: OrderBook, sim_state: LobState,
    last_time_s: jax.Array, last_time_ns: jax.Array,
    tick_size: int, event_type: int, direction: int, order_volume: int,
    order_id: int,
) -> Tuple[jax.Array, jax.Array]:
    """Create aggressive market order from current book state.

    direction: 0=buy aggression (hit ask), 1=sell aggression (hit bid).
    Same convention as S5 (passive order's side in msg_seq_raw format).
    order_id: countdown ID (same as S5's n_msg_todo sequence).

    Returns (sim_msg, msg_decoded_14).
    """
    price = jax.lax.cond(
        direction == 0,
        lambda: sim.get_best_ask(sim_state),
        lambda: sim.get_best_bid(sim_state),
    )
    best_ask_pv, best_bid_pv = sim.get_best_bid_and_ask_inclQuants(sim_state)
    avail = jax.lax.cond(
        direction == 0,
        lambda: best_ask_pv[1],
        lambda: best_bid_pv[1],
    ).astype(jnp.int32)
    quantity = jnp.minimum(jnp.int32(order_volume), avail)

    time_s = last_time_s.astype(jnp.int32)
    time_ns = (last_time_ns + 1).astype(jnp.int32)

    oid = jnp.int32(order_id)
    sim_msg = inference.construct_sim_msg(event_type, direction, quantity, price,
                                          oid, time_s, time_ns)

    mid_price = (sim.get_best_ask(sim_state) + sim.get_best_bid(sim_state)) // 2
    mid_price = (mid_price // tick_size) * tick_size
    rel_price = (price - mid_price) // tick_size

    msg_decoded = jnp.array([
        oid, event_type, direction, price, rel_price, quantity,
        0, 1, time_s, time_ns, 0, 0, 0, 0,
    ], dtype=jnp.int32)

    return sim_msg, msg_decoded


# ============================================================================
# Main scenario function
# ============================================================================

def run_rwkv_scenario(cfg: Dict[str, Any], save_folder: Path):
    """
    Main pipeline:
    1. Load RWKV model + tokenizer
    2. Load dataset (same as S5: inference.get_dataset)
    3. For each batch: condition -> generate blocks -> inject -> replay -> save
    """
    # Unpack config
    lobgen_dir = cfg['lobgen_dir']
    model_choice = cfg['model_choice']
    rwkv_type = cfg['rwkv_type']
    ckpt_path = cfg['ckpt_path']
    tokenizer_file = cfg['tokenizer_file']
    temperature = cfg.get('temperature', 1.0)
    data_dir = cfg['data_dir']
    stock = cfg['stock']
    tick_size = cfg['tick_size']
    n_cond_msgs = cfg['n_cond_msgs']
    n_gen_msgs = cfg['n_gen_msgs']
    max_tokens_per_block = cfg.get('max_tokens_per_block', 2000)
    num_insertions = cfg['num_insertions']
    num_coolings = cfg['num_coolings']
    event_type = cfg['event_type']
    direction = cfg['direction']
    order_volume = cfg['order_volume']
    n_samples = cfg['n_samples']
    batch_size = cfg['batch_size']
    rng_seed = cfg['rng_seed']
    n_levels = cfg.get('n_levels', 10)
    process_long_seq_padding = cfg.get('process_long_seq_padding', 128)
    max_cond_tokens = cfg.get('max_cond_tokens', 15000)
    n_eval_msgs_dataset = cfg.get('n_eval_msgs_dataset', 500)
    test_split = cfg.get('test_split', 0)

    total_blocks = num_insertions + num_coolings

    # === 1. Load RWKV model ===
    print(f"Adding lobgen to path: {lobgen_dir}")
    sys.path.insert(0, lobgen_dir)

    from jax_rwkv.auto import get_model, load as rwkv_load
    from jax_rwkv.utils import simple_sampler
    from transformers import PreTrainedTokenizerFast

    print(f"Loading RWKV model: {model_choice} (type={rwkv_type})")
    RWKV, _params, _ = get_model(model_choice, rwkv_type=rwkv_type)

    # Load checkpoint
    ckpt_file = ckpt_path
    if os.path.isdir(ckpt_path):
        ckpt_file = os.path.join(ckpt_path, "params.model")
    print(f"Loading checkpoint: {ckpt_file}")
    params = rwkv_load(ckpt_file)

    forward_jit = jax.jit(RWKV.forward)

    # Load tokenizer
    tokenizer_path = os.path.join(lobgen_dir, tokenizer_file)
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        clean_up_tokenization_spaces=False,
    )

    # Build vocab lookup
    print("Building vocab lookup...")
    vocab_lookup = build_vocab_lookup(tokenizer)
    print(f"Vocab lookup: {len(vocab_lookup)} entries")

    # Sanity check
    test_state = RWKV.default_state(params)
    test_out, _ = RWKV.forward(jnp.array([1]), test_state, params, 1)
    if onp.any(onp.isnan(onp.array(test_out[0]).astype(onp.float32))):
        raise RuntimeError("Checkpoint produces NaN on forward pass.")
    print("Sanity check passed (no NaN)")

    # Build vmapped conditioning function
    def process_long_seq(tokens, state, length, padding=process_long_seq_padding):
        full_len = tokens.shape[-1]
        x = (jnp.zeros_like(params['emb']['weight'][:, 0]), state)

        def inner_loop(x, i):
            true_out, st = x
            cur_len = jnp.minimum(padding, length - i)
            safe_len = jnp.maximum(cur_len, 1)
            out, new_st = forward_jit(
                jax.lax.dynamic_slice_in_dim(tokens, i, padding),
                st, params, safe_len,
            )
            st = jax.lax.cond(cur_len <= 0, lambda: st, lambda: new_st)
            true_out = jax.lax.cond(cur_len <= 0, lambda: true_out, lambda: out[cur_len - 1])
            return (true_out, st), 0

        (true_out, state), _ = jax.lax.scan(
            inner_loop, x, jnp.arange(0, full_len, padding)
        )
        return true_out, state

    v_process_long_seq = jax.jit(jax.vmap(process_long_seq))

    # Build batched generation
    batched_generate = make_batched_generate_fn(RWKV.forward, simple_sampler, temperature)

    # === 2. Load dataset (SAME as S5) ===
    print(f"Loading dataset from: {data_dir}")
    ds = inference.get_dataset(
        data_dir,
        n_cond_msgs,
        n_eval_msgs_dataset,
        test_split=test_split,
    )
    print(f"Dataset length: {len(ds)}")

    # === 3. Sample indices (SAME as S5) ===
    rng = jax.random.key(rng_seed)
    rng, _ = jax.random.split(rng)
    rng, rng_ = jax.random.split(rng)
    assert n_samples % batch_size == 0, f'n_samples ({n_samples}) must be divisible by batch_size ({batch_size})'
    sample_i = jax.random.choice(
        rng_,
        jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size),
        replace=False
    ).tolist()
    n_batches = n_samples // batch_size

    # Output folders
    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    # Simulator
    sim = OrderBook(cfg=JAXLOB_Configuration(
        cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value
    ))

    # Build aggressive positions list and total msg count (same as S5)
    # S5: n_msg_todo_total = total_gen_msgs + num_insertions
    total_gen_msgs = total_blocks * n_gen_msgs
    n_msg_todo_total = total_gen_msgs + num_insertions

    aggressive_positions = []
    pos = 0
    for block in range(total_blocks):
        pos += n_gen_msgs
        if block < num_insertions:
            aggressive_positions.append(pos)
            pos += 1
    print(f"Aggressive positions: {aggressive_positions}")
    print(f"Total messages per sample (n_msg_todo_total): {n_msg_todo_total}")

    rng, rng_ = jax.random.split(rng)

    # === 4. Batch loop ===
    for batch_idx, batch_i in enumerate(tqdm(sample_i, desc="Batches")):
        print(f'\n=== BATCH {batch_idx}: samples {batch_i} ===')

        # a. Load data (SAME as S5)
        gpu_device = jax.devices('gpu')[0]
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        msg_seq_raw = jax.device_put(jnp.array(msg_seq_raw), gpu_device)
        book_l2_init = jax.device_put(jnp.array(book_l2_init), gpu_device)
        b_seq_pv = onp.array(b_seq_pv)

        # b. Split conditioning
        m_seq_raw_cond = msg_seq_raw[:, :n_cond_msgs]
        init_time_batched = jnp.array(b_seq_pv[:, 0, 1:3], dtype=jnp.int32)
        b_seq_pv_cond = onp.array(b_seq_pv[:, :n_cond_msgs + 1, 3:])

        # c. Initialize simulators (SAME as S5)
        print("  Initializing simulators...")
        sim_states_batched = inference.get_sims_vmap(
            book_l2_init,
            m_seq_raw_cond,
            init_time_batched,
            sim,
        )

        # Unpack into list for per-sample processing
        sim_states = [
            jax.tree_util.tree_map(lambda x: x[i], sim_states_batched)
            for i in range(batch_size)
        ]

        # d. Convert conditioning messages to RWKV tokens
        m_seq_raw_cond_np = onp.array(m_seq_raw_cond)
        cond_tokens_list, cond_lengths = msg_seq_raw_to_rwkv_tokens(
            m_seq_raw_cond_np, tokenizer
        )

        # e. Pad conditioning tokens (dynamic per-batch)
        cond_lengths_clipped = [min(cl, max_cond_tokens) for cl in cond_lengths]
        batch_max_len = max(cond_lengths_clipped)
        dynamic_cond_len = (
            (batch_max_len + process_long_seq_padding - 1)
            // process_long_seq_padding
        ) * process_long_seq_padding
        dynamic_cond_len = min(dynamic_cond_len, max_cond_tokens)

        padded_cond = []
        for ct, length in zip(cond_tokens_list, cond_lengths_clipped):
            padded = onp.zeros(dynamic_cond_len, dtype=onp.int32)
            padded[:length] = ct[:length]
            padded_cond.append(padded)

        cond_batch = jnp.array(onp.stack(padded_cond))
        lengths_batch = jnp.array(cond_lengths_clipped, dtype=jnp.int32)
        print(f"  Cond tokens: min={min(cond_lengths_clipped)}, "
              f"max={max(cond_lengths_clipped)}, padded to {dynamic_cond_len}")

        # f. RWKV conditioning
        print("  Running RWKV conditioning...")
        t0 = time_mod.time()
        init_state = RWKV.default_state(params)
        states = jnp.repeat(init_state[None], batch_size, axis=0)
        outs, states = v_process_long_seq(cond_batch, states, lengths_batch)
        jax.block_until_ready(outs)
        print(f"  Conditioning done in {time_mod.time()-t0:.1f}s")

        # g. Save conditioning data (SAME as S5)
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)
            inference.msg_to_lobster_format(m_seq_raw_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_message_real_id_{sample_idx}.csv',
                index=False, header=False,
            )
            inference.book_to_lobster_format(b_seq_pv_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_orderbook_real_id_{sample_idx}.csv',
                index=False, header=False,
            )

        # h. Generation + insertion blocks
        rng, gen_rng = jax.random.split(rng)

        # Track per-sample time
        last_times_s = []
        last_times_ns = []
        cumulative_times_ns = []
        for i in range(batch_size):
            last_msg = m_seq_raw_cond_np[i, n_cond_msgs - 1]
            t_s = int(last_msg[TIMEs_i])
            t_ns = int(last_msg[TIMEns_i])
            last_times_s.append(t_s)
            last_times_ns.append(t_ns)
            cumulative_times_ns.append(t_s * 1_000_000_000 + t_ns)

        # First-pass order ID counter for RWKV text (avoid fake IDs leaking)
        # Uses same countdown as S5: n_msg_todo starts at n_msg_todo_total
        fwd_oid_counter = n_msg_todo_total

        parsed_messages_cache = []  # [block_idx][sample_idx] = (messages, cum_ns)

        for block in range(total_blocks):
            is_insertion = block < num_insertions
            block_type = "INSERT" if is_insertion else "COOL"
            t0 = time_mod.time()
            print(f"  Block {block}/{total_blocks} ({block_type}): gen {n_gen_msgs} msgs...",
                  end="", flush=True)

            # Generate
            states, outs, gen_rng, block_tokens = batched_generate(
                states, outs, gen_rng, params,
                max_tokens_per_block, n_gen_msgs,
            )
            block_tokens_np = onp.array(block_tokens)
            non_pad = onp.count_nonzero(block_tokens_np)
            print(f" done {time_mod.time()-t0:.1f}s ({non_pad} tokens)", flush=True)

            # Parse and apply to simulators
            block_parsed = []
            for i in range(batch_size):
                messages, new_cum_ns = parse_block_tokens(
                    tokenizer, block_tokens_np[i], cumulative_times_ns[i],
                    vocab_lookup=vocab_lookup,
                )
                block_parsed.append((messages, new_cum_ns))
                cumulative_times_ns[i] = new_cum_ns

                for msg in messages:
                    try:
                        sim_msg = msg_to_sim_array(msg)
                        sim_msg, _ = update_oid(sim_msg, jax.random.key(0), sim, sim_states[i])
                        # Skip if update_oid returned invalid OID for cancel/exec
                        if msg['event_type'] != 1 and int(sim_msg[4]) <= 0:
                            continue
                        sim_states[i] = sim.process_order_array(sim_states[i], sim_msg)
                        last_times_s[i] = msg['abs_time_ns'] // 1_000_000_000
                        last_times_ns[i] = msg['abs_time_ns'] % 1_000_000_000
                    except (OverflowError, ValueError):
                        pass

            # Decrement forward counter for generated messages (use max across batch)
            max_msgs_in_block = max(len(block_parsed[i][0]) for i in range(batch_size))
            fwd_oid_counter -= max_msgs_in_block

            parsed_messages_cache.append(block_parsed)

            # Injection
            if is_insertion:
                print(f"  Injecting aggressive orders...")
                # Aggressive order gets the next countdown ID
                fwd_oid_counter -= 1
                aggr_oid_for_text = fwd_oid_counter

                aggr_texts = []
                for i in range(batch_size):
                    sim_msg, msg_decoded = create_aggressive_order(
                        sim, sim_states[i],
                        jnp.int32(last_times_s[i]), jnp.int32(last_times_ns[i]),
                        tick_size, event_type, direction, order_volume,
                        order_id=aggr_oid_for_text,
                    )
                    sim_states[i] = sim.process_order_array(sim_states[i], sim_msg)

                    aggr_time_s = int(msg_decoded[TIMEs_i])
                    aggr_time_ns = int(msg_decoded[TIMEns_i])
                    last_times_s[i] = aggr_time_s
                    last_times_ns[i] = aggr_time_ns
                    cumulative_times_ns[i] = aggr_time_s * 1_000_000_000 + aggr_time_ns

                    # Direction for RWKV text: config direction -> LOBSTER format
                    # config 0 (buy aggression, sell side) -> LOBSTER -1
                    # config 1 (sell aggression, buy side) -> LOBSTER 1
                    aggr_dir_lobster = direction * 2 - 1
                    aggr_texts.append(format_aggressive_as_rwkv_text(
                        time_delta_ns=cumulative_times_ns[i], event_type=event_type,
                        order_id=aggr_oid_for_text,
                        size=int(msg_decoded[SIZE_i]),
                        price=int(msg_decoded[PRICE_ABS_i]),
                        direction_lobster=aggr_dir_lobster,
                    ))

                # Feed aggressive tokens through RWKV state
                aggr_tokenized = tokenizer(
                    aggr_texts, return_attention_mask=False, return_tensors="np"
                )["input_ids"]
                max_aggr_len = max(len(t) for t in aggr_tokenized)
                padded_aggr_len = (
                    (max_aggr_len + process_long_seq_padding - 1)
                    // process_long_seq_padding
                ) * process_long_seq_padding

                aggr_batch = onp.zeros((batch_size, padded_aggr_len), dtype=onp.int32)
                aggr_lengths = onp.zeros(batch_size, dtype=onp.int32)
                for i in range(batch_size):
                    toks = aggr_tokenized[i]
                    aggr_batch[i, :len(toks)] = toks
                    aggr_lengths[i] = len(toks)

                outs, states = v_process_long_seq(
                    jnp.array(aggr_batch), states,
                    jnp.array(aggr_lengths, dtype=jnp.int32),
                )

        # i. Post-process replay — clean re-init and collect aligned msg+book pairs
        # Uses countdown order IDs matching S5's n_msg_todo convention
        print("  Post-processing and saving...")
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)

            # Re-init simulator for clean replay
            sim_state_replay = jax.tree_util.tree_map(
                lambda x: x[i], sim_states_batched
            )

            all_msgs_decoded = []
            all_l2_books = []
            replay_last_s = int(m_seq_raw_cond_np[i, n_cond_msgs - 1, TIMEs_i])
            replay_last_ns = int(m_seq_raw_cond_np[i, n_cond_msgs - 1, TIMEns_i])

            # Countdown order ID (same as S5: starts at n_msg_todo_total)
            oid_counter = n_msg_todo_total

            for block_idx_inner in range(total_blocks):
                is_ins = block_idx_inner < num_insertions
                messages, _ = parsed_messages_cache[block_idx_inner][i]

                for msg in messages:
                    try:
                        sim_msg = msg_to_sim_array(msg)
                        sim_msg, _ = update_oid(
                            sim_msg, jax.random.key(0), sim, sim_state_replay
                        )

                        # Assign order ID (matches S5 convention):
                        # type 1: countdown ID (new unique order)
                        # type 2, 3, 4: reference from sim (update_oid)
                        et = msg['event_type']
                        if et == 1:
                            output_oid = oid_counter
                            sim_msg = sim_msg.at[4].set(jnp.int32(output_oid))
                        else:
                            output_oid = int(sim_msg[4])
                            # Skip if update_oid couldn't find a valid order
                            # (RWKV generated cancel/exec for non-existent order)
                            if output_oid <= 0:
                                oid_counter -= 1
                                continue

                        # Compute decoded msg BEFORE applying (for correct mid price)
                        msg_d14 = msg_to_decoded_14(msg, sim, sim_state_replay, tick_size)
                        msg_d14 = msg_d14.at[ORDER_ID_i].set(output_oid)

                        sim_state_replay = sim.process_order_array(
                            sim_state_replay, sim_msg
                        )
                        l2_book = sim.get_L2_state(sim_state_replay, n_levels)

                        all_msgs_decoded.append(msg_d14)
                        all_l2_books.append(l2_book)

                        replay_last_s = msg['abs_time_ns'] // 1_000_000_000
                        replay_last_ns = msg['abs_time_ns'] % 1_000_000_000
                    except (OverflowError, ValueError):
                        pass
                    finally:
                        # Always decrement (S5 decrements n_msg_todo each step)
                        oid_counter -= 1

                if is_ins:
                    # Aggressive order gets next countdown ID
                    aggr_oid = oid_counter
                    oid_counter -= 1

                    sim_msg_aggr, msg_decoded_aggr = create_aggressive_order(
                        sim, sim_state_replay,
                        jnp.int32(replay_last_s), jnp.int32(replay_last_ns),
                        tick_size, event_type, direction, order_volume,
                        order_id=aggr_oid,
                    )
                    sim_state_replay = sim.process_order_array(
                        sim_state_replay, sim_msg_aggr
                    )
                    l2_after = sim.get_L2_state(sim_state_replay, n_levels)

                    all_msgs_decoded.append(msg_decoded_aggr)
                    all_l2_books.append(l2_after)

                    replay_last_s = int(msg_decoded_aggr[TIMEs_i])
                    replay_last_ns = int(msg_decoded_aggr[TIMEns_i])

            if len(all_msgs_decoded) == 0:
                print(f"  WARNING: Sample {sample_idx} produced 0 valid messages")
                continue

            all_msgs_arr = jnp.stack(all_msgs_decoded, axis=0)
            all_books_arr = jnp.stack(all_l2_books, axis=0)

            inference.msg_to_lobster_format(all_msgs_arr).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_message_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False,
            )
            inference.book_to_lobster_format(all_books_arr).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_orderbook_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False,
            )

            if batch_idx == 0 and i == 0:
                print(f"  First sample: {all_msgs_arr.shape[0]} total messages, "
                      f"IDs from {n_msg_todo_total} to {oid_counter+1}")

        # Save aggressive indices once
        if batch_idx == 0:
            onp.savetxt(save_folder / 'aggressive_indices.csv',
                        onp.array(aggressive_positions), fmt='%d')

        rng, _ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RWKV Aggressive Scenario v2")
    parser.add_argument('--config', '-c', type=str,
                        default='lob_impact/5v2.aggressive_scenario_rwkv_config_test.yaml')
    parser.add_argument('--n_gen_msgs', type=int, default=None)
    parser.add_argument('--direction', type=int, default=None, choices=[0, 1])
    return parser.parse_args()


def main():
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")

    args = parse_args()
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.n_gen_msgs is not None:
        cfg['n_gen_msgs'] = args.n_gen_msgs
    if args.direction is not None:
        cfg['direction'] = args.direction

    print(f"Config: {cfg}")

    save_folder = create_experiment_folder(cfg['save_dir'])
    print(f"Experiment folder: {save_folder}")

    logger = setup_logging(save_folder)
    print(f"\n{'='*60}")
    print(f"RWKV Aggressive Scenario v2")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Config: {cfg}")
    print(f"Experiment folder: {save_folder}")

    with open(save_folder / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    try:
        run_rwkv_scenario(cfg, save_folder)
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
