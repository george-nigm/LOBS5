#!/usr/bin/env python
"""
RWKV Aggressive Scenario: Market Impact Simulation with RWKV Model

Simulates market impact by injecting aggressive orders into RWKV-generated
LOB sequences. Uses a Python loop with vmapped RWKV forward + sampler for
batched token generation with per-sample early stopping (newline counting).

Key difference from S5: RWKV uses BPE tokenization (20-35 tokens/message vs
fixed 22 tokens/message in S5). Generation uses Python loop instead of
jax.lax.scan to avoid XLA compilation explosion for batch > ~4.

Output format matches S5/CST/CGAN scenarios so existing analysis notebooks
(100-110) work with results from all models.
"""

import argparse
import os
import sys
import time
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

# Add parent folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(script_dir)
sys.path.insert(0, parent_folder_path)

# Add AlphaTrade submodule
if os.path.exists('/AlphaTrade'):
    sys.path.insert(0, '/AlphaTrade')
else:
    sys.path.insert(0, os.path.join(parent_folder_path, 'Alphatrade'))

from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob.inference_no_errcorr import (
    get_sims_vmap, get_dataset, construct_sim_msg,
    msg_to_lobster_format, book_to_lobster_format,
)

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

AGGRESSIVE_ORDER_ID = 77777777

# RWKV special tokens
NEWLINE_TOKEN = 36
PAD_TOKEN = 0
ORDERBOOK_START_TOKEN = 775

# RWKV text field tags (lobgen constants)
TIME_COL = "<time>"
EVENT_TYPE_COL = "<event_type>"
ORDER_ID_COL = "<order_id>"
SIZE_COL = "<size>"
PRICE_COL = "<price>"
DIRECTION_COL = "<direction>"
MESSAGE_TOKEN_TYPES = [TIME_COL, EVENT_TYPE_COL, ORDER_ID_COL, SIZE_COL, PRICE_COL, DIRECTION_COL]


# ============================================================================
# Utilities (shared with other scenario scripts)
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
# Data loading — raw LOBSTER CSV → RWKV tokens
# ============================================================================

def convert_to_nanoseconds(s: pd.Series) -> pd.Series:
    """Convert LOBSTER time strings (e.g., '34200.123456789') to nanoseconds."""
    split_times = s.str.split('.', expand=True)
    split_times.columns = ['seconds', 'fractional']
    seconds = split_times['seconds'].astype(int)
    fractional = split_times['fractional'].fillna('0').str.ljust(9, '0').str[:9].astype(int)
    return seconds * 1_000_000_000 + fractional


def convert_ns_to_lobster_time(total_ns: int) -> str:
    """Convert nanoseconds to LOBSTER time string (e.g., '34200.123456789')."""
    seconds = total_ns // 1_000_000_000
    frac = total_ns % 1_000_000_000
    frac_str = str(frac)
    # Strip trailing zeros
    while len(frac_str) > 0 and frac_str[-1] == '0':
        frac_str = frac_str[:-1]
    if not frac_str:
        frac_str = '0'
    return f"{seconds}.{frac_str}"


class DayDataCache:
    """Cache per-day processed data to avoid re-reading large CSV files."""

    def __init__(self, n_cond_msgs: int, tokenizer):
        self.n_cond_msgs = n_cond_msgs
        self.tokenizer = tokenizer
        self._cache = {}  # date_str → cached data

    def load(self, msg_file: str, ob_file: str, date_str: str):
        """Load day data, using cache if already loaded."""
        if date_str in self._cache:
            return self._cache[date_str]

        result = self._load_day_data(msg_file, ob_file, date_str)
        self._cache[date_str] = result
        return result

    def _load_day_data(self, msg_file: str, ob_file: str, date_str: str):
        """
        Load raw LOBSTER CSV data for one day.

        Only reads first ~n_cond_msgs+100 rows (with margin for event_type filtering).

        Returns:
            cond_tokens: int32 array of conditioning tokens
            cond_msg_records: list of dicts for simulator replay
            l2_init: (n_levels*4,) array of initial L2 book state
            date_str: date string
            first_time_ns: absolute nanosecond time of first message
        """
        n_cond = self.n_cond_msgs
        # Read with margin: some rows may be filtered out (event_type > 4)
        nrows_read = n_cond + 200

        # Load message CSV (limited rows)
        df = pd.read_csv(msg_file, dtype=object, header=None, nrows=nrows_read)
        if len(df.columns) > 6:
            df = df.drop(columns=6)
        df = df.dropna(axis=1).reset_index(drop=True)
        df.columns = MESSAGE_TOKEN_TYPES

        # Filter event_type <= 4
        df = df[df[EVENT_TYPE_COL].astype('int64') <= 4].reset_index(drop=True)

        # Keep original absolute times for simulator
        abs_time_ns = convert_to_nanoseconds(df[TIME_COL])
        first_time_ns = int(abs_time_ns.iloc[0])

        # Build sim-compatible raw messages from first n_cond rows
        cond_raw = df.iloc[:n_cond]
        cond_abs_ns = abs_time_ns.iloc[:n_cond]

        cond_msg_records = []
        for idx in range(len(cond_raw)):
            row = cond_raw.iloc[idx]
            abs_ns = int(cond_abs_ns.iloc[idx])
            cond_msg_records.append({
                'event_type': int(row[EVENT_TYPE_COL]),
                'direction': int(row[DIRECTION_COL]),
                'size': int(row[SIZE_COL]),
                'price': int(row[PRICE_COL]),
                'order_id': int(row[ORDER_ID_COL]),
                'time_s': abs_ns // 1_000_000_000,
                'time_ns': abs_ns % 1_000_000_000,
            })

        # Differentiate time for RWKV tokenization (matches training: differentiate_time=True)
        df[TIME_COL] = abs_time_ns.diff()
        df = df.dropna(subset=TIME_COL).reset_index(drop=True)
        df[TIME_COL] = df[TIME_COL].astype(int)
        cond_df = df.iloc[:n_cond].copy()

        # Format as RWKV text rows and tokenize
        columns = cond_df.columns.tolist()
        values = cond_df.to_numpy()
        row_strings = [
            ','.join([f"{col},{val}" for col, val in zip(columns, row) if not pd.isna(val)]) + "\n"
            for row in values
        ]
        tokenized = self.tokenizer(row_strings, return_attention_mask=False, return_tensors="np")["input_ids"]
        cond_tokens = onp.concatenate(tokenized).astype(onp.int32)

        # Load orderbook CSV — first row as L2 init
        ob_df = pd.read_csv(ob_file, dtype=float, header=None, nrows=1)
        l2_init = ob_df.values[0].astype(onp.int32)

        return cond_tokens, cond_msg_records, l2_init, date_str, first_time_ns


def enumerate_samples(raw_data_dir: str, n_cond_msgs: int):
    """
    Enumerate available samples from raw LOBSTER files.

    Returns list of (msg_file, ob_file, date_str) tuples, one per day.
    """
    from itertools import groupby
    import re

    data_path = Path(raw_data_dir)
    csv_files = sorted(data_path.glob("*.csv"))

    def extract_date(p):
        match = re.search(r'\d{4}-\d{2}-\d{2}', p.name)
        return match.group(0) if match else None

    file_groups = [(key, list(group)) for key, group in groupby(csv_files, key=extract_date)]

    samples = []
    for date_str, group in file_groups:
        if date_str is None:
            continue
        msg_file = None
        ob_file = None
        for f in group:
            if 'message' in f.name:
                msg_file = str(f)
            elif 'orderbook' in f.name:
                ob_file = str(f)
        if msg_file and ob_file:
            # Quick check: read event_type column (col 1) from first rows
            df = pd.read_csv(msg_file, dtype=object, header=None, nrows=n_cond_msgs + 500, usecols=[1])
            n_rows = len(df[df.iloc[:, 0].astype('int64') <= 4])
            if n_rows >= n_cond_msgs + 10:
                samples.append((msg_file, ob_file, date_str))

    return samples


# ============================================================================
# Simulator init from raw LOBSTER data
# ============================================================================

def init_sim_from_raw(sim: OrderBook, l2_init: onp.ndarray, cond_msgs: List[dict],
                      tick_size: int, n_levels: int) -> Tuple[LobState, int, int]:
    """
    Initialize simulator by loading L2 init and replaying conditioning messages.

    Returns:
        sim_state: initialized LobState
        last_time_s: seconds of last conditioning message
        last_time_ns: nanoseconds of last conditioning message
    """
    # Reset simulator with L2 init
    # l2_init format: [askP_0, askV_0, bidP_0, bidV_0, ..., askP_n, askV_n, bidP_n, bidV_n]
    # Simulator expects: init from L2 state
    n_lev = min(n_levels, len(l2_init) // 4)
    init_state = sim.reset(l2_init, jnp.array([cond_msgs[0]['time_s'], cond_msgs[0]['time_ns']], dtype=jnp.int32))

    # Replay conditioning messages
    sim_state = init_state
    last_time_s = cond_msgs[0]['time_s']
    last_time_ns = cond_msgs[0]['time_ns']

    for msg in cond_msgs:
        # Direction: LOBSTER uses {-1, 1} internally but our raw data has {-1, 1}
        # construct_sim_msg converts side: side*2-1, so we need side in {0, 1} format
        direction = msg['direction']
        # LOBSTER raw: 1=buy, -1=sell. We need 0=buy, 1=sell for construct_sim_msg
        if direction == 1:
            side_01 = 0
        elif direction == -1:
            side_01 = 1
        else:
            side_01 = direction  # already 0 or 1

        sim_msg = construct_sim_msg(
            jnp.int32(msg['event_type']),
            jnp.int32(side_01),
            jnp.int32(msg['size']),
            jnp.int32(msg['price']),
            jnp.int32(msg['order_id']),
            jnp.int32(msg['time_s']),
            jnp.int32(msg['time_ns']),
        )
        sim_state = sim.process_order_array(sim_state, sim_msg)
        last_time_s = msg['time_s']
        last_time_ns = msg['time_ns']

    return sim_state, last_time_s, last_time_ns


# ============================================================================
# Token ↔ message bridge
# ============================================================================

def build_vocab_lookup(tokenizer, vocab_size: int = None) -> dict:
    """
    Build a reverse vocab lookup dict: token_id → decoded string.

    Dict lookup is ~50x faster than tokenizer.decode() per token.
    """
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    lookup = {}
    for i in range(vocab_size):
        try:
            lookup[i] = tokenizer.decode(i).strip()
        except Exception:
            lookup[i] = ""
    return lookup


def parse_block_tokens(tokenizer, tokens_1d, starting_time_ns: int = 0,
                       debug: bool = False, vocab_lookup: dict = None):
    """
    Parse a 1D array of RWKV tokens into a list of message dicts.

    Adapted from lobgen/evaluate.py:csv_decode and lobgen/generate_jax.py:csv_decode.
    All field values are accumulated as strings and parsed to int only at
    newline boundaries, so multi-token BPE numbers are handled correctly.

    Each message dict contains:
        time_delta_ns, event_type, order_id, size, price, direction, abs_time_ns

    Args:
        tokenizer: PreTrainedTokenizerFast
        tokens_1d: 1D int array of tokens
        starting_time_ns: cumulative absolute time in nanoseconds (for delta accumulation)
        debug: if True, print detailed parsing info
        vocab_lookup: optional pre-built {token_id: str} dict for fast decode

    Returns:
        messages: list of parsed message dicts
        final_time_ns: updated absolute time after all messages
    """
    messages = []
    cumulative_ns = starting_time_ns

    field_strings = {}  # tag_name → accumulated string of decoded tokens
    num_tags = 0
    current_tag = None
    orderbook_active = False
    n_newlines = 0
    n_ob_sections = 0
    n_broken = 0
    n_invalid = 0

    if debug:
        # Show first 50 non-pad tokens
        non_pad = [(i, int(t)) for i, t in enumerate(tokens_1d) if int(t) != PAD_TOKEN]
        print(f"    [DEBUG] Total tokens: {len(tokens_1d)}, non-pad: {len(non_pad)}")
        for pos, tok in non_pad[:80]:
            decoded = tokenizer.decode(tok).strip() if tok not in (PAD_TOKEN, NEWLINE_TOKEN) else ('PAD' if tok == PAD_TOKEN else '\\n')
            print(f"    [DEBUG]   pos={pos} tok={tok} decoded='{decoded}'")
        if len(non_pad) > 80:
            print(f"    [DEBUG]   ... ({len(non_pad) - 80} more tokens)")

    for t in tokens_1d:
        t = int(t)
        if t == PAD_TOKEN:
            continue

        # Orderbook mode: token 775 starts it, next newline ends it
        if t == ORDERBOOK_START_TOKEN:
            orderbook_active = True
            continue

        if t == NEWLINE_TOKEN:
            if orderbook_active:
                orderbook_active = False
                n_ob_sections += 1
                continue

            n_newlines += 1
            # End of message — parse accumulated field strings
            if num_tags == 6:
                try:
                    time_str = field_strings.get('time', '0').replace(',', '').strip()
                    delta_ns = int(time_str)
                    cumulative_ns += delta_ns

                    msg = {
                        'time_delta_ns': delta_ns,
                        'event_type': int(field_strings.get('event_type', '0').replace(',', '')),
                        'order_id': int(field_strings.get('order_id', '0').replace(',', '')),
                        'size': int(field_strings.get('size', '0').replace(',', '')),
                        'price': int(field_strings.get('price', '0').replace(',', '')),
                        'direction': int(field_strings.get('direction', '0').replace(',', '')),
                        'abs_time_ns': cumulative_ns,
                    }

                    INT32_MAX = 2_147_483_647
                    vals_fit = (abs(msg['price']) < INT32_MAX and
                                abs(msg['order_id']) < INT32_MAX and
                                abs(msg['size']) < INT32_MAX)
                    if (msg['event_type'] in (1, 2, 3, 4) and msg['size'] > 0
                            and msg['price'] > 0 and vals_fit):
                        messages.append(msg)
                        if debug:
                            print(f"    [DEBUG] MSG #{len(messages)}: {msg}")
                    else:
                        n_invalid += 1
                        if debug:
                            print(f"    [DEBUG] INVALID msg (validation failed): {msg}")
                except (ValueError, OverflowError) as e:
                    n_broken += 1
                    if debug:
                        print(f"    [DEBUG] BROKEN msg (parse error: {e}): fields={field_strings}")
            else:
                n_broken += 1
                if debug:
                    print(f"    [DEBUG] BROKEN msg (num_tags={num_tags}, expected 6): fields={field_strings}")

            # Reset for next message
            field_strings = {}
            num_tags = 0
            current_tag = None
            continue

        # Skip tokens while in orderbook mode
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
            # Accumulate value string for current field
            if current_tag:
                field_strings[current_tag] = field_strings.get(current_tag, '') + decoded

    if debug:
        print(f"    [DEBUG] Summary: {n_newlines} newlines, {n_ob_sections} orderbook sections, "
              f"{len(messages)} valid msgs, {n_broken} broken, {n_invalid} invalid")

    return messages, cumulative_ns


def format_aggressive_as_rwkv_text(
    time_delta_ns: int,
    event_type: int,
    order_id: int,
    size: int,
    price: int,
    direction: int,
) -> str:
    """
    Format an aggressive order as RWKV text for tokenization.

    Returns a string in the same format as training data:
    <time>,delta_ns,<event_type>,val,<order_id>,val,<size>,val,<price>,val,<direction>,val\n
    """
    return (
        f"{TIME_COL},{time_delta_ns},"
        f"{EVENT_TYPE_COL},{event_type},"
        f"{ORDER_ID_COL},{order_id},"
        f"{SIZE_COL},{size},"
        f"{PRICE_COL},{price},"
        f"{DIRECTION_COL},{direction}\n"
    )


def msg_to_sim_array(msg: dict, tick_size: int) -> jax.Array:
    """Convert parsed message dict to 8-field JAX-LOB sim message."""
    # Direction: RWKV uses {1=buy, -1=sell}, same as LOBSTER raw
    # construct_sim_msg expects side in {0=buy, 1=sell} and does side*2-1
    direction = msg['direction']
    if direction == 1:
        side_01 = 0
    elif direction == -1:
        side_01 = 1
    else:
        side_01 = direction

    abs_ns = msg['abs_time_ns']
    time_s = abs_ns // 1_000_000_000
    time_ns = abs_ns % 1_000_000_000

    return construct_sim_msg(
        jnp.int32(msg['event_type']),
        jnp.int32(side_01),
        jnp.int32(msg['size']),
        jnp.int32(msg['price']),
        jnp.int32(msg['order_id']),
        jnp.int32(time_s),
        jnp.int32(time_ns),
    )


def msg_to_decoded_14(msg: dict, sim: OrderBook, sim_state: LobState, tick_size: int) -> jax.Array:
    """Convert parsed message dict to 14-field decoded message for LOBSTER output."""
    direction = msg['direction']
    if direction == 1:
        side_01 = 0
    elif direction == -1:
        side_01 = 1
    else:
        side_01 = direction

    abs_ns = msg['abs_time_ns']
    time_s = abs_ns // 1_000_000_000
    time_ns = abs_ns % 1_000_000_000

    # Compute relative price from current mid
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
        msg['order_id'],       # order_id
        msg['event_type'],     # event_type
        side_01,               # direction {0,1}
        msg['price'],          # price_abs
        rel_price,             # price (relative)
        msg['size'],           # size
        0,                     # delta_t_s (not used)
        0,                     # delta_t_ns (not used)
        time_s,                # time_s
        time_ns,               # time_ns
        0,                     # price_ref
        0,                     # size_ref
        0,                     # time_s_ref
        0,                     # time_ns_ref
    ], dtype=jnp.int32)


# ============================================================================
# RWKV batched generation (Python loop with vmapped forward + sampler)
# ============================================================================

def make_batched_generate_fn(rwkv_forward, simple_sampler_fn, temperature: float):
    """
    Create a batched generation function using a Python for-loop.

    The scan-based approach (jax.lax.scan + jax.vmap) causes extreme XLA
    compilation times (hours) for batch_size > ~4 due to nested scan
    (RWKV's internal layer scan × 2000 generation steps) + vmap interaction.

    This approach uses a Python loop calling vmapped forward + sampler at each
    step. JIT compilation happens once for the vmapped forward (fast, ~seconds)
    and is reused for all 2000 iterations.

    Temperature is captured as a Python constant to avoid tracing issues.
    """
    # Vmapped forward: process 1 token for each sample in batch
    v_forward_1 = jax.jit(jax.vmap(
        lambda tokens, state, params: rwkv_forward(tokens, state, params, 1),
        in_axes=(0, 0, None)
    ))
    # Vmapped sampler
    v_sampler = jax.jit(jax.vmap(
        lambda key, out: simple_sampler_fn(key, out, temperature),
        in_axes=(0, 0)
    ))

    def batched_generate_block(states, outs, rng, params, max_tokens, target_msgs):
        """Generate tokens for a batch until each has target_msgs newlines.

        Args:
            states: batched RWKV state pytree, each leaf (batch_size, ...)
            outs: (batch_size, vocab_size) — last output logits
            rng: single JAX PRNGKey
            params: RWKV params (shared, not batched)
            max_tokens: maximum tokens to generate
            target_msgs: stop after this many newlines per sample

        Returns: (states, outs, rng, tokens_array)
            tokens_array: (batch_size, max_tokens) int32, PAD=0 after completion
        """
        batch_size = outs.shape[0]
        all_tokens = onp.zeros((batch_size, max_tokens), dtype=onp.int32)
        # Keep active mask and msg_counts on GPU to avoid per-step GPU↔CPU transfers
        msg_counts = jnp.zeros(batch_size, dtype=jnp.int32)
        active = jnp.ones(batch_size, dtype=jnp.bool_)

        for t in range(max_tokens):
            # Early exit check — single GPU→CPU transfer per step (scalar bool)
            if not jnp.any(active):
                break

            # Split RNG for this step
            rng, rng_ = jax.random.split(rng)
            step_keys = jax.random.split(rng_, batch_size)

            # Sample tokens from current output logits
            sampled = v_sampler(step_keys, outs)

            # Mask inactive samples to PAD (all on GPU)
            tokens_t = jnp.where(active, sampled.astype(jnp.int32), PAD_TOKEN)
            all_tokens[:, t] = onp.array(tokens_t)  # single GPU→CPU transfer for token storage

            # Forward pass: feed tokens through RWKV (batched, 1 token each)
            token_batch = tokens_t[:, None]  # (batch, 1), already on GPU
            outs_new, states_new = v_forward_1(token_batch, states, params)
            # outs_new: (batch, 1, vocab_size), states_new: batched pytree

            # Conditional update: freeze state/output for completed samples (all on GPU)
            def cond_update(old, new):
                ndim = old.ndim
                mask_shape = (active.shape[0],) + (1,) * (ndim - 1)
                return jnp.where(active.reshape(mask_shape), new, old)

            states = jax.tree_util.tree_map(cond_update, states, states_new)
            outs = jnp.where(active[:, None], outs_new[:, 0], outs)

            # Count newlines (all on GPU)
            is_newline = (tokens_t == NEWLINE_TOKEN)
            msg_counts = msg_counts + is_newline.astype(jnp.int32) * active.astype(jnp.int32)
            active = msg_counts < target_msgs

        return states, outs, rng, jnp.array(all_tokens)

    return batched_generate_block


# ============================================================================
# Aggressive order creation
# ============================================================================

SIM_CONFIG = JAXLOB_Configuration()


def update_oid(
    msg: jax.Array,
    rng: jax.Array,
    sim: OrderBook,
    sim_state: LobState,
) -> Tuple[jax.Array, jax.Array]:
    """Assign correct order ID for cancel/execution messages using JAX-LOB state."""
    def _leave_oid(msg, rng, *args):
        return msg, rng

    def _get_random_cancel_msg(msg, rng, sim, sim_state):
        side = msg[1]
        side_array = jax.lax.cond(
            side == 1,
            lambda a, b: b,
            lambda a, b: a,
            sim_state.asks, sim_state.bids
        )
        rng, _rng = jax.random.split(rng)
        msg_dict = {
            "quantity": msg[2],
            "price": msg[3],
        }
        idx = job.get_random_id_match(SIM_CONFIG, _rng, side_array, msg_dict)
        cancelled_oid = side_array[idx, 2]
        msg = msg.at[5].set(cancelled_oid)
        return msg, rng

    def _get_oid_from_active(msg, rng, sim, sim_state):
        side = msg[1]
        oid = jax.lax.cond(
            side == 1,
            _get_top_bid_order_oid,
            _get_top_ask_order_oid,
            sim_state
        )
        return msg.at[5].set(oid), rng

    def _get_top_bid_order_oid(sim_state):
        idx = job._get_top_bid_order_idx(SIM_CONFIG, sim_state.bids).squeeze()
        return sim_state.bids[idx, 2]

    def _get_top_ask_order_oid(sim_state):
        idx = job._get_top_ask_order_idx(SIM_CONFIG, sim_state.asks).squeeze()
        return sim_state.asks[idx, 2]

    msg, rng = jax.lax.switch(
        (msg[0] == 3) + 2 * (msg[0] == 4),
        (_leave_oid, _get_random_cancel_msg, _get_oid_from_active),
        msg, rng, sim, sim_state
    )
    return msg, rng


@jax.jit
def create_aggressive_order(
    sim: OrderBook,
    sim_state: LobState,
    last_time_s: jax.Array,
    last_time_ns: jax.Array,
    tick_size: int,
    event_type: int,
    direction: int,
    order_volume: int,
) -> Tuple[jax.Array, jax.Array]:
    """
    Create an aggressive market order based on current JAX-LOB book state.

    Returns:
        sim_msg: Message for JAX-LOB simulator (8 fields)
        msg_decoded: Decoded message for storage (14 fields)
    """
    price = jax.lax.cond(
        direction == 0,
        lambda: sim.get_best_ask(sim_state),
        lambda: sim.get_best_bid(sim_state)
    )

    # Returns (best_ask=[price,vol], best_bid=[price,vol])
    best_ask_pv, best_bid_pv = sim.get_best_bid_and_ask_inclQuants(sim_state)
    avail = jax.lax.cond(
        direction == 0,
        lambda: best_ask_pv[1],   # ask volume for buy
        lambda: best_bid_pv[1],   # bid volume for sell
    ).astype(jnp.int32)

    quantity = jnp.minimum(jnp.int32(order_volume), avail)

    time_s = last_time_s.astype(jnp.int32)
    time_ns = (last_time_ns + 1).astype(jnp.int32)

    sim_msg = construct_sim_msg(
        event_type, direction, quantity, price,
        AGGRESSIVE_ORDER_ID, time_s, time_ns,
    )

    mid_price = (sim.get_best_ask(sim_state) + sim.get_best_bid(sim_state)) // 2
    mid_price = (mid_price // tick_size) * tick_size
    rel_price = (price - mid_price) // tick_size

    msg_decoded = jnp.array([
        AGGRESSIVE_ORDER_ID,
        event_type,
        direction,
        price,
        rel_price,
        quantity,
        0,
        1,
        time_s,
        time_ns,
        0, 0, 0, 0,
    ], dtype=jnp.int32)

    return sim_msg, msg_decoded


# ============================================================================
# Main scenario function
# ============================================================================

def run_rwkv_scenario(cfg: Dict[str, Any], save_folder: Path):
    """
    Main function for RWKV aggressive scenario.

    Flow:
    1. Load RWKV model and tokenizer
    2. Enumerate raw data samples
    3. For each batch:
       a. Load & tokenize conditioning data
       b. Feed through RWKV (vmapped process_long_seq)
       c. Initialize JAX-LOB simulators
       d. Generate + inject blocks
       e. Post-process and save
    """
    # Unpack config
    lobgen_dir = cfg['lobgen_dir']
    model_choice = cfg['model_choice']
    rwkv_type = cfg['rwkv_type']
    ckpt_path = cfg['ckpt_path']
    tokenizer_file = cfg['tokenizer_file']
    temperature = cfg.get('temperature', 1.0)
    raw_data_dir = cfg['raw_data_dir']
    stock = cfg['stock']
    tick_size = cfg['tick_size']
    n_cond_msgs = cfg['n_cond_msgs']
    n_gen_msgs = cfg['n_gen_msgs']
    max_tokens_per_block = cfg['max_tokens_per_block']
    num_insertions = cfg['num_insertions']
    num_coolings = cfg['num_coolings']
    event_type = cfg['event_type']
    direction = cfg['direction']
    order_volume = cfg['order_volume']
    n_samples = cfg['n_samples']
    batch_size = cfg['batch_size']
    rng_seed = cfg['rng_seed']
    n_levels = cfg['n_levels']
    process_long_seq_padding = cfg.get('process_long_seq_padding', 128)
    max_cond_tokens = cfg.get('max_cond_tokens', 15000)

    total_blocks = num_insertions + num_coolings

    # === 1. Load RWKV model ===
    print(f"Adding lobgen to path: {lobgen_dir}")
    sys.path.insert(0, lobgen_dir)

    from jax_rwkv.auto import get_model, load as rwkv_load
    from jax_rwkv.utils import simple_sampler
    from transformers import PreTrainedTokenizerFast

    print(f"Loading RWKV model: {model_choice} (type={rwkv_type})")
    RWKV, _params, _ = get_model(model_choice, rwkv_type=rwkv_type)
    print(f"Loading checkpoint from: {ckpt_path}")
    params = rwkv_load(os.path.join(ckpt_path, "params.model"))

    forward_jit = jax.jit(RWKV.forward)

    tokenizer_path = os.path.join(lobgen_dir, tokenizer_file)
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        clean_up_tokenization_spaces=False
    )

    # Build reverse vocab lookup for fast token decoding (~50x faster than tokenizer.decode())
    print("Building vocab lookup table...")
    vocab_lookup = build_vocab_lookup(tokenizer)
    print(f"Vocab lookup built: {len(vocab_lookup)} entries")

    # Sanity check: 1-token forward pass to verify checkpoint isn't NaN
    test_state = RWKV.default_state(params)
    test_out, _ = RWKV.forward(jnp.array([1]), test_state, params, 1)
    test_out_np = onp.array(test_out[0]).astype(onp.float32)
    has_nan = onp.any(onp.isnan(test_out_np))
    print(f"Sanity check: 1-token forward → NaN={has_nan}")
    if has_nan:
        raise RuntimeError(
            "Checkpoint params produce NaN on forward pass. "
            "The checkpoint may be corrupted (training diverged to NaN). "
            "Try a different checkpoint step."
        )

    # Build vmapped process_long_seq (from lobgen/evaluate.py)
    # Key fix: clamp cur_len to max(1, ...) to avoid passing negative length
    # to RWKV.forward, which uses length for indexing (x[length-1]) and
    # state selection (jax.lax.select(t < length, ...)). Negative lengths
    # cause NaN in XLA even though results are conditionally discarded.
    def process_long_seq(tokens, state, length, padding=process_long_seq_padding):
        full_instruction_length = tokens.shape[-1]
        instruction_len = length
        x = (jnp.zeros_like(params['emb']['weight'][:, 0]), state)

        def inner_loop(x, i):
            (true_out, state) = x
            cur_len = jnp.minimum(padding, instruction_len - i)
            # Clamp to 1 so RWKV.forward always gets a valid length.
            # The jax.lax.cond below still uses the original cur_len to
            # decide whether to keep the results.
            safe_len = jnp.maximum(cur_len, 1)
            out, new_state = forward_jit(
                jax.lax.dynamic_slice_in_dim(tokens, i, padding),
                state, params, safe_len
            )
            state = jax.lax.cond(cur_len <= 0, lambda: state, lambda: new_state)
            true_out = jax.lax.cond(cur_len <= 0, lambda: true_out, lambda: out[cur_len - 1])
            return (true_out, state), 0

        (true_out, state), _ = jax.lax.scan(
            inner_loop, x, jnp.arange(0, full_instruction_length, padding)
        )
        return true_out, state

    v_process_long_seq = jax.jit(jax.vmap(process_long_seq))

    # Build batched generation function (Python loop with vmapped forward + sampler)
    # Replaces the scan-based approach which caused hours of XLA compilation for batch > ~4
    batched_generate = make_batched_generate_fn(RWKV.forward, simple_sampler, temperature)

    # Single-sample token feeding (for aggressive order injection)
    def feed_tokens_single(tokens, state, length, padding=process_long_seq_padding):
        """Feed a short token sequence through RWKV, returning final out and updated state."""
        # Pad to multiple of padding
        n_tokens = tokens.shape[0]
        pad_len = ((n_tokens + padding - 1) // padding) * padding
        padded = jnp.zeros(pad_len, dtype=jnp.int32)
        padded = padded.at[:n_tokens].set(tokens)

        x = (jnp.zeros_like(params['emb']['weight'][:, 0]), state)

        def inner_loop(x, i):
            (true_out, st) = x
            cur_len = jnp.minimum(padding, length - i)
            safe_len = jnp.maximum(cur_len, 1)
            out, new_st = forward_jit(
                jax.lax.dynamic_slice_in_dim(padded, i, padding),
                st, params, safe_len
            )
            st = jax.lax.cond(cur_len <= 0, lambda: st, lambda: new_st)
            true_out = jax.lax.cond(cur_len <= 0, lambda: true_out, lambda: out[cur_len - 1])
            return (true_out, st), 0

        (true_out, state), _ = jax.lax.scan(
            inner_loop, x, jnp.arange(0, pad_len, padding)
        )
        return true_out, state

    # === 2. Enumerate samples ===
    print(f"Enumerating samples from: {raw_data_dir}")
    day_samples = enumerate_samples(raw_data_dir, n_cond_msgs)
    print(f"Found {len(day_samples)} days with sufficient data")

    # Create flat sample index: each day contributes 1 sample for now
    # (could be extended to multiple windows per day)
    n_available = len(day_samples)
    assert n_available > 0, f"No valid day files found in {raw_data_dir}"

    # Sample indices with RNG
    rng = jax.random.key(rng_seed)
    rng, _ = jax.random.split(rng)
    rng, rng_ = jax.random.split(rng)

    assert n_samples % batch_size == 0, \
        f'n_samples ({n_samples}) must be divisible by batch_size ({batch_size})'

    # Sample with replacement if n_samples > n_available
    replace = n_samples > n_available
    sample_indices = jax.random.choice(
        rng_,
        jnp.arange(n_available, dtype=jnp.int32),
        shape=(n_samples,),
        replace=replace,
    )
    sample_indices = onp.array(sample_indices)

    # Reshape into batches
    n_batches = n_samples // batch_size
    sample_batches = sample_indices.reshape(n_batches, batch_size)

    # Create output folders
    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    # Initialize JAX-LOB simulator
    sim = OrderBook(cfg=JAXLOB_Configuration(
        cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value
    ))

    # Build aggressive order insertion schedule
    aggressive_positions = []
    pos = 0
    for block in range(total_blocks):
        pos += n_gen_msgs
        if block < num_insertions:
            aggressive_positions.append(pos)
            pos += 1
    print(f"Aggressive order positions (0-indexed): {aggressive_positions}")
    print(f"Total messages per sample: {pos}")

    # RNG for generation
    rng, rng_ = jax.random.split(rng)

    # Day data cache (avoids re-reading same CSV files for different samples from same day)
    day_cache = DayDataCache(n_cond_msgs, tokenizer)

    # === 3. Batch loop ===
    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        batch_indices = sample_batches[batch_idx]
        print(f'\n=== BATCH {batch_idx}: day indices {batch_indices.tolist()} ===')

        # a. Load batch data (cached per day)
        batch_cond_tokens = []
        batch_cond_msgs = []
        batch_l2_inits = []
        batch_dates = []
        batch_first_times_ns = []

        for idx in batch_indices:
            msg_file, ob_file, date_str = day_samples[idx]
            cond_tokens, cond_msgs, l2_init, date_str, first_time_ns = day_cache.load(
                msg_file, ob_file, date_str
            )
            batch_cond_tokens.append(cond_tokens)
            batch_cond_msgs.append(cond_msgs)
            batch_l2_inits.append(l2_init)
            batch_dates.append(date_str)
            batch_first_times_ns.append(first_time_ns)

        # Pad conditioning tokens to dynamic length (per-batch, rounded up to padding boundary)
        # This eliminates ~47% empty conditioning chunks vs fixed max_cond_tokens=15000
        cond_lengths = []
        for ct in batch_cond_tokens:
            length = min(len(ct), max_cond_tokens)
            cond_lengths.append(length)

        # Dynamic padding: round up to next multiple of process_long_seq_padding
        batch_max_len = max(cond_lengths)
        dynamic_cond_len = ((batch_max_len + process_long_seq_padding - 1) // process_long_seq_padding) * process_long_seq_padding
        # Clamp to configured max_cond_tokens
        dynamic_cond_len = min(dynamic_cond_len, max_cond_tokens)

        padded_cond = []
        for ct, length in zip(batch_cond_tokens, cond_lengths):
            padded = onp.zeros(dynamic_cond_len, dtype=onp.int32)
            padded[:length] = ct[:length]
            padded_cond.append(padded)

        cond_batch = jnp.array(onp.stack(padded_cond))  # (batch_size, dynamic_cond_len)
        lengths_batch = jnp.array(cond_lengths, dtype=jnp.int32)  # (batch_size,)

        print(f"  Conditioning token lengths: min={min(cond_lengths)}, max={max(cond_lengths)}, "
              f"padded to {dynamic_cond_len} (config max: {max_cond_tokens})")

        # b. RWKV conditioning (vmapped)
        print("  Running RWKV conditioning...")
        init_state = RWKV.default_state(params)
        states = jnp.repeat(init_state[None], batch_size, axis=0)
        outs, states = v_process_long_seq(cond_batch, states, lengths_batch)
        print(f"  Conditioning done. Output shape: {outs.shape}")

        # c. Initialize simulators (sequential per sample)
        print("  Initializing simulators...")
        sim_states = []
        last_times_s = []
        last_times_ns = []
        for i in range(batch_size):
            sim_state_i, lt_s, lt_ns = init_sim_from_raw(
                sim, batch_l2_inits[i], batch_cond_msgs[i], tick_size, n_levels
            )
            sim_states.append(sim_state_i)
            last_times_s.append(lt_s)
            last_times_ns.append(lt_ns)

        # Save conditioning data
        for i in range(batch_size):
            sample_idx = batch_idx * batch_size + i
            date = batch_dates[i]
            # Save conditioning messages as LOBSTER CSV
            cond_rows = []
            for msg in batch_cond_msgs[i]:
                abs_ns = msg['time_s'] * 1_000_000_000 + msg['time_ns']
                time_str = convert_ns_to_lobster_time(abs_ns)
                d = msg['direction']
                cond_rows.append([
                    time_str,
                    msg['event_type'],
                    msg['order_id'],
                    msg['size'],
                    msg['price'],
                    d,
                ])
            cond_df = pd.DataFrame(cond_rows)
            cond_df.to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_message_real_id_{sample_idx}.csv',
                index=False, header=False
            )
            # Save initial L2 orderbook
            ob_df = pd.DataFrame([batch_l2_inits[i]])
            ob_df.to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_orderbook_real_id_{sample_idx}.csv',
                index=False, header=False
            )

        # d. RNG for generation (single key, split inside Python loop)
        rng, gen_rng = jax.random.split(rng)

        # e. Generation + insertion blocks
        # Track cumulative absolute times per sample for message parsing
        cumulative_times_ns = []
        for i in range(batch_size):
            # Last conditioning message absolute time
            last_msg = batch_cond_msgs[i][-1]
            cumulative_times_ns.append(last_msg['time_s'] * 1_000_000_000 + last_msg['time_ns'])

        all_block_data = []  # list of (block_tokens_np, is_insertion_block)
        # Cache parsed messages per block per sample to avoid double parsing (Bottleneck #4)
        # parsed_messages_cache[block_idx][sample_idx] = (messages, final_cum_ns)
        parsed_messages_cache = []

        for block in range(total_blocks):
            is_insertion = block < num_insertions
            block_type = "INSERTION" if is_insertion else "COOLING"
            block_t0 = time.time()
            print(f"  Block {block}/{total_blocks} ({block_type}): generating {n_gen_msgs} messages...",
                  end="", flush=True)

            # GENERATE: batched Python loop with vmapped forward
            states, outs, gen_rng, block_tokens = batched_generate(
                states, outs, gen_rng, params,
                max_tokens_per_block, n_gen_msgs
            )
            block_tokens_np = onp.array(block_tokens)  # (batch_size, max_tokens_per_block)
            non_pad = onp.count_nonzero(block_tokens_np)
            print(f" done in {time.time() - block_t0:.1f}s ({non_pad} non-pad tokens)", flush=True)

            # Parse generated messages → update simulators (sequential per sample)
            block_parsed = []  # per-sample cache for this block
            for i in range(batch_size):
                messages, new_cum_ns = parse_block_tokens(
                    tokenizer, block_tokens_np[i], cumulative_times_ns[i],
                    vocab_lookup=vocab_lookup
                )
                block_parsed.append((messages, new_cum_ns))
                cumulative_times_ns[i] = new_cum_ns

                # Apply to simulator
                for msg in messages:
                    try:
                        sim_msg = msg_to_sim_array(msg, tick_size)
                        # Update OID for cancel/execution
                        sim_msg, _ = update_oid(
                            sim_msg, jax.random.key(0), sim, sim_states[i]
                        )
                        sim_states[i] = sim.process_order_array(sim_states[i], sim_msg)
                        last_times_s[i] = msg['abs_time_ns'] // 1_000_000_000
                        last_times_ns[i] = msg['abs_time_ns'] % 1_000_000_000
                    except (OverflowError, ValueError) as e:
                        pass  # skip malformed messages that slip through parser

            all_block_data.append((block_tokens_np, is_insertion))
            parsed_messages_cache.append(block_parsed)

            # INJECTION (first num_insertions blocks only)
            if is_insertion:
                print(f"  Injecting aggressive orders...")
                # Create and apply aggressive orders per sample
                aggr_texts = []
                for i in range(batch_size):
                    # Create aggressive order from simulator state
                    sim_msg, msg_decoded = create_aggressive_order(
                        sim, sim_states[i],
                        jnp.int32(last_times_s[i]),
                        jnp.int32(last_times_ns[i]),
                        tick_size, event_type, direction, order_volume,
                    )

                    # Apply to simulator
                    sim_states[i] = sim.process_order_array(sim_states[i], sim_msg)

                    # Update time tracking
                    aggr_time_s = int(msg_decoded[TIMEs_i])
                    aggr_time_ns = int(msg_decoded[TIMEns_i])
                    last_times_s[i] = aggr_time_s
                    last_times_ns[i] = aggr_time_ns
                    cumulative_times_ns[i] = aggr_time_s * 1_000_000_000 + aggr_time_ns

                    # Format as RWKV text for model feeding
                    # Delta = 1ns (minimal time increment)
                    aggr_direction_raw = 1 if direction == 0 else -1
                    aggr_text = format_aggressive_as_rwkv_text(
                        time_delta_ns=1,
                        event_type=event_type,
                        order_id=AGGRESSIVE_ORDER_ID,
                        size=int(msg_decoded[SIZE_i]),
                        price=int(msg_decoded[PRICE_ABS_i]),
                        direction=aggr_direction_raw,
                    )
                    aggr_texts.append(aggr_text)

                # Tokenize aggressive orders
                aggr_tokenized = tokenizer(aggr_texts, return_attention_mask=False, return_tensors="np")["input_ids"]
                # Pad to common length for vmapped feeding
                max_aggr_len = max(len(t) for t in aggr_tokenized)
                # Round up to padding boundary
                padded_aggr_len = ((max_aggr_len + process_long_seq_padding - 1) // process_long_seq_padding) * process_long_seq_padding

                aggr_batch = onp.zeros((batch_size, padded_aggr_len), dtype=onp.int32)
                aggr_lengths = onp.zeros(batch_size, dtype=onp.int32)
                for i in range(batch_size):
                    toks = aggr_tokenized[i]
                    aggr_batch[i, :len(toks)] = toks
                    aggr_lengths[i] = len(toks)

                # Feed through RWKV (vmapped)
                outs, states = v_process_long_seq(
                    jnp.array(aggr_batch),
                    states,
                    jnp.array(aggr_lengths, dtype=jnp.int32)
                )

        # f. Post-process: replay ALL tokens through simulator → LOBSTER output
        print("  Post-processing and saving...")
        for i in range(batch_size):
            sample_idx = batch_idx * batch_size + i
            date = batch_dates[i]

            # Re-initialize simulator for clean replay
            sim_state_replay, _, _ = init_sim_from_raw(
                sim, batch_l2_inits[i], batch_cond_msgs[i], tick_size, n_levels
            )

            # Replay all blocks, collecting decoded messages and L2 states
            all_msgs_decoded = []
            all_l2_books = []
            replay_last_s = batch_cond_msgs[i][-1]['time_s']
            replay_last_ns = batch_cond_msgs[i][-1]['time_ns']

            for block_idx_inner, (block_tokens_np, is_insertion) in enumerate(all_block_data):
                # Use cached parsed messages instead of re-parsing (Bottleneck #4 fix)
                messages, _ = parsed_messages_cache[block_idx_inner][i]

                for msg in messages:
                    try:
                        sim_msg = msg_to_sim_array(msg, tick_size)
                        sim_msg, _ = update_oid(sim_msg, jax.random.key(0), sim, sim_state_replay)
                        sim_state_replay = sim.process_order_array(sim_state_replay, sim_msg)

                        # Build 14-field decoded message
                        msg_decoded = msg_to_decoded_14(msg, sim, sim_state_replay, tick_size)
                        l2_book = sim.get_L2_state(sim_state_replay, n_levels)

                        all_msgs_decoded.append(msg_decoded)
                        all_l2_books.append(l2_book)

                        replay_last_s = msg['abs_time_ns'] // 1_000_000_000
                        replay_last_ns = msg['abs_time_ns'] % 1_000_000_000
                    except (OverflowError, ValueError):
                        pass  # skip malformed messages

                # Aggressive order injection
                if is_insertion:
                    sim_msg_aggr, msg_decoded_aggr = create_aggressive_order(
                        sim, sim_state_replay,
                        jnp.int32(replay_last_s),
                        jnp.int32(replay_last_ns),
                        tick_size, event_type, direction, order_volume,
                    )
                    sim_state_replay = sim.process_order_array(sim_state_replay, sim_msg_aggr)
                    l2_after = sim.get_L2_state(sim_state_replay, n_levels)

                    all_msgs_decoded.append(msg_decoded_aggr)
                    all_l2_books.append(l2_after)

                    replay_last_s = int(msg_decoded_aggr[TIMEs_i])
                    replay_last_ns = int(msg_decoded_aggr[TIMEns_i])

            if len(all_msgs_decoded) == 0:
                print(f"  WARNING: Sample {sample_idx} produced 0 valid messages, skipping")
                continue

            # Stack and save
            all_msgs_arr = jnp.stack(all_msgs_decoded, axis=0)
            all_books_arr = jnp.stack(all_l2_books, axis=0)

            msg_to_lobster_format(all_msgs_arr).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_message_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )
            book_to_lobster_format(all_books_arr).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_orderbook_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )

            if batch_idx == 0 and i == 0:
                print(f"  First sample: {all_msgs_arr.shape[0]} total messages "
                      f"({total_blocks} blocks x ~{n_gen_msgs} + {num_insertions} aggressive)")

        # Save aggressive indices once
        if batch_idx == 0:
            aggressive_indices = onp.array(aggressive_positions)
            onp.savetxt(save_folder / 'aggressive_indices.csv', aggressive_indices, fmt='%d')

        rng, _ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="RWKV Aggressive Scenario with Continuous State")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='lob_impact/5.aggressive_scenario_rwkv_config.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument('--n_gen_msgs', type=int, default=None, help='Override n_gen_msgs from config')
    parser.add_argument('--direction', type=int, default=None, choices=[0, 1],
                        help='Override direction (0=buy, 1=sell)')
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
    if args.direction is not None:
        cfg['direction'] = args.direction

    print(f"Configuration: {cfg}")

    # Create experiment folder
    save_folder = create_experiment_folder(cfg['save_dir'])
    print(f"Experiment folder: {save_folder}")

    # Set up logging
    logger = setup_logging(save_folder)
    print(f"\n{'='*60}")
    print(f"RWKV Aggressive Scenario Experiment")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Configuration: {cfg}")
    print(f"Experiment folder: {save_folder}")

    # Save config to experiment folder
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
