#!/usr/bin/env python
"""
CST Model Aggressive Scenario: Parametric LOB Generation with Aggressive Order Injection

Uses the Continuous Stoikov-Talreja (CST) parametric model as a baseline alternative
to the S5 neural model. CST generates LOB messages by sampling from exponential
inter-arrival times and categorical event distributions (power-law LO placement,
Poisson MO arrivals, depth-dependent cancellations).

Unlike S5, CST has no hidden state — the book state IS the full model state.

Output format matches S5 scenario (1.aggressive_scenario_s5.py) so existing
analysis notebooks (100-110) work with both.
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import jax
import jax.numpy as jnp
import numpy as onp
from tqdm import tqdm

# Add parent folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_folder_path)

# Add AlphaTrade submodule
if os.path.exists('/AlphaTrade'):
    sys.path.insert(0, '/AlphaTrade')
else:
    sys.path.insert(0, os.path.join(parent_folder_path, 'Alphatrade'))

# Add CST model path (alias as stoikov to avoid conflict with jaxob_constants alias)
# NOTE: cst.py / param_estimation.py live in ../lob_bench/cst_model/ and were NOT transferred
# with this submodule copy. Vendor them from the original lob_bench/cst_model/ to run CST. See README "Known gaps".
sys.path.insert(0, os.path.join(parent_folder_path, 'lob_bench', 'cst_model'))
import cst as stoikov
from param_estimation import load_params

from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob.inference_no_errcorr import (
    get_sims_vmap, get_dataset, msg_to_jnp, msgs_to_jnp,
    msg_to_lobster_format, book_to_lobster_format, construct_sim_msg,
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
PRICE_REF_i = 10
SIZE_REF_i = 11
TIMEs_REF_i = 12
TIMEns_REF_i = 13

START_OID = 100  # unused, kept for reference


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
# Adapted from lob_bench/cst_model/lobster_conversion.py
# Inlined because the original imports from jaxlob (pip package) while we use
# the Alphatrade submodule (different import paths).
# ============================================================================

SIM_CONFIG = JAXLOB_Configuration()


def msg_to_jaxlob(msg: stoikov.Message, oid: jax.Array) -> jax.Array:
    """Convert CST Message to JAX-LOB simulator message format (8 fields)."""
    return jnp.array([
        msg.event_type,
        msg.direction,
        jnp.abs(msg.size),
        msg.price,
        0,           # trader ID (not used)
        oid,         # order ID
        msg.time,    # whole seconds
        msg.time % 1 * 1e9,  # fractional seconds to nanoseconds
    ], dtype=jnp.int32)


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
        # 0 -> leave, 1 -> random cancel, 2 -> get executable oid from active
        (msg[0] == 3) + 2 * (msg[0] == 4),
        (_leave_oid, _get_random_cancel_msg, _get_oid_from_active),
        msg, rng, sim, sim_state
    )
    return msg, rng


# ============================================================================
# CST-specific functions
# ============================================================================

def apply_aggressive_to_cst_book(
    book: stoikov.Book,
    direction: int,
    quantity: int,
    params: stoikov.CSTParams,
) -> stoikov.Book:
    """
    Update CST Book state after aggressive order injection.

    For a buy (direction=0), removes liquidity from the ask side.
    For a sell (direction=1), removes liquidity from the bid side.
    Uses the same depth calculation as CST's MO logic.
    """
    # bid_side: CST convention — 0=ask side (buy hits asks), 1=bid side (sell hits bids)
    bid_side = direction

    # depth_i: same formula as CST _apply_event for MOs
    depth_i = (book.best_ask - book.best_bid) // params.tick_size - 1

    # Negative qty removes liquidity (same as MO in CST)
    book = stoikov._change_vol(book, depth_i, -quantity, bid_side, params)
    return book


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
    order_id: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Create an aggressive market order based on current JAX-LOB book state.
    order_id: descending counter (n_msg_todo), same convention as S5/historic/heuristic.

    Returns:
        sim_msg: Message for JAX-LOB simulator (8 fields)
        msg_decoded: Decoded message for storage (14 fields)
    """
    # Get best price on the side we will aggress
    price = jax.lax.cond(
        direction == 0,
        lambda: sim.get_best_ask(sim_state),
        lambda: sim.get_best_bid(sim_state)
    )

    # Get available volume at best level
    # get_best_bid_and_ask_inclQuants returns (best_ask=[p,v], best_bid=[p,v])
    best_ask_pv, best_bid_pv = sim.get_best_bid_and_ask_inclQuants(sim_state)
    avail = jax.lax.cond(
        direction == 0,
        lambda: best_ask_pv[1],  # ask volume for buy (we hit asks)
        lambda: best_bid_pv[1],  # bid volume for sell (we hit bids)
    ).astype(jnp.int32)

    # Cap order size at available volume
    quantity = jnp.minimum(jnp.int32(order_volume), avail)

    # Use time from last message + small increment
    time_s = last_time_s.astype(jnp.int32)
    time_ns = (last_time_ns + 1).astype(jnp.int32)

    # Build simulator message (8 fields)
    sim_msg = construct_sim_msg(
        event_type, direction, quantity, price,
        order_id, time_s, time_ns,
    )

    # Build decoded message for storage (14 fields)
    mid_price = (sim.get_best_ask(sim_state) + sim.get_best_bid(sim_state)) // 2
    mid_price = (mid_price // tick_size) * tick_size
    rel_price = (price - mid_price) // tick_size

    msg_decoded = jnp.array([
        order_id,             # order_id (descending counter)
        event_type,           # event_type
        direction,            # direction
        price,                # price_abs
        rel_price,            # price (relative)
        quantity,             # size
        0,                    # delta_t_s
        1,                    # delta_t_ns
        time_s,               # time_s
        time_ns,              # time_ns
        0,                    # price_ref
        0,                    # size_ref
        0,                    # time_s_ref
        0,                    # time_ns_ref
    ], dtype=jnp.int32)

    return sim_msg, msg_decoded


def make_cst_scan_fn(n_levels: int, sim: OrderBook):
    """
    Returns a JAX-scannable step function for one CST generation block.

    Carry: (sim_state, cst_book, n_msg_todo, base_rates, params, rng)
    Output per step: (l2_book_flat, lobster_msg_decoded)

    n_msg_todo is a descending counter used as order_id (same as S5/historic/heuristic).
    """
    def _step_fn(carry, _):
        sim_state, cst_book, n_msg_todo, base_rates, params, rng = carry

        # order_id = n_msg_todo (descending, same as S5)
        order_id = n_msg_todo

        # 1. CST step: generate one message
        cst_book, message, rng = stoikov.step_book(cst_book, base_rates, params, rng)

        # 2. Convert to JAX-LOB format
        msg_jaxlob = msg_to_jaxlob(message, order_id)

        # 3. Assign correct OID for cancel/execution messages
        msg_jaxlob, rng = update_oid(msg_jaxlob, rng, sim, sim_state)

        # 4. Apply to JAX-LOB simulator
        sim_state = sim.process_order_array(sim_state, msg_jaxlob)

        # 5. Extract L2 state
        l2_book = sim.get_L2_state(sim_state, n_levels)

        # 6. Build decoded message (14 fields) for LOBSTER output
        # Time: CST uses fractional seconds, convert to (s, ns)
        time_s = jnp.floor(message.time).astype(jnp.int32)
        time_ns = ((message.time - time_s) * 1e9).astype(jnp.int32)

        # Compute mid price and relative price
        best_ask = sim.get_best_ask(sim_state)
        best_bid = sim.get_best_bid(sim_state)
        mid_price = (best_ask + best_bid) // 2
        tick_size = params.tick_size
        mid_price = (mid_price // tick_size) * tick_size
        rel_price = jax.lax.cond(
            mid_price > 0,
            lambda: (message.price - mid_price) // tick_size,
            lambda: jnp.int32(0),
        )

        # Convert CST direction {-1, 1} to {0, 1}
        direction_01 = ((message.direction + 1) // 2).astype(jnp.int32)

        msg_decoded = jnp.array([
            order_id,                              # order_id (descending counter)
            message.event_type.astype(jnp.int32),  # event_type
            direction_01,                           # direction {0,1}
            message.price.astype(jnp.int32),        # price_abs
            rel_price,                              # price (relative)
            jnp.abs(message.size).astype(jnp.int32), # size
            0,                                      # delta_t_s (not used)
            0,                                      # delta_t_ns (not used)
            time_s,                                 # time_s
            time_ns,                                # time_ns
            0,                                      # price_ref
            0,                                      # size_ref
            0,                                      # time_s_ref
            0,                                      # time_ns_ref
        ], dtype=jnp.int32)

        n_msg_todo = n_msg_todo - 1
        new_carry = (sim_state, cst_book, n_msg_todo, base_rates, params, rng)
        return new_carry, (l2_book, msg_decoded)

    return _step_fn


# ============================================================================
# Main scenario function
# ============================================================================

def run_cst_scenario(cfg: Dict[str, Any], save_folder: Path,
                     worker_id: int = 0, num_workers: int = 1):
    """
    Main function for CST aggressive scenario.

    Flow:
    1. Load CST params and compute base rates
    2. Load dataset and sample indices
    3. Split batches across workers (deterministic — same RNG everywhere)
    4. Initialize JAX-LOB simulator with conditioning messages
    5. For each sample: run CST generation blocks with aggressive order injections
    6. Save results in LOBSTER CSV format
    """
    # Unpack config
    n_gen_msgs = cfg['n_gen_msgs']
    num_insertions = cfg['num_insertions']
    num_coolings = cfg['num_coolings']
    n_cond_msgs = cfg['n_cond_msgs']
    n_samples = cfg['n_samples']
    batch_size = cfg['batch_size']
    rng_seed = cfg['rng_seed']
    stock = cfg['stock']
    data_dir = cfg['data_dir']
    tick_size = cfg['tick_size']
    n_levels = cfg['n_levels']
    test_split = cfg.get('test_split', 0)
    event_type = cfg['event_type']
    direction = cfg['direction']
    n_eval_msgs_dataset = cfg.get('n_eval_msgs_dataset', 500)
    order_volume = cfg['order_volume']
    params_file = cfg['params_file']

    total_blocks = num_insertions + num_coolings

    # 1. Load CST params
    print(f"Loading CST params from {params_file}")
    params_dict = load_params(params_file)
    cst_params, lo_lambda, co_theta = stoikov.init_params(params_dict)
    print(f"CST params: {cst_params}")
    print(f"lo_lambda shape: {lo_lambda.shape}, co_theta shape: {co_theta.shape}")

    # Compute base rates
    base_rates = stoikov.get_event_base_rates(
        cst_params,
        cancel_rates=co_theta,
        lo_rates=lo_lambda,
    )
    print(f"Base rates shape: {base_rates.shape}")

    # 2. Initialize
    rng = jax.random.key(rng_seed)

    # Initialize JAX-LOB simulator
    sim = OrderBook(cfg=JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))

    # Per-day mode: restrict to specific day
    day_index = cfg.get('day_index', None)
    day_indeces = [day_index] if day_index is not None else None

    # Load dataset
    print(f"Loading dataset from {data_dir} (day_indeces={day_indeces})")
    ds = get_dataset(
        data_dir,
        n_cond_msgs,
        n_eval_msgs_dataset,
        test_split=test_split,
        day_indeces=day_indeces,
    )
    print(f"Dataset length: {len(ds)}")

    # Create output folders
    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    # Sample indices (same RNG pattern as other scenarios for reproducibility)
    assert n_samples % batch_size == 0, f'n_samples ({n_samples}) must be divisible by batch_size ({batch_size})'
    rng, _ = jax.random.split(rng)
    rng, rng_ = jax.random.split(rng)
    all_sample_i = jax.random.choice(
        rng_,
        jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size),
        replace=(n_samples > len(ds)),  # >windows -> sample with replacement (reach 2048/side)
    ).tolist()

    # Split batches across workers (deterministic round-robin)
    sample_i = all_sample_i[worker_id::num_workers]
    n_total_batches = len(all_sample_i)
    n_my_batches = len(sample_i)
    print(f"Worker {worker_id}: {n_my_batches}/{n_total_batches} batches "
          f"({n_my_batches * batch_size} samples)")

    # Build insertion schedule (0-indexed positions of aggressive orders in output)
    # Each block produces n_gen_msgs messages. After each insertion block, one aggressive order is added.
    aggressive_positions = []
    pos = 0
    for block in range(total_blocks):
        pos += n_gen_msgs  # messages from this block
        if block < num_insertions:
            aggressive_positions.append(pos)
            pos += 1  # the aggressive order itself
    print(f"Aggressive order positions (0-indexed): {aggressive_positions}")
    print(f"Total messages per sample: {pos}")

    # Create scannable step function (JIT-compiled)
    step_fn = make_cst_scan_fn(n_levels, sim)

    # 3. Process batches
    for batch_idx, batch_i in enumerate(tqdm(sample_i, desc="Batches")):
        print(f'\n=== BATCH {batch_idx}: samples {batch_i} ===')

        # Load data
        device = jax.devices()[0]
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        msg_seq_raw = jax.device_put(jnp.array(msg_seq_raw), device)
        book_l2_init = jax.device_put(jnp.array(book_l2_init), device)
        b_seq_pv = jnp.array(b_seq_pv)

        # Split conditioning
        m_seq_raw_cond = msg_seq_raw[:, :n_cond_msgs, :]
        b_seq_pv_cond = onp.array(b_seq_pv[:, :n_cond_msgs + 1, 3:])
        init_time = b_seq_pv[:, 0, 1:3]
        init_time = jax.device_put(jnp.array(init_time), device)

        # Initialize JAX-LOB simulators by replaying conditioning messages
        sim_states = get_sims_vmap(
            book_l2_init, m_seq_raw_cond, init_time, sim,
        )

        # Save conditioning data for each sample in batch
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)
            msg_to_lobster_format(m_seq_raw_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_message_real_id_{sample_idx}.csv',
                index=False, header=False
            )
            book_to_lobster_format(b_seq_pv_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_orderbook_real_id_{sample_idx}.csv',
                index=False, header=False
            )

        # 4. Process each sample individually (CST is fast enough — no need for vmap)
        for i, sample_idx in enumerate(batch_i):
            # Extract per-sample JAX-LOB state
            sim_state_i = jax.tree.map(lambda x: x[i], sim_states)

            # Get post-conditioning L2 from JAX-LOB
            l2_flat = sim.get_L2_state(sim_state_i, n_levels)

            # Get init time for this sample
            init_time_i = jnp.float32(init_time[i, 0]) + jnp.float32(init_time[i, 1]) / 1e9

            # Initialize CST Book from L2 state
            cst_book = stoikov.init_book(l2_flat, cst_params, init_time_i)

            # Per-sample RNG
            rng, rng_sample = jax.random.split(rng)

            all_msgs = []
            all_books = []
            # Total messages per sample: total_blocks * n_gen_msgs + num_insertions
            total_msgs_per_sample = total_blocks * n_gen_msgs + num_insertions
            n_msg_todo = jnp.int32(total_msgs_per_sample)

            for block in range(total_blocks):
                # Re-sync the CST internal book to the authoritative JAX-LOB book at the
                # start of every block, preserving CST's advancing clock. The saved book is
                # JAX-LOB's; CST message *prices* come from cst_book. Without re-sync the two
                # diverge over a run, so CST background flow stops tracking the real book and
                # accelerates one-sided depletion (the negative-impact collapse). Re-anchoring
                # each block keeps generated LO/CA placement relative to the real best prices.
                # Re-anchor from a DEEPER L2 than the saved n_levels (=10): init_book zeroes all ticks
                # and only refills the levels present, so a shallow resync would discard CST's deep
                # resting liquidity every block and thin the book (-> faster depletion). Saving stays
                # 10-level; the resync uses up to RESYNC_LEVELS so deep liquidity is preserved.
                RESYNC_LEVELS = min(max(n_levels, 50), cst_params.num_ticks)
                l2_resync = sim.get_L2_state(sim_state_i, RESYNC_LEVELS)
                cst_book = stoikov.init_book(l2_resync, cst_params, cst_book.time)

                # Run one block of CST generation
                carry = (sim_state_i, cst_book, n_msg_todo, base_rates, cst_params, rng_sample)
                carry, (l2_books_block, msgs_block) = jax.lax.scan(
                    step_fn, carry, None, length=n_gen_msgs,
                )
                sim_state_i, cst_book, n_msg_todo, _, _, rng_sample = carry

                all_msgs.append(msgs_block)
                all_books.append(l2_books_block)

                # Inject aggressive order after insertion blocks
                if block < num_insertions:
                    # Get last message time from the block (msgs_block is (n_gen_msgs, 14))
                    last_time_s = msgs_block[-1, TIMEs_i]
                    last_time_ns = msgs_block[-1, TIMEns_i]

                    # Aggressive order gets order_id = n_msg_todo (descending)
                    sim_msg, msg_decoded = create_aggressive_order(
                        sim, sim_state_i,
                        last_time_s, last_time_ns,
                        tick_size, event_type, direction, order_volume,
                        n_msg_todo,
                    )

                    # Apply to JAX-LOB
                    sim_state_i = sim.process_order_array(sim_state_i, sim_msg)

                    # Apply to CST Book
                    # Get quantity actually used (from msg_decoded)
                    aggr_qty = msg_decoded[SIZE_i]
                    cst_book = apply_aggressive_to_cst_book(
                        cst_book, direction, aggr_qty, cst_params,
                    )

                    # Get L2 after aggressive order
                    l2_after_aggr = sim.get_L2_state(sim_state_i, n_levels)

                    # Decrement counter after aggressive order
                    n_msg_todo = n_msg_todo - 1

                    # Append aggressive msg and book as single-row arrays
                    all_msgs.append(msg_decoded[None, :])  # (1, 14)
                    all_books.append(l2_after_aggr[None, :])  # (1, n_levels*4)

            # Stack all messages and books
            all_msgs_arr = jnp.concatenate(all_msgs, axis=0)   # (total_msgs, 14)
            all_books_arr = jnp.concatenate(all_books, axis=0)  # (total_msgs, n_levels*4)

            # Save generated data
            date = ds.get_date(sample_idx)
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
                      f"({total_blocks} blocks x {n_gen_msgs} + {num_insertions} aggressive)")

        # Save aggressive indices once (worker 0 only). PER-DAY file (aggressive_indices_<date>.csv):
        # in per-day mode each day has its own mb -> its own insertion positions; the shared file gets
        # overwritten by the last day and is wrong for every other day (the analysis reads per-day).
        if batch_idx == 0 and worker_id == 0:
            aggressive_indices = onp.array(aggressive_positions)
            onp.savetxt(save_folder / 'aggressive_indices.csv', aggressive_indices, fmt='%d')
            onp.savetxt(save_folder / f'aggressive_indices_{date}.csv', aggressive_indices, fmt='%d')

        rng, _ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="CST Model Aggressive Scenario")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='lob_impact/3_scenarios/4.aggressive_scenario_cst_config.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument('--n_gen_msgs', type=int, default=None, help='Override n_gen_msgs from config')
    parser.add_argument('--direction', type=int, default=None, choices=[0, 1], help='Override direction (0=buy, 1=sell)')
    parser.add_argument('--worker_id', type=int, default=0, help='Worker index (0-based)')
    parser.add_argument('--num_workers', type=int, default=1, help='Total number of parallel workers')
    parser.add_argument('--save_folder', type=str, default=None,
                        help='Shared experiment folder (created by launcher). If not set, creates a new one.')
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

    worker_id = args.worker_id
    num_workers = args.num_workers

    # Create or reuse experiment folder
    if args.save_folder:
        save_folder = Path(args.save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
    else:
        save_folder = create_experiment_folder(cfg['save_dir'])
    print(f"Experiment folder: {save_folder}")

    # Set up logging (per-worker log file if multi-worker)
    log_suffix = f'_worker{worker_id}' if num_workers > 1 else ''
    log_file = save_folder / f'experiment{log_suffix}.log'
    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"\n{'='*60}")
    print(f"CST Model Aggressive Scenario Experiment")
    print(f"Worker {worker_id}/{num_workers}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Configuration: {cfg}")
    print(f"Experiment folder: {save_folder}")

    # Save config (only worker 0 or single-worker mode)
    if worker_id == 0:
        with open(save_folder / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)

    try:
        if cfg.get('per_day_params'):
            import pandas as pd
            per_day_csv = cfg['per_day_params']
            print(f"\nPer-day mode: loading {per_day_csv}")
            pd_df = pd.read_csv(per_day_csv)
            mult_target = cfg.get('order_volume_mult', 1.0)
            pd_df = pd_df[pd_df['mult'] == mult_target].reset_index(drop=True)
            print(f"  {len(pd_df)} days (mult={mult_target})")

            bsz = cfg['batch_size']
            n_samples_per_day = cfg.get('n_samples_per_day', max(bsz, cfg['n_samples'] // len(pd_df)))
            n_samples_per_day = max((n_samples_per_day // bsz) * bsz, bsz)
            print(f"  n_samples_per_day = {n_samples_per_day}")

            for day_idx, row in pd_df.iterrows():
                cfg_d = dict(cfg)
                cfg_d['order_volume'] = int(row['child'])
                cfg_d['n_gen_msgs'] = int(row['mb'])
                cfg_d['day_index'] = int(day_idx)
                cfg_d['n_samples'] = n_samples_per_day
                cfg_d.pop('per_day_params', None)
                print(f"\n--- Day {day_idx}: {row['day']}, child={row['child']}, mb={row['mb']} ---")
                run_cst_scenario(cfg_d, save_folder, worker_id=worker_id, num_workers=num_workers)
        else:
            run_cst_scenario(cfg, save_folder, worker_id=worker_id, num_workers=num_workers)

        print(f"\n{'='*60}")
        print(f"Worker {worker_id} completed!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {save_folder}")
        print(f"{'='*60}")
    finally:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
