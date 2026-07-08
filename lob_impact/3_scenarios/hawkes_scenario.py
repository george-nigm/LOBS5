#!/usr/bin/env python
"""
Hawkes Order-Flow Aggressive Scenario: self-exciting LOB generation with aggressive
order injection.

A near-verbatim clone of `cst_scenario.py` (same book plumbing, per-day loop,
aggressive-order insertion, aggressive_indices_<date>.csv writing, data_gen saving,
and output format) but with the generative core swapped from the CST parametric
model to a multivariate self-exciting **Hawkes** order-flow generator
(`lob_bench/hawkes_model/hawkes.py`).

Differences from cst_scenario.py:
  * Generative core: `hawkes.simulate_block` (Ogata thinning) instead of CST's
    `step_book`. Per block we (a) re-anchor a lightweight price-tracking book from
    the authoritative JAX-LOB L2 state, (b) pre-generate `n_gen_msgs` messages in
    pure numpy, (c) replay them into the JAX-LOB simulator via a jitted lax.scan,
    carrying only the Hawkes intensity state (S, t) across blocks for continuity.
  * Message -> JAX-LOB conversion uses the column layout the simulator actually
    reads (orderid in slot 4, traderid in slot 5 — see `construct_sim_msg` /
    `cond_type_side`). Cancel (type-3) order-ids are resolved against the live
    resting book; market (type-4) and limit (type-1) orders keep the descending
    order-id counter (same convention as the aggressive order / S5 / historic).

Output format matches the S5 / CST scenarios so existing analysis works with all.
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

# Hawkes generative core
sys.path.insert(0, os.path.join(parent_folder_path, 'lob_bench', 'hawkes_model'))
import hawkes

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
# JAX-LOB message helpers
# ============================================================================

# MUST match the sim's own cancel_mode (CANCEL_UNIFORM_AND_LARGE=3): get_random_id_match only
# applies the price-only "large" fallback when cfg.cancel_mode==3. With the default config
# (INCLUDE_INITS) a cancel whose sampled size exceeds every resting order at that price fails
# the strict (price AND qty) match -> idx=-1 -> garbage oid from the last array slot.
SIM_CONFIG = JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value)


def hawkes_update_cancel_oid(
    msg: jax.Array,
    rng: jax.Array,
    sim: OrderBook,
    sim_state: LobState,
) -> Tuple[jax.Array, jax.Array]:
    """
    For a cancel/delete message (type 3), resolve a real resting order id from the
    live JAX-LOB state and write it into the order-id slot (index 4 — the slot the
    simulator's `cond_type_side` reads). Limit (1) and market (4) orders are left
    unchanged (they keep the descending order-id counter). This corrects the slot
    used by CST's `update_oid` (which targeted slot 5).
    """
    def _leave(msg, rng):
        return msg, rng

    def _random_cancel(msg, rng):
        side = msg[1]  # data[1]: +1 -> bid side, -1 -> ask side (no flip for type 3)
        side_array = jax.lax.cond(
            side == 1,
            lambda a, b: b,
            lambda a, b: a,
            sim_state.asks, sim_state.bids,
        )
        rng, _rng = jax.random.split(rng)
        msg_dict = {"quantity": msg[2], "price": msg[3]}
        idx = job.get_random_id_match(SIM_CONFIG, _rng, side_array, msg_dict)
        # idx == -1 means NO order at that price: keep the synthetic oid (harmless no-op in the
        # sim) instead of indexing slot -1 (wraps to the last array row -> random real order).
        cancelled_oid = jnp.where(idx >= 0, side_array[idx, 2], msg[4])
        msg = msg.at[4].set(cancelled_oid)  # slot 4 = orderid (read by the simulator)
        return msg, rng

    msg, rng = jax.lax.cond(
        msg[0] == 3,
        _random_cancel,
        _leave,
        msg, rng,
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
    order_id: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Create an aggressive market order based on current JAX-LOB book state.
    order_id: descending counter (n_msg_todo), same convention as S5/historic/CST.

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
    best_bid_ask = sim.get_best_bid_and_ask_inclQuants(sim_state)
    avail = jax.lax.cond(
        direction == 0,
        lambda: best_bid_ask[1][1],  # ask volume for buy
        lambda: best_bid_ask[0][1],  # bid volume for sell
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


def make_apply_scan_fn(n_levels: int, sim: OrderBook, tick_size: int):
    """
    Returns a JAX-scannable step function that REPLAYS one pre-generated Hawkes
    message into the JAX-LOB simulator (no generation inside the scan).

    Carry: (sim_state, rng)
    xs per step: (row, order_id) where
        row = [event_type, side01, size, price, time_s, time_ns]  (int32, len 6)
        order_id = descending counter (int32)
    Output per step: (l2_book_flat, lobster_msg_decoded[14])
    """
    def _step_fn(carry, x):
        sim_state, rng = carry
        row, order_id = x
        event_type = row[0]
        side01 = row[1]
        size = row[2]
        price = row[3]
        time_s = row[4]
        time_ns = row[5]

        # data[1]: side in {-1, +1}; simulator flips it internally for type-4 orders
        data1 = side01 * 2 - 1

        # 8-field simulator message (orderid in slot 4, traderid in slot 5)
        msg_jaxlob = jnp.array([
            event_type, data1, size, price, order_id, -88, time_s, time_ns,
        ], dtype=jnp.int32)

        # Resolve a real resting order id for cancels (type 3)
        msg_jaxlob, rng = hawkes_update_cancel_oid(msg_jaxlob, rng, sim, sim_state)

        # Apply to JAX-LOB simulator
        sim_state = sim.process_order_array(sim_state, msg_jaxlob)

        # Extract L2 state
        l2_book = sim.get_L2_state(sim_state, n_levels)

        # Relative price for storage
        best_ask = sim.get_best_ask(sim_state)
        best_bid = sim.get_best_bid(sim_state)
        mid_price = (best_ask + best_bid) // 2
        mid_price = (mid_price // tick_size) * tick_size
        rel_price = jax.lax.cond(
            mid_price > 0,
            lambda: (price - mid_price) // tick_size,
            lambda: jnp.int32(0),
        )

        msg_decoded = jnp.array([
            order_id,                       # order_id (descending counter)
            event_type,                     # event_type
            side01,                         # direction {0,1}
            price,                          # price_abs
            rel_price,                      # price (relative)
            jnp.abs(size),                  # size
            0,                              # delta_t_s
            0,                              # delta_t_ns
            time_s,                         # time_s
            time_ns,                        # time_ns
            0,                              # price_ref
            0,                              # size_ref
            0,                              # time_s_ref
            0,                              # time_ns_ref
        ], dtype=jnp.int32)

        return (sim_state, rng), (l2_book, msg_decoded)

    return _step_fn


def _hawkes_msgs_to_rows(msgs_np: onp.ndarray) -> onp.ndarray:
    """
    Convert hawkes.simulate_block output (n,8) -> int32 rows (n,6) for the apply scan:
      [event_type, side01, size, price, time_s, time_ns].
    """
    t = msgs_np[:, hawkes.COL_TIME]
    time_s = onp.floor(t).astype(onp.int64)
    time_ns = onp.rint((t - time_s) * 1e9).astype(onp.int64)
    time_ns = onp.clip(time_ns, 0, 999_999_999)
    rows = onp.stack([
        msgs_np[:, hawkes.COL_ETYPE].astype(onp.int64),
        msgs_np[:, hawkes.COL_SIDE01].astype(onp.int64),
        msgs_np[:, hawkes.COL_SIZE].astype(onp.int64),
        msgs_np[:, hawkes.COL_PRICE].astype(onp.int64),
        time_s,
        time_ns,
    ], axis=1).astype(onp.int32)
    return rows


# ============================================================================
# Main scenario function
# ============================================================================

def run_hawkes_scenario(cfg: Dict[str, Any], save_folder: Path,
                        worker_id: int = 0, num_workers: int = 1):
    """
    Main function for the Hawkes aggressive scenario. Same flow as run_cst_scenario,
    with the generative core swapped for hawkes.simulate_block.
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

    # 1. Load Hawkes params
    print(f"Loading Hawkes params from {params_file}")
    params = hawkes.load_params(params_file)
    print(f"Hawkes params: n_dims={params['n_dims']} tick_size={params['tick_size']} "
          f"spectral_radius={params.get('spectral_radius'):.4g}")
    print(f"  mu={onp.asarray(params['mu'])}")
    print(f"  beta={onp.asarray(params['beta'])}")
    print(f"  event_counts={onp.asarray(params['event_counts'])}")

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
    aggressive_positions = []
    pos = 0
    for block in range(total_blocks):
        pos += n_gen_msgs  # messages from this block
        if block < num_insertions:
            aggressive_positions.append(pos)
            pos += 1  # the aggressive order itself
    print(f"Aggressive order positions (0-indexed): {aggressive_positions}")
    print(f"Total messages per sample: {pos}")

    # Create scannable apply function (JIT-compiled)
    apply_fn = make_apply_scan_fn(n_levels, sim, tick_size)

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

        # 4. Process each sample individually
        for i, sample_idx in enumerate(batch_i):
            # Extract per-sample JAX-LOB state
            sim_state_i = jax.tree.map(lambda x: x[i], sim_states)

            # GLOBAL generation id — unique across batches AND workers. Windows are drawn with
            # replacement (few windows/day), so keying files and the RNG on sample_idx alone made
            # duplicate draws byte-identical AND overwrite each other via gen_id_0.
            gid = (worker_id + batch_idx * num_workers) * batch_size + i

            # Get init time for this sample (fractional seconds)
            init_time_i = float(init_time[i, 0]) + float(init_time[i, 1]) / 1e9

            # Per-sample numpy RNG (deterministic from seed + window + generation id)
            np_rng = onp.random.default_rng([int(rng_seed), int(sample_idx), int(gid)])
            # Per-sample JAX RNG for cancel-id resolution inside the scan
            rng, rng_sample = jax.random.split(rng)

            # Hawkes intensity carry (continuity across blocks)
            hk_state = hawkes.init_state(params, init_time_i)

            all_msgs = []
            all_books = []
            total_msgs_per_sample = total_blocks * n_gen_msgs + num_insertions
            n_msg_todo = int(total_msgs_per_sample)

            for block in range(total_blocks):
                # Re-anchor internal book from the authoritative JAX-LOB L2 state
                l2_flat = onp.asarray(sim.get_L2_state(sim_state_i, n_levels))
                hk_book = hawkes.make_book(params, l2_flat)

                # Generate one block of messages (pure numpy)
                msgs_np, hk_state = hawkes.simulate_block(
                    params, n_gen_msgs, hk_book, hk_state, np_rng,
                )
                rows_np = _hawkes_msgs_to_rows(msgs_np)
                rows = jnp.asarray(rows_np)
                order_ids = jnp.arange(n_msg_todo, n_msg_todo - n_gen_msgs, -1, dtype=jnp.int32)

                # Replay block into JAX-LOB via scan
                (sim_state_i, rng_sample), (l2_books_block, msgs_block) = jax.lax.scan(
                    apply_fn, (sim_state_i, rng_sample), (rows, order_ids),
                )
                n_msg_todo = n_msg_todo - n_gen_msgs

                all_msgs.append(msgs_block)
                all_books.append(l2_books_block)

                # Inject aggressive order after insertion blocks
                if block < num_insertions:
                    last_time_s = msgs_block[-1, TIMEs_i]
                    last_time_ns = msgs_block[-1, TIMEns_i]

                    sim_msg, msg_decoded = create_aggressive_order(
                        sim, sim_state_i,
                        last_time_s, last_time_ns,
                        tick_size, event_type, direction, order_volume,
                        jnp.int32(n_msg_todo),
                    )

                    sim_state_i = sim.process_order_array(sim_state_i, sim_msg)

                    l2_after_aggr = sim.get_L2_state(sim_state_i, n_levels)

                    n_msg_todo = n_msg_todo - 1

                    all_msgs.append(msg_decoded[None, :])      # (1, 14)
                    all_books.append(l2_after_aggr[None, :])   # (1, n_levels*4)

            # Stack all messages and books
            all_msgs_arr = jnp.concatenate(all_msgs, axis=0)
            all_books_arr = jnp.concatenate(all_books, axis=0)

            # Save generated data
            date = ds.get_date(sample_idx)
            msg_to_lobster_format(all_msgs_arr).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_message_real_id_{sample_idx}_gen_id_{gid}.csv',
                index=False, header=False
            )
            book_to_lobster_format(all_books_arr).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_orderbook_real_id_{sample_idx}_gen_id_{gid}.csv',
                index=False, header=False
            )

            if batch_idx == 0 and i == 0:
                print(f"  First sample: {all_msgs_arr.shape[0]} total messages "
                      f"({total_blocks} blocks x {n_gen_msgs} + {num_insertions} aggressive)")

        # Save aggressive indices once (worker 0 only). PER-DAY file is the one analysis reads.
        if batch_idx == 0 and worker_id == 0:
            aggressive_indices = onp.array(aggressive_positions)
            onp.savetxt(save_folder / 'aggressive_indices.csv', aggressive_indices, fmt='%d')
            onp.savetxt(save_folder / f'aggressive_indices_{date}.csv', aggressive_indices, fmt='%d')

        rng, _ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="Hawkes Order-Flow Aggressive Scenario")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='lob_impact/3_scenarios/config_bet_composition.yaml',
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
    print(f"Hawkes Order-Flow Aggressive Scenario Experiment")
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

            if cfg.get('sample_slice') is not None:
                _k, _n = int(cfg['sample_slice']), int(cfg.get('n_slices', 1))
                import numpy as _np
                _keep = _np.array_split(_np.arange(len(pd_df)), _n)[_k]
                pd_df = pd_df.iloc[_keep]
                print(f"  DAY SLICE {_k}/{_n}: days {list(pd_df.index)}")

            for day_idx, row in pd_df.iterrows():
                cfg_d = dict(cfg)
                cfg_d['order_volume'] = int(row['child'])
                cfg_d['n_gen_msgs'] = int(row['mb'])
                cfg_d['day_index'] = int(day_idx)
                cfg_d['n_samples'] = n_samples_per_day
                cfg_d.pop('per_day_params', None)
                cfg_d.pop('sample_slice', None)
                cfg_d.pop('n_slices', None)
                print(f"\n--- Day {day_idx}: {row['day']}, child={row['child']}, mb={row['mb']} ---")
                run_hawkes_scenario(cfg_d, save_folder, worker_id=worker_id, num_workers=num_workers)
        else:
            run_hawkes_scenario(cfg, save_folder, worker_id=worker_id, num_workers=num_workers)

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
