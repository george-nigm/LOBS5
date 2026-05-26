#!/usr/bin/env python
"""
Heuristic Scenario: Historical Replay with Aggressive Insertions and Price Shifting

Same as historic scenario (2.historic_scenario.py), but when an aggressive order
fully consumes the best level (SIZE == available_volume), a shift_ticks counter
increments and subsequent historical limit/execution messages have their prices
shifted by tick_size * shift_ticks to compensate for consumed liquidity.

Output: LOBSTER CSV files in data_cond/ and data_gen/ folders.
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


@jax.jit
def create_aggressive_order(
    sim: OrderBook,
    sim_state: LobState,
    last_msg_decoded: jax.Array,
    tick_size: int,
    event_type: int,
    direction: int,
    order_volume: int,
    order_id: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Create an aggressive market order based on current book state.

    Returns:
        sim_msg: Message for simulator (8 fields)
        msg_decoded: Decoded message for storage (14 fields)
        level_consumed: Boolean — whether SIZE == available volume at best level
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
        lambda: best_ask_pv[1],  # ask volume for buy
        lambda: best_bid_pv[1],  # bid volume for sell
    ).astype(jnp.int32)

    quantity = jnp.minimum(jnp.int32(order_volume), avail)
    level_consumed = (quantity == avail) & (avail > 0)

    time_s = last_msg_decoded[TIMEs_i].astype(jnp.int32)
    time_ns = (last_msg_decoded[TIMEns_i] + 1).astype(jnp.int32)

    sim_msg = construct_sim_msg(
        event_type, direction, quantity, price,
        order_id, time_s, time_ns,
    )

    mid_price = (sim.get_best_ask(sim_state) + sim.get_best_bid(sim_state)) // 2
    mid_price = (mid_price // tick_size) * tick_size
    rel_price = (price - mid_price) // tick_size

    msg_decoded = jnp.array([
        order_id,             # order_id (descending counter)
        event_type,
        direction,
        price,
        rel_price,
        quantity,
        0,
        1,
        time_s,
        time_ns,
        0,
        0,
        0,
        0,
    ], dtype=jnp.int32)

    return sim_msg, msg_decoded, level_consumed


# Batched versions
create_aggressive_order_batched = jax.jit(
    jax.vmap(
        create_aggressive_order,
        in_axes=(None, 0, 0, None, None, None, None, 0)
    ),
    static_argnums=(0,)
)

msg_to_jnp_vmap = jax.jit(jax.vmap(msg_to_jnp))


def shift_prices(
    msg: jnp.ndarray,
    shift_amount: jnp.ndarray,
    direction: int,
) -> jnp.ndarray:
    """
    Shift prices of limit/execution historical messages to compensate for
    consumed liquidity.

    The sign of the shift follows (1 - 2*direction) from the heuristic file:
    - direction=0 (buying, hitting ask): shift UP (+tick_size per consumed level)
    - direction=1 (selling, hitting bid): shift DOWN (-tick_size per consumed level)

    Args:
        msg: Historical messages, shape (batch, 14)
        shift_amount: tick_size * shift_ticks per batch item, shape (batch,)
        direction: 0=buy, 1=sell
    """
    sign = 1 - 2 * direction  # +1 for buy, -1 for sell

    # Shift sell (ask-side) limit/execution orders
    is_sell_relevant = (
        ((msg[:, EVENT_TYPE_i] == 1) | (msg[:, EVENT_TYPE_i] == 4))
        & (msg[:, DIRECTION_i] == 0)
    )
    msg = msg.at[:, PRICE_ABS_i].set(
        jnp.where(is_sell_relevant, msg[:, PRICE_ABS_i] + sign * shift_amount, msg[:, PRICE_ABS_i])
    )

    # Shift buy (bid-side) limit/execution orders
    is_buy_relevant = (
        ((msg[:, EVENT_TYPE_i] == 1) | (msg[:, EVENT_TYPE_i] == 4))
        & (msg[:, DIRECTION_i] == 1)
    )
    msg = msg.at[:, PRICE_ABS_i].set(
        jnp.where(is_buy_relevant, msg[:, PRICE_ABS_i] + sign * shift_amount, msg[:, PRICE_ABS_i])
    )

    return msg


def run_heuristic_scenario(cfg: Dict[str, Any], save_folder: Path):
    """
    Main function for heuristic scenario.

    Same as historic scenario but with price shifting: when an aggressive order
    fully consumes the best level, shift_ticks increments and subsequent historical
    messages have their prices adjusted.
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
    n_vol_series = cfg['n_vol_series']
    book_dim = cfg['book_dim']
    test_split = cfg.get('test_split', 0)
    event_type = cfg['event_type']
    direction = cfg['direction']
    n_eval_msgs_dataset = cfg.get('n_eval_msgs_dataset', 500)
    order_volume = cfg['order_volume']

    total_eval_msgs_needed = (num_insertions + num_coolings) * n_gen_msgs

    # Initialize
    rng = jax.random.key(rng_seed)

    # Initialize simulator
    sim = OrderBook(cfg=JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))

    # Vmapped operations
    process_msg_vmap = jax.jit(jax.vmap(sim.process_order_array, in_axes=(0, 0)))
    get_L2_vmap = jax.jit(jax.vmap(sim.get_L2_state, in_axes=(0, None)), static_argnums=(1,))

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
    (save_folder / 'data_real').mkdir(exist_ok=True, parents=True)

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

    # Build insertion schedule
    insertion_steps = set()
    offset = 0
    for i in range(num_insertions):
        insert_step = (i + 1) * n_gen_msgs + offset
        insertion_steps.add(insert_step)
        offset += 1
    total_steps = total_eval_msgs_needed + num_insertions
    print(f"Insertion steps: {sorted(insertion_steps)}")
    print(f"Total steps per batch: {total_steps}")

    book_levels = book_dim // 4 if book_dim > 4 else 10

    # Process batches
    for batch_idx, batch_i in enumerate(tqdm(sample_i, desc="Batches")):
        print(f'\n=== BATCH {batch_idx}: samples {batch_i} ===')

        # Load data
        device = jax.devices()[0]
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        msg_seq_raw = jax.device_put(jnp.array(msg_seq_raw), device)
        book_l2_init = jax.device_put(jnp.array(book_l2_init), device)
        b_seq_pv = jnp.array(b_seq_pv)

        # Split conditioning and evaluation
        m_seq_raw_cond = msg_seq_raw[:, :n_cond_msgs, :]
        m_seq_raw_eval = msg_seq_raw[:, n_cond_msgs:, :]
        b_seq_pv_cond = onp.array(b_seq_pv[:, :n_cond_msgs + 1, 3:])
        init_time = b_seq_pv[:, 0, 1:3]
        init_time = jax.device_put(jnp.array(init_time), device)

        # Initialize simulators
        sim_states = get_sims_vmap(
            book_l2_init, m_seq_raw_cond, init_time, sim,
        )

        # Sequential processing loop
        all_msgs = []
        all_books = []
        eval_idx = 0
        shift_ticks = jnp.zeros(batch_size, dtype=jnp.int32)
        n_msg_todo = jnp.full(batch_size, total_steps, dtype=jnp.int32)

        for step in range(total_steps):
            if step in insertion_steps:
                last_msg = all_msgs[-1] if all_msgs else m_seq_raw_cond[:, -1, :]

                # Aggressive order gets n_msg_todo - 1 (descending counter)
                order_ids = n_msg_todo - 1

                # Create and process aggressive order
                sim_msg, msg_decoded, level_consumed = create_aggressive_order_batched(
                    sim, sim_states, last_msg, tick_size,
                    event_type, direction, order_volume,
                    order_ids,
                )
                sim_states = process_msg_vmap(sim_states, sim_msg)
                all_msgs.append(msg_decoded)
                n_msg_todo = n_msg_todo - 1

                # Update shift_ticks where level was fully consumed
                shift_ticks = shift_ticks + level_consumed.astype(jnp.int32)

            else:
                # Process historical message WITH price shifting
                msg = m_seq_raw_eval[:, eval_idx, :]

                # Apply price shift to compensate for consumed liquidity
                shift_amount = tick_size * shift_ticks
                msg = shift_prices(msg, shift_amount, direction)

                sim_msg = msg_to_jnp_vmap(msg)
                sim_states = process_msg_vmap(sim_states, sim_msg)
                all_msgs.append(msg)
                eval_idx += 1
                n_msg_todo = n_msg_todo - 1

            # Extract L2 book state
            l2_state = get_L2_vmap(sim_states, book_levels)
            all_books.append(l2_state[:, :book_l2_init.shape[1]])

        print(f"  Processed {len(all_msgs)} steps ({eval_idx} historical + {num_insertions} aggressive)")
        print(f"  Final shift_ticks: {shift_ticks}")

        # Stack results
        all_msgs_arr = jnp.stack(all_msgs, axis=1)
        all_books_arr = jnp.stack(all_books, axis=1)

        # Save for each sample in batch
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)

            # Conditioning data
            msg_to_lobster_format(m_seq_raw_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_message_real_id_{sample_idx}.csv',
                index=False, header=False
            )
            book_to_lobster_format(b_seq_pv_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_orderbook_real_id_{sample_idx}.csv',
                index=False, header=False
            )

            # Generated data (historical shifted + aggressive)
            msg_to_lobster_format(all_msgs_arr[i]).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_message_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )
            book_to_lobster_format(all_books_arr[i]).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_orderbook_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )

        # Save aggressive indices once
        if batch_idx == 0:
            aggressive_indices = onp.array(sorted(insertion_steps))
            onp.savetxt(save_folder / 'aggressive_indices.csv', aggressive_indices, fmt='%d')

        rng, _ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="Heuristic Scenario: Historical Replay with Price Shifting")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='lob_impact/3.heuristic_scenario_config.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument('--n_gen_msgs', type=int, default=None, help='Override n_gen_msgs from config')
    parser.add_argument('--direction', type=int, default=None, choices=[0, 1], help='Override direction (0=buy, 1=sell)')
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
    print(f"Heuristic Scenario Experiment")
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
                run_heuristic_scenario(cfg_d, save_folder)
        else:
            run_heuristic_scenario(cfg, save_folder)

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
