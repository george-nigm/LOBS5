#!/usr/bin/env python
"""
K-NN Resampling Aggressive Scenario (Giegrich, Oomen, Reisinger, arXiv:2409.06514).

A clone of `qr_scenario.py` (same book plumbing, per-day loop, aggressive-order
insertion, aggressive_indices_<date>.csv writing, data_gen saving, output format)
with the generative core swapped for K-NN message resampling
(`lob_impact/core/knn_resampler.py`): at each step the current JAX-LOB L2 state
is matched (Euclidean, volume profile around the mid, K=20, k ~ U{1..K}) against
a pool of historical states from the SAME trading day, and the neighbor's next
`knn_block` messages are adopted with additive price re-anchoring to the live
mid. No training, no checkpoint — the pool is the model.

Config extras (all optional): knn_l_ticks (10), knn_block (25), knn_pool_max
(1_000_000), knn_K (20). Paper-faithful cadence = knn_block 1 (25 is the
throughput compromise; deviation documented in MODEL_CATALOG.md).
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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

from gymnax_exchange.jaxob.jorderbook import OrderBook
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst

from lob.inference_no_errcorr import (
    get_sims_vmap, get_dataset,
    msg_to_lobster_format, book_to_lobster_format,
)

from lob_impact.core.knn_resampler import build_day_pools

# Reuse the shared scenario plumbing from the QR clone (same directory, run as a file)
sys.path.insert(0, script_dir)
from qr_scenario import (  # noqa: E402
    TeeLogger, create_experiment_folder,
    create_aggressive_order, make_apply_scan_fn,
    TIMEs_i, TIMEns_i,
)


def run_knn_scenario(cfg: Dict[str, Any], save_folder: Path,
                     worker_id: int = 0, num_workers: int = 1):
    """Same flow as run_qr_scenario, generative core = K-NN resampling."""
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
    n_levels = cfg.get('n_levels', 10)
    test_split = cfg.get('test_split', 0)
    event_type = cfg['event_type']
    direction = cfg['direction']
    n_eval_msgs_dataset = cfg.get('n_eval_msgs_dataset', 500)
    order_volume = cfg['order_volume']

    knn_l_ticks = cfg.get('knn_l_ticks', 10)
    knn_block = cfg.get('knn_block', 25)
    knn_pool_max = cfg.get('knn_pool_max', 1_000_000)
    knn_K = cfg.get('knn_K', 20)

    total_blocks = num_insertions + num_coolings

    print(f"K-NN resampler: l_ticks={knn_l_ticks} block={knn_block} "
          f"pool_max={knn_pool_max} K={knn_K}")
    pools = build_day_pools(data_dir, tick_size, knn_l_ticks, knn_pool_max, knn_block)

    rng = jax.random.key(rng_seed)
    sim = OrderBook(cfg=JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))

    day_index = cfg.get('day_index', None)
    day_indeces = [day_index] if day_index is not None else None

    print(f"Loading dataset from {data_dir} (day_indeces={day_indeces})")
    ds = get_dataset(
        data_dir, n_cond_msgs, n_eval_msgs_dataset,
        test_split=test_split, day_indeces=day_indeces,
    )
    print(f"Dataset length: {len(ds)}")

    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    assert n_samples % batch_size == 0, f'n_samples ({n_samples}) must be divisible by batch_size ({batch_size})'
    rng, _ = jax.random.split(rng)
    rng, rng_ = jax.random.split(rng)
    all_sample_i = jax.random.choice(
        rng_,
        jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size),
        replace=(n_samples > len(ds)),
    ).tolist()

    sample_i = all_sample_i[worker_id::num_workers]
    print(f"Worker {worker_id}: {len(sample_i)}/{len(all_sample_i)} batches "
          f"({len(sample_i) * batch_size} samples)")

    aggressive_positions = []
    pos = 0
    for block in range(total_blocks):
        pos += n_gen_msgs
        if block < num_insertions:
            aggressive_positions.append(pos)
            pos += 1
    print(f"Aggressive order positions (0-indexed): {aggressive_positions}")
    print(f"Total messages per sample: {pos}")

    apply_fn = make_apply_scan_fn(n_levels, sim, tick_size)

    for batch_idx, batch_i in enumerate(tqdm(sample_i, desc="Batches")):
        print(f'\n=== BATCH {batch_idx}: samples {batch_i} ===')

        device = jax.devices()[0]
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        msg_seq_raw = jax.device_put(jnp.array(msg_seq_raw), device)
        book_l2_init = jax.device_put(jnp.array(book_l2_init), device)
        b_seq_pv = jnp.array(b_seq_pv)

        m_seq_raw_cond = msg_seq_raw[:, :n_cond_msgs, :]
        b_seq_pv_cond = onp.array(b_seq_pv[:, :n_cond_msgs + 1, 3:])
        init_time = b_seq_pv[:, 0, 1:3]
        init_time = jax.device_put(jnp.array(init_time), device)

        sim_states = get_sims_vmap(book_l2_init, m_seq_raw_cond, init_time, sim)

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

        for i, sample_idx in enumerate(batch_i):
            sim_state_i = jax.tree.map(lambda x: x[i], sim_states)
            gid = (worker_id + batch_idx * num_workers) * batch_size + i
            date = ds.get_date(sample_idx)
            pool = pools[date]

            np_rng = onp.random.default_rng([int(rng_seed), int(sample_idx), int(gid)])
            rng, rng_sample = jax.random.split(rng)

            # clock starts where the conditioning window ends
            last_msg_cond = onp.asarray(m_seq_raw_cond[i][-1])
            last_ts = int(last_msg_cond[TIMEs_i])
            last_tns = int(last_msg_cond[TIMEns_i])

            all_msgs = []
            all_books = []
            total_msgs_per_sample = total_blocks * n_gen_msgs + num_insertions
            n_msg_todo = int(total_msgs_per_sample)

            for block in range(total_blocks):
                done = 0
                while done < n_gen_msgs:
                    m = min(knn_block, n_gen_msgs - done)

                    # current authoritative book state -> query + price anchor
                    l2_flat = onp.asarray(sim.get_L2_state(sim_state_i, n_levels))
                    best_ask, best_bid = int(l2_flat[0]), int(l2_flat[2])
                    if best_ask > 0 and best_bid > 0:
                        cur_mid = ((best_ask + best_bid) // 2 // tick_size) * tick_size
                    else:
                        cur_mid = int(last_msg_cond[3])  # degenerate book: fall back to last cond price

                    anchor = pool.query(l2_flat, knn_K, np_rng)
                    rows_np, last_ts, last_tns = pool.take_block(
                        anchor, m, cur_mid, last_ts, last_tns)

                    rows = jnp.asarray(rows_np.astype(onp.int32))
                    order_ids = jnp.arange(n_msg_todo, n_msg_todo - m, -1, dtype=jnp.int32)

                    (sim_state_i, rng_sample), (l2_books_block, msgs_block) = jax.lax.scan(
                        apply_fn, (sim_state_i, rng_sample), (rows, order_ids),
                    )
                    n_msg_todo -= m
                    done += m
                    all_msgs.append(msgs_block)
                    all_books.append(l2_books_block)

                if block < num_insertions:
                    last_time_s = all_msgs[-1][-1, TIMEs_i]
                    last_time_ns = all_msgs[-1][-1, TIMEns_i]

                    sim_msg, msg_decoded = create_aggressive_order(
                        sim, sim_state_i,
                        last_time_s, last_time_ns,
                        tick_size, event_type, direction, order_volume,
                        jnp.int32(n_msg_todo),
                    )
                    sim_state_i = sim.process_order_array(sim_state_i, sim_msg)
                    l2_after_aggr = sim.get_L2_state(sim_state_i, n_levels)
                    n_msg_todo -= 1
                    last_ts = int(last_time_s)
                    last_tns = int(last_time_ns) + 1

                    all_msgs.append(msg_decoded[None, :])
                    all_books.append(l2_after_aggr[None, :])

            all_msgs_arr = jnp.concatenate(all_msgs, axis=0)
            all_books_arr = jnp.concatenate(all_books, axis=0)

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

        if batch_idx == 0 and worker_id == 0:
            aggressive_indices = onp.array(aggressive_positions)
            onp.savetxt(save_folder / 'aggressive_indices.csv', aggressive_indices, fmt='%d')
            onp.savetxt(save_folder / f'aggressive_indices_{date}.csv', aggressive_indices, fmt='%d')

        rng, _ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="K-NN Resampling Aggressive Scenario")
    parser.add_argument('--config', '-c', type=str,
                        default='lob_impact/3_scenarios/config_bet_composition.yaml')
    parser.add_argument('--n_gen_msgs', type=int, default=None)
    parser.add_argument('--direction', type=int, default=None, choices=[0, 1])
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_folder', type=str, default=None)
    return parser.parse_args()


def main():
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.n_gen_msgs is not None:
        cfg['n_gen_msgs'] = args.n_gen_msgs
    if args.direction is not None:
        cfg['direction'] = args.direction

    worker_id = args.worker_id
    num_workers = args.num_workers

    if args.save_folder:
        save_folder = Path(args.save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
    else:
        save_folder = create_experiment_folder(cfg['save_dir'])

    log_suffix = f'_worker{worker_id}' if num_workers > 1 else ''
    log_file = save_folder / f'experiment{log_suffix}.log'
    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"\n{'='*60}")
    print(f"K-NN Resampling Aggressive Scenario")
    print(f"Worker {worker_id}/{num_workers}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Configuration: {cfg}")
    print(f"Experiment folder: {save_folder}")

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
                _keep = onp.array_split(onp.arange(len(pd_df)), _n)[_k]
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
                run_knn_scenario(cfg_d, save_folder, worker_id=worker_id, num_workers=num_workers)
        else:
            run_knn_scenario(cfg, save_folder, worker_id=worker_id, num_workers=num_workers)

        print(f"\n{'='*60}")
        print(f"Worker {worker_id} completed!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
    finally:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
