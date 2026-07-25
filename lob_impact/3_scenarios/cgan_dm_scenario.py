#!/usr/bin/env python
"""
DeepMarket-CGAN Aggressive Scenario (Coletta-style cGAN, TSLA/INTC Jan-2015
checkpoints from github.com/LeonardoBerti00/DeepMarket).

Same config/output contract as qr_scenario.py, but the batch is generated in
LOCKSTEP: one generator forward serves all B samples per message (the CGAN
conditions on 256 rolling market-feature windows, so per-sample sequential
generation would waste the GPU). JAX-LOB runs vmapped on CPU
(JAX_PLATFORMS=cpu from the launcher); torch uses CUDA.

Extra config keys: deepmarket_root, cgan_ckpt, stock2015 (TSLA|INTC — selects
the 2015 normalization constants; must match the checkpoint).
Interarrival times are gamma-fit per day on the conditioning windows (upstream
WorldAgent behavior). All conditioning features are price-level-free, so the
2026 transfer needs no price re-anchoring.
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".40")

import jax
import jax.numpy as jnp
import numpy as onp
from scipy import stats as sp_stats
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_folder_path)

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

from lob_impact.core.cgan_dm_sampler import CganDmSampler

sys.path.insert(0, script_dir)
from qr_scenario import (  # noqa: E402
    TeeLogger, create_experiment_folder,
    create_aggressive_order, hawkes_update_cancel_oid,
    TIMEs_i, TIMEns_i, DIRECTION_i, DTs_i, DTns_i,
)

MAX_RESAMPLE_ROUNDS = 8


def run_cgan_scenario(cfg: Dict[str, Any], save_folder: Path,
                      worker_id: int = 0, num_workers: int = 1):
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

    deepmarket_root = cfg['deepmarket_root']
    cgan_ckpt = cfg['cgan_ckpt']
    stock2015 = cfg.get('stock2015', stock)

    total_blocks = num_insertions + num_coolings

    import torch
    sampler = CganDmSampler(deepmarket_root, cgan_ckpt, stock2015,
                            device='cuda' if torch.cuda.is_available() else 'cpu')

    rng = jax.random.key(rng_seed)
    sim = OrderBook(cfg=JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))

    day_index = cfg.get('day_index', None)
    day_indeces = [day_index] if day_index is not None else None

    print(f"Loading dataset from {data_dir} (day_indeces={day_indeces})")
    ds = get_dataset(data_dir, n_cond_msgs, n_eval_msgs_dataset,
                     test_split=test_split, day_indeces=day_indeces)
    print(f"Dataset length: {len(ds)}")

    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    assert n_samples % batch_size == 0
    rng, _ = jax.random.split(rng)
    rng, rng_ = jax.random.split(rng)
    all_sample_i = jax.random.choice(
        rng_, jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size),
        replace=(n_samples > len(ds)),
    ).tolist()

    sample_i = all_sample_i[worker_id::num_workers]
    print(f"Worker {worker_id}: {len(sample_i)}/{len(all_sample_i)} batches")

    aggressive_positions = []
    pos = 0
    for block in range(total_blocks):
        pos += n_gen_msgs
        if block < num_insertions:
            aggressive_positions.append(pos)
            pos += 1
    print(f"Aggressive order positions (0-indexed): {aggressive_positions}")
    print(f"Total messages per sample: {pos}")

    # vmapped JAX-LOB ops (batch axis over samples)
    step_vmap = jax.jit(jax.vmap(lambda st, m: sim.process_order_array(st, m)))
    l2_vmap = jax.jit(jax.vmap(lambda st: sim.get_L2_state(st, n_levels)))
    cancel_vmap = jax.jit(jax.vmap(
        lambda m, r, st: hawkes_update_cancel_oid(m, r, sim, st)))
    aggr_vmap = jax.jit(jax.vmap(
        lambda st, ts, tns, oid: create_aggressive_order(
            sim, st, ts, tns, tick_size, event_type, direction, order_volume, oid)))

    for batch_idx, batch_i in enumerate(tqdm(sample_i, desc="Batches")):
        print(f'\n=== BATCH {batch_idx}: samples {batch_i} ===')
        B = len(batch_i)

        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        msg_seq_raw = jnp.array(msg_seq_raw)
        book_l2_init = jnp.array(book_l2_init)
        b_seq_pv = jnp.array(b_seq_pv)

        m_seq_raw_cond = msg_seq_raw[:, :n_cond_msgs, :]
        b_seq_pv_cond = onp.array(b_seq_pv[:, :n_cond_msgs + 1, 3:])
        init_time = jnp.array(b_seq_pv[:, 0, 1:3])

        sim_states = get_sims_vmap(book_l2_init, m_seq_raw_cond, init_time, sim)

        # gamma fit of interarrivals on the conditioning windows (per batch)
        cond_np = onp.asarray(m_seq_raw_cond)
        dts = cond_np[:, :, DTs_i].astype(onp.float64) \
            + cond_np[:, :, DTns_i].astype(onp.float64) / 1e9
        dts = dts[dts > 0]
        ia_shape, ia_loc, ia_scale = sp_stats.gamma.fit(dts)
        print(f"gamma interarrival fit: shape={ia_shape:.4f} loc={ia_loc:.2e} scale={ia_scale:.4f}")

        # save conditioning
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)
            msg_to_lobster_format(m_seq_raw_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_message_real_id_{sample_idx}.csv',
                index=False, header=False)
            book_to_lobster_format(b_seq_pv_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_orderbook_real_id_{sample_idx}.csv',
                index=False, header=False)

        # ring-buffer init: L2 snapshots + order signs from the conditioning window
        sampler.init_state(b_seq_pv_cond, cond_np[:, :, DIRECTION_i])

        np_rng = onp.random.default_rng([int(rng_seed), worker_id, batch_idx])
        torch_gen = None  # generator noise from torch default (seeded below)
        import torch
        torch.manual_seed(int(rng_seed) * 100000 + worker_id * 1000 + batch_idx)

        rng, rng_sample = jax.random.split(rng)

        # per-sample clocks start at the end of the conditioning window
        clock_ns = (cond_np[:, -1, TIMEs_i].astype(onp.int64) * 1_000_000_000
                    + cond_np[:, -1, TIMEns_i].astype(onp.int64))

        total_msgs_per_sample = total_blocks * n_gen_msgs + num_insertions
        n_msg_todo = int(total_msgs_per_sample)

        msgs_out = onp.zeros((B, total_msgs_per_sample, 14), dtype=onp.int64)
        books_out = onp.zeros((B, total_msgs_per_sample, n_levels * 4), dtype=onp.int64)
        out_ptr = 0
        n_fallback = 0

        for block in range(total_blocks):
            for _ in range(n_gen_msgs):
                l2_np = onp.asarray(l2_vmap(sim_states))

                raw = sampler.sample_raw(torch_gen)
                etype, side01, size, price, ok = sampler.decode(raw, l2_np)
                rounds = 0
                while (~ok).any() and rounds < MAX_RESAMPLE_ROUNDS:
                    raw2 = sampler.sample_raw(torch_gen)
                    e2, s2, z2, p2, ok2 = sampler.decode(raw2, l2_np)
                    take = (~ok) & ok2
                    etype = onp.where(take, e2, etype)
                    side01 = onp.where(take, s2, side01)
                    size = onp.where(take, z2, size)
                    price = onp.where(take, p2, price)
                    ok |= ok2
                    rounds += 1
                if (~ok).any():
                    # neutral fallback: 1-lot deep LO at the 10th visible level, own side
                    bad = ~ok
                    n_fallback += int(bad.sum())
                    etype = onp.where(bad, 1, etype)
                    side01 = onp.where(bad, 1 - direction, side01)
                    deep = onp.where(side01 == 1, l2_np[:, 2 + 9 * 4], l2_np[:, 0 + 9 * 4])
                    deep = onp.where(deep > 0, deep,
                                     onp.where(side01 == 1, l2_np[:, 2], l2_np[:, 0]))
                    price = onp.where(bad, deep, price)
                    size = onp.where(bad, 1, size)

                # advance clocks with gamma interarrivals
                dt_s = sp_stats.gamma.rvs(ia_shape, ia_loc, ia_scale,
                                          size=B, random_state=np_rng)
                dt_ns = onp.maximum((dt_s * 1e9).astype(onp.int64), 1)
                clock_ns = clock_ns + dt_ns
                ts = clock_ns // 1_000_000_000
                tns = clock_ns % 1_000_000_000

                data1 = side01 * 2 - 1
                msgs = onp.stack([
                    etype, data1, size, price,
                    onp.full(B, n_msg_todo, dtype=onp.int64),
                    onp.full(B, -88, dtype=onp.int64),
                    ts, tns,
                ], axis=1).astype(onp.int32)
                msgs_j = jnp.asarray(msgs)

                rng_sample, sub = jax.random.split(rng_sample)
                keys = jax.random.split(sub, B)
                msgs_j, _ = cancel_vmap(msgs_j, keys, sim_states)

                sim_states = step_vmap(sim_states, msgs_j)
                l2_new = onp.asarray(l2_vmap(sim_states))

                mid = ((l2_new[:, 0] + l2_new[:, 2]) // 2 // tick_size) * tick_size
                rel = onp.where(mid > 0, (price - mid) // tick_size, 0)
                msgs_out[:, out_ptr, :] = onp.stack([
                    onp.full(B, n_msg_todo, dtype=onp.int64), etype, side01,
                    price, rel, onp.abs(size),
                    dt_ns // 1_000_000_000, dt_ns % 1_000_000_000,
                    ts, tns,
                    onp.zeros(B, onp.int64), onp.zeros(B, onp.int64),
                    onp.zeros(B, onp.int64), onp.zeros(B, onp.int64),
                ], axis=1)
                books_out[:, out_ptr, :] = l2_new
                out_ptr += 1
                n_msg_todo -= 1

                sampler.push(l2_new, side01)

            if block < num_insertions:
                ts_j = jnp.asarray(msgs_out[:, out_ptr - 1, TIMEs_i].astype(onp.int32))
                tns_j = jnp.asarray(msgs_out[:, out_ptr - 1, TIMEns_i].astype(onp.int32))
                oids = jnp.full((B,), n_msg_todo, dtype=jnp.int32)
                sim_msgs, msg_dec = aggr_vmap(sim_states, ts_j, tns_j, oids)
                sim_states = step_vmap(sim_states, sim_msgs)
                l2_new = onp.asarray(l2_vmap(sim_states))

                msgs_out[:, out_ptr, :] = onp.asarray(msg_dec)
                books_out[:, out_ptr, :] = l2_new
                out_ptr += 1
                n_msg_todo -= 1
                clock_ns = clock_ns + 1
                sampler.push(l2_new, onp.asarray(msg_dec)[:, DIRECTION_i])

        print(f"  fallback messages: {n_fallback} "
              f"({100.0 * n_fallback / (B * total_msgs_per_sample):.2f}%)")

        for i, sample_idx in enumerate(batch_i):
            gid = (worker_id + batch_idx * num_workers) * batch_size + i
            date = ds.get_date(sample_idx)
            msg_to_lobster_format(jnp.asarray(msgs_out[i])).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_message_real_id_{sample_idx}_gen_id_{gid}.csv',
                index=False, header=False)
            book_to_lobster_format(jnp.asarray(books_out[i])).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_orderbook_real_id_{sample_idx}_gen_id_{gid}.csv',
                index=False, header=False)

        if batch_idx == 0 and worker_id == 0:
            aggressive_indices = onp.array(aggressive_positions)
            onp.savetxt(save_folder / 'aggressive_indices.csv', aggressive_indices, fmt='%d')
            onp.savetxt(save_folder / f'aggressive_indices_{date}.csv', aggressive_indices, fmt='%d')

        rng, _ = jax.random.split(rng)

    print(f"\nResults saved to: {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepMarket-CGAN Aggressive Scenario")
    parser.add_argument('--config', '-c', type=str,
                        default='lob_impact/3_scenarios/config_bet_composition.yaml')
    parser.add_argument('--n_gen_msgs', type=int, default=None)
    parser.add_argument('--direction', type=int, default=None, choices=[0, 1])
    parser.add_argument('--worker_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_folder', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.n_gen_msgs is not None:
        cfg['n_gen_msgs'] = args.n_gen_msgs
    if args.direction is not None:
        cfg['direction'] = args.direction

    worker_id, num_workers = args.worker_id, args.num_workers

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
    print(f"DeepMarket-CGAN Aggressive Scenario — worker {worker_id}/{num_workers}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"Configuration: {cfg}")

    if worker_id == 0:
        with open(save_folder / 'config.yaml', 'w') as f:
            yaml.dump(cfg, f)

    try:
        if cfg.get('per_day_params'):
            import pandas as pd
            pd_df = pd.read_csv(cfg['per_day_params'])
            mult_target = cfg.get('order_volume_mult', 1.0)
            pd_df = pd_df[pd_df['mult'] == mult_target].reset_index(drop=True)
            bsz = cfg['batch_size']
            n_samples_per_day = cfg.get('n_samples_per_day', max(bsz, cfg['n_samples'] // len(pd_df)))
            n_samples_per_day = max((n_samples_per_day // bsz) * bsz, bsz)
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
                for k in ('per_day_params', 'sample_slice', 'n_slices'):
                    cfg_d.pop(k, None)
                print(f"\n--- Day {day_idx}: {row['day']}, child={row['child']}, mb={row['mb']} ---")
                run_cgan_scenario(cfg_d, save_folder, worker_id=worker_id, num_workers=num_workers)
        else:
            run_cgan_scenario(cfg, save_folder, worker_id=worker_id, num_workers=num_workers)

        print(f"\nWorker {worker_id} completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    finally:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
