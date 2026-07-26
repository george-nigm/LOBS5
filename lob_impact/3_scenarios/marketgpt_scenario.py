#!/usr/bin/env python
"""
MarketGPT Aggressive Scenario (message-level ITCH transformer from
github.com/aaron-wheeler/MarketGPT, AAPL fine-tuned `ckpt_finetune_AAPL_v3.pt`).

Same config/output contract as cgan_dm_scenario.py, and the batch is generated
in LOCKSTEP the same way: one 24-token autoregressive message serves all B
samples per step (batched KV cache), JAX-LOB runs vmapped on CPU
(JAX_PLATFORMS=cpu from the launcher); torch uses CUDA.

Differences from the CGAN scenario:
  * The model is autoregressive over its OWN message history: the conditioning
    window is tokenized (proc-ITCH 24-token encoding) and prefilled into the
    KV cache; after every applied message the cache is refreshed with the
    CORRECTED tokens (actual matched ref-order fields, clock-consistent
    times), so the model also SEES the injected aggressive metaorder.
  * Reference events (E/C/D/R) are resolved against the live JAX-LOB L3 state
    with the price+size+time cascade from simulate.ipynb's
    find_matching_order; unresolvable / malformed messages are resampled
    (cap MAX_RESAMPLE_ROUNDS), then fall back to a neutral deep 1-lot LO.
  * R (replace) applies TWO sim ops (cancel old + add new) inside ONE message
    slot; only the add leg is written to the output CSV (the cancel is folded
    into the same row's book transition). Writing both rows would desync the
    fixed per-sample message count and the shared aggressive_indices between
    streams — noted deviation from the 2-row option.

Extra config keys: marketgpt_root, marketgpt_ckpt (+ optional ctx_msgs=111,
new_block_size=2688, temperature=1.02, top_p=0.98, marketgpt_ticker_id=7
[AAPL in dataset/symbols/custom_symbols.txt]).
Timestamps come from the model's decoded delta_t fields (monotonic clamp,
per-stream clock); interarrivals are generated, not gamma-fit.
"""

import argparse
import os
import sys
import time as pytime
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".40")

import jax
import jax.numpy as jnp
import numpy as onp
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

from lob_impact.core.marketgpt_core import (
    load_marketgpt, MarketGPTSampler,
    lobster14_to_itch18, itch_to_lobster_action, build_aggressive_itch18,
    NA_VAL, PRICE_REL_CLIP, TICKER_AAPL,
    IT_TICKER, IT_TYPE, IT_DIR, IT_PABS, IT_PRICE, IT_FILL,
    IT_DTS, IT_DTNS, IT_TS, IT_TNS,
)

sys.path.insert(0, script_dir)
from qr_scenario import (  # noqa: E402
    TeeLogger, create_experiment_folder,
    create_aggressive_order,
    TIMEs_i, TIMEns_i,
)

MAX_RESAMPLE_ROUNDS = 8

# model is reused across per-day sub-runs (keyed by ckpt path)
_MODEL_CACHE: Dict[str, tuple] = {}


def _get_sampler(cfg: Dict[str, Any]):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = cfg['marketgpt_ckpt']
    if ckpt_path not in _MODEL_CACHE:
        _MODEL_CACHE[ckpt_path] = load_marketgpt(
            cfg['marketgpt_root'], ckpt_path, device=device)
    model, vocab, itch_enc = _MODEL_CACHE[ckpt_path]
    return MarketGPTSampler(
        model, vocab, itch_enc, device,
        ctx_msgs=cfg.get('ctx_msgs', 111),
        new_block_size=cfg.get('new_block_size', 2688),
        temperature=cfg.get('temperature', 1.02),
        top_p=cfg.get('top_p', 0.98),
        ticker_id=cfg.get('marketgpt_ticker_id', TICKER_AAPL),
    )


def run_marketgpt_scenario(cfg: Dict[str, Any], save_folder: Path,
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
    ticker_id = cfg.get('marketgpt_ticker_id', TICKER_AAPL)

    total_blocks = num_insertions + num_coolings

    import torch
    sampler = _get_sampler(cfg)

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

        # save conditioning
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)
            msg_to_lobster_format(m_seq_raw_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_message_real_id_{sample_idx}.csv',
                index=False, header=False)
            book_to_lobster_format(b_seq_pv_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_orderbook_real_id_{sample_idx}.csv',
                index=False, header=False)

        # prime the KV cache from the conditioning window (last ctx_msgs msgs);
        # book state BEFORE msg j = b_seq_pv_cond[:, j] (the priming crux: rel
        # prices are recomputed in cents against the pre-message mid)
        cond_np = onp.asarray(m_seq_raw_cond)
        itch_cond = lobster14_to_itch18(
            cond_np, b_seq_pv_cond[:, :n_cond_msgs, :], tick_size, ticker_id)
        torch.manual_seed(int(rng_seed) * 100000 + worker_id * 1000 + batch_idx)
        t_prime = pytime.time()
        sampler.prime(itch_cond)
        print(f"  KV prefill: {sampler.x.shape[1]} tokens in {pytime.time() - t_prime:.1f}s")

        # per-sample clocks start at the end of the conditioning window
        clock_ns = (cond_np[:, -1, TIMEs_i].astype(onp.int64) * 1_000_000_000
                    + cond_np[:, -1, TIMEns_i].astype(onp.int64))

        total_msgs_per_sample = total_blocks * n_gen_msgs + num_insertions
        n_msg_todo = int(total_msgs_per_sample)

        msgs_out = onp.zeros((B, total_msgs_per_sample, 14), dtype=onp.int64)
        books_out = onp.zeros((B, total_msgs_per_sample, n_levels * 4), dtype=onp.int64)
        out_ptr = 0
        n_fallback = 0
        n_resampled = 0
        t_gen = pytime.time()
        # Steady-state probe: the first messages of a batch also pay the one-time KV-cache
        # prime (110 context messages x 24 tokens for the whole batch) plus CUDA warm-up, so
        # the whole-batch rate badly understates a real run of 13k+ messages. Time from
        # message WARMUP_MSGS onward and report that separately.
        WARMUP_MSGS = 10
        t_steady = None

        for block in range(total_blocks):
            for _ in range(n_gen_msgs):
                if out_ptr == WARMUP_MSGS:
                    t_steady = pytime.time()
                l2_np = onp.asarray(l2_vmap(sim_states))
                asks_np = onp.asarray(sim_states.asks)
                bids_np = onp.asarray(sim_states.bids)

                toks = sampler.sample_next()
                dec = sampler.decode_batch(toks)
                act, ok, corrected = itch_to_lobster_action(
                    dec, l2_np, asks_np, bids_np, tick_size, ticker_id, n_msg_todo)
                rounds = 0
                while (~ok).any() and rounds < MAX_RESAMPLE_ROUNDS:
                    # RETRY re-samples the tokens (cache slots are simply
                    # overwritten; refresh_context finalizes them below)
                    toks2 = sampler.sample_next()
                    dec2 = sampler.decode_batch(toks2)
                    act2, ok2, corr2 = itch_to_lobster_action(
                        dec2, l2_np, asks_np, bids_np, tick_size, ticker_id, n_msg_todo)
                    take = (~ok) & ok2
                    for k in act:
                        act[k] = onp.where(take, act2[k], act[k])
                    corrected[take] = corr2[take]
                    n_resampled += int(take.sum())
                    ok |= ok2
                    rounds += 1
                if (~ok).any():
                    # neutral fallback: 1-lot deep LO at the 10th visible level, own side
                    bad = ~ok
                    n_fallback += int(bad.sum())
                    fb_side = onp.full(B, 1 - direction, dtype=onp.int64)
                    deep = onp.where(fb_side == 1, l2_np[:, 2 + 9 * 4], l2_np[:, 0 + 9 * 4])
                    deep = onp.where(deep > 0, deep,
                                     onp.where(fb_side == 1, l2_np[:, 2], l2_np[:, 0]))
                    act['etype'] = onp.where(bad, 1, act['etype'])
                    act['side01'] = onp.where(bad, fb_side, act['side01'])
                    act['size'] = onp.where(bad, 1, act['size'])
                    act['price'] = onp.where(bad, deep, act['price'])
                    act['oid'] = onp.where(bad, n_msg_todo, act['oid'])
                    act['is_r'] = act['is_r'] & ~bad
                    # corrected ITCH A row for the fallback (cache must match sim)
                    mid_c = ((l2_np[:, 0].astype(onp.int64)
                              + l2_np[:, 2].astype(onp.int64)) // 2) // 100
                    for b in onp.nonzero(bad)[0]:
                        corrected[b, :] = NA_VAL
                        corrected[b, IT_TICKER] = ticker_id
                        corrected[b, IT_TYPE] = 1
                        corrected[b, IT_DIR] = 1 - int(fb_side[b])
                        corrected[b, IT_PABS] = int(deep[b]) // 100
                        corrected[b, IT_PRICE] = int(onp.clip(
                            int(deep[b]) // 100 - mid_c[b], -PRICE_REL_CLIP, PRICE_REL_CLIP))
                        corrected[b, IT_FILL] = 1
                        corrected[b, IT_DTS] = 0
                        corrected[b, IT_DTNS] = 1

                # advance clocks with the model's decoded interarrivals
                # (monotonic clamp: at least 1 ns per message)
                dt_s = onp.where(corrected[:, IT_DTS] == NA_VAL, 0, corrected[:, IT_DTS])
                dt_ns_f = onp.where(corrected[:, IT_DTNS] == NA_VAL, 1, corrected[:, IT_DTNS])
                dt_total = onp.maximum(dt_s * 1_000_000_000 + dt_ns_f, 1)
                clock_ns = clock_ns + dt_total
                ts = clock_ns // 1_000_000_000
                tns = clock_ns % 1_000_000_000
                corrected[:, IT_DTS] = dt_total // 1_000_000_000
                corrected[:, IT_DTNS] = dt_total % 1_000_000_000
                corrected[:, IT_TS] = ts
                corrected[:, IT_TNS] = tns

                # refresh the KV cache with what will actually be applied
                sampler.refresh_context(sampler.encode_batch(corrected))

                # apply to JAX-LOB: primary op (for R: the cancel of the old order)
                data1 = act['side01'] * 2 - 1
                msgs = onp.stack([
                    act['etype'], data1, act['size'], act['price'],
                    act['oid'],
                    onp.full(B, -88, dtype=onp.int64),
                    ts, tns,
                ], axis=1).astype(onp.int32)
                sim_states = step_vmap(sim_states, jnp.asarray(msgs))

                # second op for R streams (the new add); no-op (type 0) elsewhere
                if act['is_r'].any():
                    r = act['is_r']
                    msgs2 = onp.zeros((B, 8), dtype=onp.int64)
                    msgs2[:, 6], msgs2[:, 7] = ts, tns
                    msgs2[r, 0] = 1
                    msgs2[r, 1] = data1[r]
                    msgs2[r, 2] = act['size2'][r]
                    msgs2[r, 3] = act['price2'][r]
                    msgs2[r, 4] = n_msg_todo
                    msgs2[r, 5] = -88
                    sim_states = step_vmap(sim_states, jnp.asarray(msgs2.astype(onp.int32)))
                l2_new = onp.asarray(l2_vmap(sim_states))

                # written row: for R this is the ADD leg (cancel folded, see header)
                w_etype = onp.where(act['is_r'], 1, act['etype'])
                w_size = onp.where(act['is_r'], act['size2'], act['size'])
                w_price = onp.where(act['is_r'], act['price2'], act['price'])
                mid = ((l2_new[:, 0] + l2_new[:, 2]) // 2 // tick_size) * tick_size
                rel = onp.where(mid > 0, (w_price - mid) // tick_size, 0)
                msgs_out[:, out_ptr, :] = onp.stack([
                    onp.full(B, n_msg_todo, dtype=onp.int64), w_etype, act['side01'],
                    w_price, rel, onp.abs(w_size),
                    dt_total // 1_000_000_000, dt_total % 1_000_000_000,
                    ts, tns,
                    onp.zeros(B, onp.int64), onp.zeros(B, onp.int64),
                    onp.zeros(B, onp.int64), onp.zeros(B, onp.int64),
                ], axis=1)
                books_out[:, out_ptr, :] = l2_new
                out_ptr += 1
                n_msg_todo -= 1

            if block < num_insertions:
                # book state BEFORE the insertion (for the model's E message)
                l2_pre = onp.asarray(l2_vmap(sim_states))
                asks_pre = onp.asarray(sim_states.asks)
                bids_pre = onp.asarray(sim_states.bids)

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

                # the model MUST see the metaorder: encode its ITCH E row and
                # refresh the cache (append — nothing was sampled here)
                itch_aggr = build_aggressive_itch18(
                    onp.asarray(msg_dec), l2_pre, asks_pre, bids_pre, ticker_id)
                sampler.refresh_context(sampler.encode_batch(itch_aggr))

        elapsed = pytime.time() - t_gen
        print(f"  fallback messages: {n_fallback} "
              f"({100.0 * n_fallback / (B * total_msgs_per_sample):.2f}%), "
              f"resampled-in: {n_resampled}")
        print(f"  throughput: {B * total_msgs_per_sample / elapsed:.1f} msgs/s "
              f"({elapsed:.0f}s for {B}x{total_msgs_per_sample})")
        if t_steady is not None and total_msgs_per_sample > WARMUP_MSGS:
            el_s = pytime.time() - t_steady
            n_s = B * (total_msgs_per_sample - WARMUP_MSGS)
            rate = n_s / el_s
            print(f"  throughput STEADY (excl. prime+warmup): {rate:.1f} msgs/s "
                  f"({el_s:.0f}s for {n_s} msgs) -> a 100-insertion run of "
                  f"{100 * cfg.get('n_gen_msgs', 0)} msgs x {B} samples would take "
                  f"{100 * cfg.get('n_gen_msgs', 0) * B / max(rate, 1e-9) / 3600:.1f} h/batch")

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
    parser = argparse.ArgumentParser(description="MarketGPT Aggressive Scenario")
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
    print(f"MarketGPT Aggressive Scenario — worker {worker_id}/{num_workers}")
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
                run_marketgpt_scenario(cfg_d, save_folder, worker_id=worker_id, num_workers=num_workers)
        else:
            run_marketgpt_scenario(cfg, save_folder, worker_id=worker_id, num_workers=num_workers)

        print(f"\nWorker {worker_id} completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    finally:
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
