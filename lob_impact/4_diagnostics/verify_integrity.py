#!/usr/bin/env python3
"""
Action 4 — end-to-end integrity re-audit of generated impact samples (grid_v2).

Per experiment (model x shape x dir), over a subsample of trajectories, checks:
  1. context->generation boundary: mid/spread jump between last data_cond book row
     and first data_gen book row (ticks)
  2. insertion rows: event_type==4 at every aggressive index, executed size vs
     configured order_volume vs touch depth of the PREVIOUS book row (clip evidence),
     level-walk check (did deeper levels get eaten), direction sanity
  3. mechanical vs drift decomposition of the build-up:
     mech  = sum_k [mid(i_k) - mid(i_k - 1)]      (jump AT the insertion row)
     drift = sum_k [mid(i_k - 1) - mid(i_{k-1})]  (model-generated move between insertions)
     final = mid(last) - mid(0);  final ~= mech + drift + tail
  4. book sanity: crossed/sentinel rows, spread first-10% vs last-10%, L10 ask+bid
     total depth first vs last (depletion), touch-refill after insertions
  5. message->book consistency: fraction of exec msgs whose price equals the previous
     row's opposite touch (book applied what the message says)

Output: one CSV row per sample + printed per-experiment aggregate table.

  python 4_diagnostics/verify_integrity.py --grid /lus/.../lob_impact_grid_v2 \
      --stock EA --models Mamba3,Mamba3_4k,S5_4k,Historic,Heuristic,Hawkes,CST \
      --shapes beta --n_samples 30 --out_dir 4_diagnostics/results/verify_<ts>
"""
import os, re, csv, glob, json, argparse
import numpy as np

TICK = 100
EXEC = 4
SENT = 2147483647


def read_csv(f):
    return np.array([r for r in csv.reader(open(f)) if r], dtype=float)


def discover_exp(side_dir):
    """exp_* subdir or exact-folder (slice-merged 4k) layout."""
    e = sorted(glob.glob(os.path.join(side_dir, 'exp_*')))
    if e:
        return e[0]
    if os.path.isdir(os.path.join(side_dir, 'data_gen')):
        return side_dir
    return None


def aggr_for(exp, date, L):
    f = os.path.join(exp, f'aggressive_indices_{date}.csv')
    if not os.path.exists(f):
        f = os.path.join(exp, 'aggressive_indices.csv')
    if not os.path.exists(f):
        return np.array([], dtype=int)
    a = np.loadtxt(f, dtype=int, ndmin=1)
    return a[a < L]


def book_mid_spread(b):
    ask_p, bid_p = b[:, 0], b[:, 2]
    bad = (ask_p >= SENT) | (bid_p >= SENT) | (ask_p <= 0) | (bid_p <= 0)
    mid = (ask_p + bid_p) / 2.0
    spr = (ask_p - bid_p) / TICK
    mid[bad] = np.nan
    spr[bad] = np.nan
    return mid, spr, bad


def l10_depth(b):
    """total visible volume per side across all levels present (40-col: p,v alternating a/b)."""
    nlev = b.shape[1] // 4
    av = b[:, [4 * l + 1 for l in range(nlev)]].sum(1)
    bv = b[:, [4 * l + 3 for l in range(nlev)]].sum(1)
    return av, bv


def check_sample(exp, gen_msg_f, order_volume, direction):
    r = {}
    base = os.path.basename(gen_msg_f)
    m_date = re.search(r'_(\d{4}-\d{2}-\d{2})_', base)
    date = m_date.group(1) if m_date else ''
    gen_book_f = gen_msg_f.replace('message', 'orderbook')
    cond_msg_f = os.path.join(exp, 'data_cond', re.sub(r'_gen_id_\d+', '', base))
    cond_book_f = cond_msg_f.replace('message', 'orderbook')
    if not os.path.exists(gen_book_f):
        return None
    msg = read_csv(gen_msg_f)
    book = read_csv(gen_book_f)
    L = min(len(msg), len(book))
    msg, book = msg[:L], book[:L]
    mid, spr, bad = book_mid_spread(book)
    r['sample'] = base
    r['date'] = date
    r['L'] = L
    r['bad_rows'] = int(bad.sum())
    ask_p, ask_v, bid_p, bid_v = book[:, 0], book[:, 1], book[:, 2], book[:, 3]
    r['crossed_rows'] = int(((ask_p <= bid_p) & ~bad).sum())

    # 1. boundary continuity
    r['boundary_jump_ticks'] = np.nan
    r['boundary_spread_cond'] = np.nan
    if os.path.exists(cond_book_f):
        cb = read_csv(cond_book_f)
        cmid, cspr, cbad = book_mid_spread(cb)
        if len(cmid) and not cbad[-1] and not bad[0]:
            r['boundary_jump_ticks'] = float((mid[0] - cmid[-1]) / TICK)
            r['boundary_spread_cond'] = float(cspr[-1])

    # 2 + 3. insertions
    ai = aggr_for(exp, date, L)
    ai = ai[ai > 0]
    r['n_insertions_found'] = len(ai)
    if len(ai):
        ev_ok = (msg[ai, 1] == EXEC)
        r['insertion_event4_frac'] = float(ev_ok.mean())
        sizes = msg[ai, 3]
        r['agg_size_mean'] = float(sizes.mean())
        r['agg_size_eq_vol_frac'] = float((sizes == order_volume).mean())
        prev_touch_v = ask_v[ai - 1] if direction == 0 else bid_v[ai - 1]
        prev_touch_p = ask_p[ai - 1] if direction == 0 else bid_p[ai - 1]
        with np.errstate(invalid='ignore'):
            r['agg_size_eq_prev_touch_frac'] = float((sizes == prev_touch_v).mean())
            r['agg_size_clipped_frac'] = float((sizes < order_volume).mean())
            r['prev_touch_depth_mean'] = float(np.nanmean(prev_touch_v))
            # full-wipe: touch level fully consumed -> price at touch changed after insertion
            cur_touch_p = ask_p[ai] if direction == 0 else bid_p[ai]
            r['touch_level_wiped_frac'] = float(np.nanmean((cur_touch_p != prev_touch_p) & ev_ok))
            r['agg_price_eq_prev_touch_frac'] = float(np.nanmean(msg[ai, 4] == prev_touch_p))
        # mechanical vs drift
        mech = mid[ai] - mid[ai - 1]
        segs = np.concatenate([[mid[ai[0] - 1] - mid[0]],
                               mid[ai[1:] - 1] - mid[ai[:-1]]]) if len(ai) > 1 else \
               np.array([mid[ai[0] - 1] - mid[0]])
        r['mech_sum_ticks'] = float(np.nansum(mech) / TICK)
        r['drift_sum_ticks'] = float(np.nansum(segs) / TICK)
        r['mech_per_insertion_ticks'] = float(np.nanmean(mech) / TICK)
        # refill: does the touch price come back within the next window? relaxation proxy
        back = []
        for k, i in enumerate(ai[:-1]):
            nxt = ai[k + 1] - 1
            if not np.isnan(mid[i]) and not np.isnan(mid[nxt]):
                back.append((mid[nxt] - mid[i]) / TICK)
        r['post_insertion_reversion_ticks'] = float(np.mean(back)) if back else np.nan
    if not np.isnan(mid[0]):
        end = mid[-1] if not np.isnan(mid[-1]) else np.nanmean(mid[-20:])
        r['final_impact_ticks'] = float((end - mid[0]) / TICK)
    else:
        r['final_impact_ticks'] = np.nan

    # 4. book sanity
    n10 = max(L // 10, 1)
    r['spread_first10'] = float(np.nanmean(spr[:n10]))
    r['spread_last10'] = float(np.nanmean(spr[-n10:]))
    av, bv = l10_depth(book)
    r['askdepth_first10'] = float(np.nanmean(av[:n10]))
    r['askdepth_last10'] = float(np.nanmean(av[-n10:]))
    r['biddepth_first10'] = float(np.nanmean(bv[:n10]))
    r['biddepth_last10'] = float(np.nanmean(bv[-n10:]))

    # 5. message->book consistency: exec price == previous row opposite touch
    ex = np.where(msg[:, 1] == EXEC)[0]
    ex = ex[ex > 0]
    if len(ex):
        p = msg[ex, 4]
        d = msg[ex, 5]  # LOBSTER: direction of the RESTING order (-1 sell side hit by buy)
        prev_ask, prev_bid = ask_p[ex - 1], bid_p[ex - 1]
        at_touch = ((d == -1) & (p == prev_ask)) | ((d == 1) & (p == prev_bid))
        r['exec_at_prev_touch_frac'] = float(at_touch.mean())
        r['n_exec'] = int(len(ex))
    return r


AGG_COLS = ['boundary_jump_ticks', 'insertion_event4_frac', 'agg_size_mean',
            'agg_size_eq_vol_frac', 'agg_size_eq_prev_touch_frac', 'agg_size_clipped_frac',
            'prev_touch_depth_mean', 'touch_level_wiped_frac', 'agg_price_eq_prev_touch_frac',
            'mech_sum_ticks', 'drift_sum_ticks', 'mech_per_insertion_ticks',
            'post_insertion_reversion_ticks', 'final_impact_ticks',
            'spread_first10', 'spread_last10', 'askdepth_first10', 'askdepth_last10',
            'biddepth_first10', 'biddepth_last10', 'exec_at_prev_touch_frac',
            'bad_rows', 'crossed_rows']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Mamba3,Mamba3_4k,S5_4k,Historic,Heuristic,Hawkes,CST')
    ap.add_argument('--shapes', default='beta')
    ap.add_argument('--dirs', default='buy,sell')
    ap.add_argument('--n_samples', type=int, default=30)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for model in args.models.split(','):
        for shape in args.shapes.split(','):
            for side in args.dirs.split(','):
                d = os.path.join(args.grid, f'{args.stock}-{model}-{shape}', side)
                exp = discover_exp(d) if os.path.isdir(d) else None
                if not exp:
                    print(f'-- {model}/{shape}/{side}: NOT FOUND')
                    continue
                import yaml
                cfg = yaml.safe_load(open(os.path.join(exp, 'config.yaml'))) \
                    if os.path.exists(os.path.join(exp, 'config.yaml')) else {}
                vol = cfg.get('order_volume', -1)
                direction = 0 if side == 'buy' else 1
                gens = sorted(glob.glob(os.path.join(exp, 'data_gen', '*message*gen*.csv')))
                if not gens:
                    print(f'-- {model}/{shape}/{side}: no gen files')
                    continue
                step = max(len(gens) // args.n_samples, 1)
                rows = []
                for f in gens[::step][:args.n_samples]:
                    try:
                        rr = check_sample(exp, f, vol, direction)
                        if rr:
                            rows.append(rr)
                    except Exception as e:
                        print(f'   ERR {os.path.basename(f)}: {e}')
                if not rows:
                    continue
                key = f'{model}-{shape}-{side}'
                cols = sorted({c for r in rows for c in r})
                with open(os.path.join(args.out_dir, f'per_sample_{key}.csv'), 'w', newline='') as fo:
                    w = csv.DictWriter(fo, fieldnames=cols)
                    w.writeheader()
                    w.writerows(rows)
                agg = {}
                for c in AGG_COLS:
                    v = np.array([r.get(c, np.nan) for r in rows], dtype=float)
                    if np.all(np.isnan(v)):
                        continue
                    agg[c] = round(float(np.nanmean(v)), 4)
                agg['n_samples_checked'] = len(rows)
                agg['order_volume_cfg'] = vol
                summary[key] = agg
                print(f'== {key} (n={len(rows)}, vol_cfg={vol})')
                for c in AGG_COLS:
                    if c in agg:
                        print(f'   {c:38s} {agg[c]:>12}')
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as fo:
        json.dump(summary, fo, indent=1)
    print(f'\nwrote {args.out_dir}/summary.json')


if __name__ == '__main__':
    main()
