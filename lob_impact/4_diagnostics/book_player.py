#!/usr/bin/env python3
"""
Interactive order-book PLAYER — the established notebook style (ipywidgets + FigureWidget).

Reproduces the user's canonical players (20.lob_real_vs_gen / 23.lob_buy_sell_combined)
on the NEW-pipeline grid. Pick a sample from the dropdown, watch the mid-price evolve,
and step through messages with ◀ ▶ (or the slider / "→ Next Aggr"). Each step shows the
book as a before/after state diff:

  col1  BOOK t-1  — bar ladder, all GRAY (asks up, bids down)
  col2  BOOK t    — same ladder, every level coloured by the change vs t-1:
                    RED = volume ↑ (added),  BLUE = volume ↓ (executed/removed),  gray = same
  col3  MID-PRICE — full mid path with a step cursor, aggressive-order ▲ marks, junction line

The decoded message that produced step t is shown below (badged red if it is an injected
aggressive order). cond+gen are stacked; junction = conditioning length; aggressive indices
are gen-relative -> absolute = junction + idx (matches 23.lob_buy_sell_combined).

Usage (in JupyterLab, lobs5 kernel — has plotly 5.22 + ipywidgets 8.1):
    from book_player import lob_player, GRID
    lob_player(GRID, exp='EA-Mamba3-beta', side='buy')

Run as a script to write a STATIC one-step snapshot for a quick eyeball (no kernel):
    python book_player.py --exp EA-Mamba3-beta --side buy --step aggr
"""
import os, glob, re, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, '..', '3_scenarios', 'results', 'grid')

SENT = 2147483647
EV = {1: 'LIMIT', 2: 'CANCEL', 3: 'DELETE', 4: 'EXECUTE', 5: 'HID-EXEC', 6: 'CROSS'}
EVCOLOR = {1: '#2F5DA3', 2: '#7F8C8D', 3: '#C0392B', 4: '#111111', 5: '#8E44AD', 6: '#E67E22'}
C_GRAY, C_UP, C_DOWN = '#BBBBBB', '#C0392B', '#2F5DA3'   # t-1 / vol↑ / vol↓  (user's convention)


# ----------------------------------------------------------------------- data
def _csv(f):
    return np.loadtxt(f, delimiter=',', ndmin=2)


def _tick(exp_dir):
    p = os.path.join(exp_dir, 'config.yaml')
    if os.path.exists(p):
        m = re.search(r'^tick_size\s*:\s*([0-9]+)', open(p).read(), re.M)
        if m:
            return int(m.group(1))
    return 100


def discover(grid):
    """{exp: {side: latest_exp_dir}} for every <stock>-<model>-<shape> leaf."""
    out = {}
    for d in sorted(glob.glob(os.path.join(grid, '*-*-*'))):
        if not os.path.isdir(d):
            continue
        sides = {}
        for side in ('buy', 'sell'):
            e = sorted(glob.glob(os.path.join(d, side, 'exp_*')))
            if e:
                sides[side] = e[-1]
            elif os.path.isdir(os.path.join(d, side, 'data_gen')):
                # slice-merged exact-folder layout (4k fleets): no exp_* level
                sides[side] = os.path.join(d, side)
        if sides:
            out[os.path.basename(d)] = sides
    return out


def list_samples(exp_dir):
    """[(ticker, date, sid, ob_file), ...] from data_gen."""
    res = []
    for ob in sorted(glob.glob(os.path.join(exp_dir, 'data_gen', '*orderbook*gen*.csv'))):
        m = re.search(r'([A-Z]+)_(\d{4}-\d{2}-\d{2})_orderbook_real_id_(\d+)_gen', os.path.basename(ob))
        if m:
            res.append((m.group(1), m.group(2), int(m.group(3)), ob))
    return res


def pick_across_days(samples, n):
    """Round-robin across distinct days so a small subset spans as many days as possible."""
    by_day = {}
    for s in samples:
        by_day.setdefault(s[1], []).append(s)
    days = sorted(by_day)
    out = []
    while len(out) < n and any(by_day[d] for d in days):
        for d in days:
            if by_day[d]:
                out.append(by_day[d].pop(0))
                if len(out) >= n:
                    break
    return out


def load_sample(exp_dir, ticker, date, sid, ob_file):
    """Stack cond+gen books/msgs. Returns (books, msgs, junction, aggr_set, tick)."""
    gb, gm = _csv(ob_file), _csv(ob_file.replace('orderbook', 'message'))
    cbf = os.path.join(exp_dir, 'data_cond', f'{ticker}_{date}_orderbook_real_id_{sid}.csv')
    cmf = os.path.join(exp_dir, 'data_cond', f'{ticker}_{date}_message_real_id_{sid}.csv')
    if os.path.exists(cbf) and os.path.exists(cmf):
        cb, cm = _csv(cbf), _csv(cmf)
        if cb.shape[0] == cm.shape[0] + 1:      # cond book carries an extra initial-state row
            cb = cb[1:]
        n = min(cb.shape[0], cm.shape[0])
        cb, cm = cb[:n], cm[:n]
        junction = cm.shape[0]
        books, msgs = np.vstack([cb, gb]), np.vstack([cm, gm])
    else:
        junction, books, msgs = 0, gb, gm
    n = min(books.shape[0], msgs.shape[0])
    books, msgs = books[:n], msgs[:n]
    aggr = set()
    # per-day grids: mb varies by day, so prefer the day-matched indices file over the
    # summary aggressive_indices.csv (which only reflects the last generated day)
    af = os.path.join(exp_dir, f'aggressive_indices_{date}.csv')
    if not os.path.exists(af):
        af = os.path.join(exp_dir, 'aggressive_indices.csv')
    if os.path.exists(af):
        a = np.loadtxt(af, ndmin=1).astype(int)
        aggr = {int(junction + i) for i in a if 0 <= junction + i < n}
    return books, msgs, junction, aggr, _tick(exp_dir)


# ------------------------------------------------------------------- book math
def book_sides(row):
    """40-int L10 row -> (ask{price:vol}, bid{price:vol}); sentinels/empties dropped."""
    ask, bid = {}, {}
    ap, av, bp, bv = row[0::4], row[1::4], row[2::4], row[3::4]
    for p, v in zip(ap, av):
        p, v = int(p), int(v)
        if 0 < p < SENT and v > 0:
            ask[p] = ask.get(p, 0) + v
    for p, v in zip(bp, bv):
        p, v = int(p), int(v)
        if 0 < p < SENT and v > 0:
            bid[p] = bid.get(p, 0) + v
    return ask, bid


def midprice(row):
    ask, bid = book_sides(row)
    if not ask or not bid:
        return np.nan
    return (min(ask) + max(bid)) / 2.0


def ladder(row, prev_row, tick, diff):
    """Bars for one book panel: asks up (+vol), bids down (-vol), x = price/tick.
    If diff, colour each bar vs prev_row (RED vol↑ / BLUE vol↓ / gray same); else all gray."""
    ask, bid = book_sides(row)
    p_ask, p_bid = (book_sides(prev_row) if prev_row is not None else ({}, {}))
    xs, ys, cols, txt = [], [], [], []

    def _col(v, q):
        return C_GRAY if (not diff or v == q) else (C_UP if v > q else C_DOWN)

    for p, v in sorted(ask.items()):                 # asks: positive
        xs.append(p / tick); ys.append(v); txt.append(f'ask {v}')
        cols.append(_col(v, p_ask.get(p, 0)))
    for p, v in sorted(bid.items()):                 # bids: negative
        xs.append(p / tick); ys.append(-v); txt.append(f'bid {v}')
        cols.append(_col(v, p_bid.get(p, 0)))
    return xs, ys, cols, txt


def decode_msg(m, tick):
    et, sz, pr, dr = int(m[1]), int(m[3]), m[4], int(m[5])
    side = 'BUY/bid' if dr > 0 else 'SELL/ask'
    return dict(event=EV.get(et, f'et{et}'), color=EVCOLOR.get(et, '#555'),
                side=side, size=sz, price=pr / tick, oid=int(m[2]), t=m[0])


# ----------------------------------------------------------------- the player
def lob_player(grid=GRID, exp='EA-Mamba3-beta', side='buy', max_samples=16):
    """Build and display the interactive FigureWidget player in a notebook."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import ipywidgets as W
    from IPython.display import display

    grid = os.path.abspath(grid)
    exp_dir = discover(grid).get(exp, {}).get(side)
    if not exp_dir:
        raise SystemExit(f'no {exp}/{side} under {grid}')
    samples = pick_across_days(list_samples(exp_dir), max_samples)
    if not samples:
        raise SystemExit('no samples')

    cache = {}

    def get(idx):
        if idx not in cache:
            tk, dt, sid, ob = samples[idx]
            books, msgs, junc, aggr, tick = load_sample(exp_dir, tk, dt, sid, ob)
            mids = np.array([midprice(r) for r in books])
            cache[idx] = dict(books=books, msgs=msgs, junc=junc, aggr=aggr,
                              tick=tick, mids=mids, label=f'{dt} · id{sid}', tk=tk)
        return cache[idx]

    fig = make_subplots(rows=1, cols=3, column_widths=[0.3, 0.3, 0.4],
                        subplot_titles=['Было — Book t-1', 'Стало — Book t', 'Mid-price'])
    fig.add_trace(go.Bar(x=[], y=[], marker_color=C_GRAY, width=0.4, name='t-1'), 1, 1)
    fig.add_trace(go.Bar(x=[], y=[], marker_color=[], width=0.4, name='t'), 1, 2)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', line=dict(color='#1a1a1a', width=1)), 1, 3)
    fig.add_trace(go.Scatter(x=[], y=[], mode='markers',
                             marker=dict(color='#2E7D52', size=6, symbol='triangle-up')), 1, 3)
    figw = go.FigureWidget(fig)
    figw.update_layout(width=1280, height=420, template='plotly_white', showlegend=False,
                       margin=dict(l=40, r=20, t=40, b=35), bargap=0.1)
    figw.update_xaxes(title_text='price', row=1, col=1)
    figw.update_xaxes(title_text='price', row=1, col=2)
    figw.update_yaxes(title_text='signed qty (ask + / bid −)', row=1, col=1)
    figw.update_xaxes(title_text='step', row=1, col=3)
    figw.update_yaxes(title_text='mid', row=1, col=3)
    # mid cursor + junction (shapes on x3)
    figw.add_shape(type='line', xref='x3', yref='paper', x0=0, x1=0, y0=0, y1=1,
                   line=dict(color='#888', width=1, dash='dash'))
    figw.add_shape(type='line', xref='x3', yref='paper', x0=0, x1=0, y0=0, y1=1,
                   line=dict(color='#C0392B', width=2))

    dd = W.Dropdown(options=[(f'#{i} · {s[1]} id{s[2]}', i) for i, s in enumerate(samples)],
                    description='Sample:')
    sl = W.IntSlider(min=0, max=1, value=0, description='t:', continuous_update=True,
                     layout=W.Layout(width='420px'))
    b_prev = W.Button(description='◀', layout=W.Layout(width='46px'))
    b_next = W.Button(description='▶', layout=W.Layout(width='46px'))
    b_junc = W.Button(description='→ Junction', layout=W.Layout(width='110px'))
    b_aggr = W.Button(description='→ Next Aggr', layout=W.Layout(width='110px'))
    info = W.HTML()

    def render(*_):
        d = get(dd.value)
        t = sl.value
        books, msgs, tick = d['books'], d['msgs'], d['tick']
        prev = books[t - 1] if t > 0 else None
        x0, y0, c0, _ = ladder(books[t - 1] if t > 0 else books[t], None, tick, False)
        x1, y1, c1, _ = ladder(books[t], prev, tick, True)
        m = decode_msg(msgs[t], tick)
        agg = t in d['aggr']
        with figw.batch_update():
            figw.data[0].x, figw.data[0].y = x0, y0
            figw.data[1].x, figw.data[1].y, figw.data[1].marker.color = x1, y1, c1
            # shared price (x) + qty (y) ranges so the two panels align and a removed/executed
            # level shows as a visible GAP in "стало" while still present in "было"
            allx = (x0 or []) + (x1 or [])
            if allx:
                xr = [min(allx) - 0.6, max(allx) + 0.6]
                figw.layout.xaxis.range = xr
                figw.layout.xaxis2.range = xr
            ym = max([abs(v) for v in (y0 + y1)] or [1]) * 1.12
            figw.layout.yaxis.range = [-ym, ym]
            figw.layout.yaxis2.range = [-ym, ym]
            xs = np.arange(len(d['mids']))
            figw.data[2].x, figw.data[2].y = xs, d['mids']
            av = sorted(i for i in d['aggr'] if i < len(d['mids']))
            figw.data[3].x, figw.data[3].y = av, d['mids'][av] if av else []
            figw.layout.shapes[0].x0 = figw.layout.shapes[0].x1 = t
            figw.layout.shapes[1].x0 = figw.layout.shapes[1].x1 = d['junc']
            figw.layout.annotations[0].text = f'Было — Book t={t-1}'
            figw.layout.annotations[1].text = f'Стало — Book t={t}'
        badge = ("<span style='background:#E67E22;color:#fff;padding:2px 8px;border-radius:5px;"
                 "font-weight:700'>◆ AGGRESSIVE</span>") if agg else ''
        info.value = (
            f"<div style='font:13px monospace;padding:6px 0'>"
            f"<b>step {t}/{len(books)-1}</b> &nbsp; "
            f"<span style='background:{m['color']};color:#fff;padding:2px 8px;border-radius:5px;"
            f"font-weight:700'>{m['event']}</span> &nbsp; <b>{m['side']}</b> &nbsp; "
            f"{m['size']} @ <b>{m['price']:.2f}</b> &nbsp; "
            f"<span style='color:#888'>order {m['oid']} · t={m['t']}</span> &nbsp; {badge}</div>")

    def on_sample(*_):
        d = get(dd.value)
        sl.max = len(d['books']) - 1
        sl.value = min(d['junc'], sl.max)     # start at the cond/gen junction
        render()

    def step(delta):
        sl.value = int(np.clip(sl.value + delta, 0, sl.max))

    def next_aggr(*_):
        d = get(dd.value)
        nxt = [i for i in sorted(d['aggr']) if i > sl.value]
        sl.value = nxt[0] if nxt else sl.value

    dd.observe(on_sample, names='value')
    sl.observe(render, names='value')
    b_prev.on_click(lambda _: step(-1))
    b_next.on_click(lambda _: step(1))
    b_junc.on_click(lambda _: setattr(sl, 'value', get(dd.value)['junc']))
    b_aggr.on_click(next_aggr)

    display(W.HTML(f"<h3 style='margin:4px 0'>📖 Order-book player — {exp} · {side}</h3>"
                   "<div style='font:12px sans-serif;color:#666'>◀ ▶ step · "
                   "<span style='color:#C0392B'>red = volume ↑ (added)</span> · "
                   "<span style='color:#2F5DA3'>blue = volume ↓ (executed/removed)</span></div>"))
    display(W.HBox([dd, b_prev, b_next, b_junc, b_aggr]))
    display(sl, figw, info)
    on_sample()
    return figw


# ----------------------------------------------------------- static smoke test
def _snapshot(grid, exp, side, which='aggr', out=None):
    """Headless: build the figure at one step and write a static HTML to eyeball (no kernel)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    grid = os.path.abspath(grid)
    exp_dir = discover(grid)[exp][side]
    tk, dt, sid, ob = list_samples(exp_dir)[0]
    books, msgs, junc, aggr, tick = load_sample(exp_dir, tk, dt, sid, ob)
    t = (sorted(aggr)[0] if (which == 'aggr' and aggr) else int(which) if str(which).isdigit()
         else junc + 1)
    mids = np.array([midprice(r) for r in books])
    x0, y0, c0, _ = ladder(books[t - 1], None, tick, False)
    x1, y1, c1, tx = ladder(books[t], books[t - 1], tick, True)
    m = decode_msg(msgs[t], tick)
    print(f'{exp}/{side} {dt} id{sid}: {len(books)} steps, junction={junc}, '
          f'{len(aggr)} aggr; showing step {t} ({"AGGR" if t in aggr else "normal"})')
    print(f'  message: {m["event"]} {m["side"]} {m["size"]} @ {m["price"]:.2f}')
    print(f'  before mid={midprice(books[t-1]):.1f}  after mid={midprice(books[t]):.1f}  '
          f'(tick={tick})')
    print(f'  changed bars (after): ' +
          ', '.join(f'{xx:.0f}:{("↑" if cc==C_UP else "↓" if cc==C_DOWN else "=")}'
                    for xx, cc in zip(x1, c1) if cc != C_GRAY) or '  (none)')
    fig = make_subplots(rows=1, cols=3, column_widths=[0.3, 0.3, 0.4],
                        subplot_titles=[f'Было t={t-1}', f'Стало t={t}', 'Mid-price'])
    fig.add_trace(go.Bar(x=x0, y=y0, marker_color=C_GRAY, width=0.4), 1, 1)
    fig.add_trace(go.Bar(x=x1, y=y1, marker_color=c1, width=0.4), 1, 2)
    fig.add_trace(go.Scatter(x=np.arange(len(mids)), y=mids, mode='lines',
                             line=dict(color='#1a1a1a', width=1)), 1, 3)
    av = sorted(i for i in aggr if i < len(mids))
    fig.add_trace(go.Scatter(x=av, y=mids[av], mode='markers',
                             marker=dict(color='#2E7D52', size=6, symbol='triangle-up')), 1, 3)
    fig.add_vline(x=t, line=dict(color='#888', width=1, dash='dash'), row=1, col=3)
    allx = x0 + x1
    if allx:
        xr = [min(allx) - 0.6, max(allx) + 0.6]
        fig.update_xaxes(range=xr, row=1, col=1); fig.update_xaxes(range=xr, row=1, col=2)
    ym = max([abs(v) for v in (y0 + y1)] or [1]) * 1.12
    fig.update_yaxes(range=[-ym, ym], row=1, col=1); fig.update_yaxes(range=[-ym, ym], row=1, col=2)
    fig.update_layout(width=1280, height=420, template='plotly_white', showlegend=False,
                      margin=dict(l=40, r=20, t=40, b=35), title=f'{exp} · {side} · step {t}')
    out = out or os.path.join(HERE, 'results', 'book_player', f'snapshot_{exp}_{side}.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out, include_plotlyjs='cdn')
    print(f'  wrote {out} ({os.path.getsize(out)/1e3:.0f} KB)')


# --------------------------------------------------- standalone HTML (no kernel)
def _delta_encode(books):
    """books: (N,40) int -> (book0:list, deltas:list[[col,val],...]) vs previous row."""
    b = books.astype(int)
    book0 = b[0].tolist()
    deltas = [[]]
    for k in range(1, len(b)):
        ch = np.nonzero(b[k] != b[k - 1])[0]
        deltas.append([[int(c), int(b[k][c])] for c in ch])
    return book0, deltas


def export_html(grid=GRID, exp='EA-Mamba3-beta', side='buy', n_samples=6,
                max_steps=0, out=None):
    """Self-contained interactive HTML player (Plotly.js via CDN, data embedded, delta-coded).
    Same canonical look as lob_player but needs no Jupyter kernel — just open the file."""
    import json
    grid = os.path.abspath(grid)
    exp_dir = discover(grid).get(exp, {}).get(side)
    if not exp_dir:
        raise SystemExit(f'no {exp}/{side} under {grid}')
    # per-day child (the order_volume that actually executes that day, =p50) from per_day_params —
    # this is the real per-day stat (NOT the stale config order_volume=75).
    pdp = {}  # day -> (child=p50 volume, mb=msgs_btw)
    cfgp = os.path.join(exp_dir, 'config.yaml')
    if os.path.exists(cfgp):
        m = re.search(r'^per_day_params\s*:\s*(\S+)', open(cfgp).read(), re.M)
        if m and os.path.exists(m.group(1)):
            for r in csv.DictReader(open(m.group(1))):
                if abs(float(r.get('mult', 1)) - 1.0) < 1e-9:
                    pdp[r['day']] = (int(float(r['child'])), int(float(r['mb'])))
    samples = []
    for tk, dt, sid, ob in pick_across_days(list_samples(exp_dir), n_samples):
        books, msgs, junc, aggr, tick = load_sample(exp_dir, tk, dt, sid, ob)
        if max_steps and max_steps < len(books):
            books, msgs = books[:max_steps], msgs[:max_steps]
            aggr = {i for i in aggr if i < max_steps}
        book0, deltas = _delta_encode(books)
        mids = [None if not np.isfinite(midprice(r)) else round(float(midprice(r)) / tick, 4)
                for r in books]
        # reference mid = mid just before the first aggressive insertion (else at the cond/gen junction)
        a0 = (min(aggr) if aggr else junc)
        ref = next((mids[i] for i in range(a0 - 1, -1, -1) if mids[i] is not None), None)
        child, mb = pdp.get(dt, (None, None))
        # compact message: [event_type, direction, size, price_ticks, order_id, time]
        cmsgs = [[int(m[1]), int(m[5]), int(m[3]), int(m[4]), int(m[2]), round(float(m[0]), 9)]
                 for m in msgs]
        # Find the aggressive executions PER SAMPLE directly (do NOT trust aggressive_indices.csv: in
        # per-day mode it is ONE file for the whole experiment, but each day has a different mb, so its
        # positions are only right for the one day that wrote it last). The metaorder fills are the
        # et=4 messages of size==child at ~mb spacing in the gen region (step > junction).
        aggr2 = []
        if child and mb and mb > 0:
            # The aggressive fills are et=4 of size≈child, ~mb apart. Their position in the OUTPUT
            # array DRIFTS (each market order adds its own execution messages), so a fixed junc+i*mb
            # schedule misses the later ones. Instead track relative to the LAST found insertion:
            # next expected ≈ last + mb. Prefer an exact size==child fill, else a partial (size<=child).
            W, e = 8, junc + mb
            while e < len(cmsgs):
                lo, hi = max(junc, e - W), min(len(cmsgs), e + W + 1)
                cands = [j for j in range(lo, hi) if cmsgs[j][0] == 4 and 0 < cmsgs[j][2] <= child]
                exact = [j for j in cands if cmsgs[j][2] == child]
                pool = exact or cands
                if pool:
                    j = min(pool, key=lambda j: abs(j - e))
                    if not aggr2 or j > aggr2[-1]:
                        aggr2.append(j)
                    e = j + mb               # advance from where we actually landed (tracks drift)
                else:
                    e += mb                  # nothing here; keep the cadence
            aggr2 = sorted(set(aggr2))
        else:                                 # fallback: the (possibly off) recorded indices
            aggr2 = sorted(int(i) for i in aggr if int(i) < len(cmsgs))
        samples.append(dict(
            label=f'{dt} · id{sid}', day=dt, n=len(books), junc=int(junc),
            child=child, mb=mb, ref=ref,
            aggr=aggr2, book0=book0, deltas=deltas, mids=mids, tick=tick, msgs=cmsgs))
    if not samples:
        raise SystemExit('no samples')
    parts = exp.split('-')
    stock = parts[0]
    model = parts[1] if len(parts) > 2 else '?'
    shape = parts[-1]
    ov = re.search(r'^order_volume\s*:\s*([0-9]+)', open(os.path.join(exp_dir, 'config.yaml')).read(), re.M) \
        if os.path.exists(os.path.join(exp_dir, 'config.yaml')) else None
    order_volume = int(ov.group(1)) if ov else None
    days = sorted({s['label'].split(' · ')[0] for s in samples})
    # full per-day calibration table (ALL days), so the player can show day/child/mb for every day
    daytable = [{'day': d, 'child': c, 'mb': m} for d, (c, m) in sorted(pdp.items())]
    payload = dict(exp=exp, side=side, stock=stock, model=model, shape=shape,
                   order_volume=order_volume, n_total=len(list_samples(exp_dir)),
                   days=days, daytable=daytable, tick=samples[0]['tick'], samples=samples)
    from plotly.offline import get_plotlyjs          # inline Plotly -> works OFFLINE (no CDN needed)
    html = _HTML.replace('%%TITLE%%', f'{exp} · {side}') \
                .replace('%%PLOTLYJS%%', get_plotlyjs()) \
                .replace('%%DATA%%', json.dumps(payload, separators=(',', ':')))
    out = out or os.path.join(HERE, 'results', 'book_player', f'player_{exp}_{side}.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w').write(html)
    print(f'wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, {len(samples)} samples, '
          f'{samples[0]["n"]} steps)')


_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Book player — %%TITLE%%</title>
<script>%%PLOTLYJS%%</script>
<style>
 body{font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#1a1a1a;}
 header{padding:9px 16px;background:#1a1a1a;color:#fff;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;}
 header h1{font-size:15px;margin:0;} header .sub{color:#9aa;font-size:12px;}
 .wrap{max-width:1320px;margin:0 auto;padding:12px 16px;}
 .ctl{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0;}
 select,button{font:13px inherit;padding:5px 9px;border:1px solid #ccc;border-radius:6px;background:#fff;cursor:pointer;}
 button:hover{background:#eef;} button:disabled{opacity:.4;cursor:default;}
 #slider{flex:1;min-width:240px;}
 #fig{width:100%;height:430px;}
 .msgbar{padding:9px 12px;border-radius:8px;background:#fff;border:1px solid #e2e2e2;margin:8px 0;
   display:flex;gap:13px;align-items:center;flex-wrap:wrap;font:13px ui-monospace,monospace;}
 #daytable{margin:8px 0;overflow-x:auto;}
 #daytable table{border-collapse:collapse;font:12px ui-monospace,monospace;}
 #daytable th,#daytable td{padding:3px 10px;border:1px solid #e6e6e6;text-align:right;white-space:nowrap;}
 #daytable th{background:#f2f2f2;color:#555;font-weight:600;}
 #daytable td:first-child,#daytable th:first-child{text-align:left;}
 #daytable tr.cur td{background:#fff3d6;font-weight:700;outline:2px solid #E8B04B;}
 #daytable tr:hover td{background:#eef4ff;cursor:default;}
 .tag{font-weight:700;padding:2px 8px;border-radius:5px;color:#fff;font-size:12px;}
 .badge{background:#E67E22;color:#fff;font-weight:700;padding:2px 8px;border-radius:5px;font-size:12px;}
 .legend{font-size:12px;color:#666;margin-top:6px;}
 .legend b{padding:1px 6px;border-radius:3px;} kbd{background:#eee;border:1px solid #ccc;border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:11px;}
</style></head><body>
<header><h1>📖 Order-book player</h1>
 <span id="stock" style="font-size:15px;font-weight:700;color:#E8B04B"></span>
 <span class="sub" id="hsub"></span>
 <span class="sub">step with <kbd>←</kbd> <kbd>→</kbd> · click the mid chart to jump</span></header>
<div class="wrap">
 <div class="ctl">
  <label>sample <select id="sample"></select></label>
  <button id="junc">→ junction</button><button id="prev">◀</button>
  <input id="slider" type="range" min="0" value="0">
  <button id="next">▶</button><button id="nagg">→ next aggr</button>
  <span id="info" class="sub" style="color:#333;min-width:120px;"></span>
 </div>
 <div id="fig"></div>
 <div class="msgbar" id="msgbar"></div>
 <div class="msgbar" id="statbar" style="background:#f7f9fc;"></div>
 <div class="msgbar" id="insbar" style="background:#fbf6ef;"></div>
 <div id="daytable"></div>
 <p class="legend">change on “стало”:
  <b style="background:rgba(192,57,43,.18);outline:1px solid #C0392B">red = volume ↑ (added)</b>&nbsp;
  <b style="background:rgba(47,93,163,.18);outline:1px solid #2F5DA3">blue = volume ↓ (executed/removed)</b>
  &nbsp;— an executed level is in “было” and gone (gap) in “стало”.</p>
</div>
<script>
const DATA=%%DATA%%, TICK=DATA.tick||1, SENT=2147483647;
const EV={1:'LIMIT',2:'CANCEL',3:'DELETE',4:'EXECUTE',5:'HID-EXEC',6:'CROSS'};
const EVC={1:'#2F5DA3',2:'#7F8C8D',3:'#C0392B',4:'#111',5:'#8E44AD',6:'#E67E22'};
const GRAY='#BBBBBB',UP='#C0392B',DOWN='#2F5DA3';
let S=null,rows=null,k=0,AGG=null;
function hms(t){ // LOBSTER time = seconds after midnight -> HH:MM:SS.nnnnnnnnn (+ raw seconds)
 if(t==null)return'?'; const s=Math.floor(t), ns=Math.round((t-s)*1e9);
 const p=(n,w)=>String(n).padStart(w||2,'0');
 return p(Math.floor(s/3600))+':'+p(Math.floor(s%3600/60))+':'+p(s%60)+'.'+p(ns,9)+
        ' <span class="sub">('+t.toFixed(9)+'s)</span>';}

function reconstruct(s){const r=new Array(s.n);let cur=s.book0.slice();r[0]=cur.slice();
 for(let i=1;i<s.n;i++){for(const[c,v]of s.deltas[i])cur[c]=v;r[i]=cur.slice();}return r;}
function sides(row){const a={},b={};for(let l=0;l<10;l++){
  const ap=row[4*l],av=row[4*l+1],bp=row[4*l+2],bv=row[4*l+3];
  if(ap>0&&ap<SENT&&av>0)a[ap]=(a[ap]||0)+av; if(bp>0&&bp<SENT&&bv>0)b[bp]=(b[bp]||0)+bv;}return{a,b};}
function bars(row,prev,diff){const cur=sides(row),pv=prev?sides(prev):{a:{},b:{}};
 const x=[],y=[],c=[];
 const col=(v,q)=>(!diff||v===q)?GRAY:(v>q?UP:DOWN);
 for(const p of Object.keys(cur.a).map(Number).sort((u,w)=>u-w)){x.push(p/TICK);y.push(cur.a[p]);c.push(col(cur.a[p],pv.a[p]||0));}
 for(const p of Object.keys(cur.b).map(Number).sort((u,w)=>u-w)){x.push(p/TICK);y.push(-cur.b[p]);c.push(col(cur.b[p],pv.b[p]||0));}
 return{x,y,c};}
function rangeOf(a,b){const xs=a.x.concat(b.x);if(!xs.length)return null;
 // ROBUST y-cap: a few very deep resting orders (e.g. 4771 vs ~200 at the touch) would otherwise
 // dwarf every near-touch bar. Cap at ~the 88th percentile so the touch stays readable; bigger
 // bars just clip at the top edge (real value still in hover).
 const ys=a.y.concat(b.y).map(Math.abs).filter(v=>v>0).sort((p,q)=>p-q);
 let ym=1; if(ys.length){const p=ys[Math.min(ys.length-1,Math.floor(ys.length*0.88))];
   ym=Math.max(1,Math.min(ys[ys.length-1],p*1.5));}
 return{x:[Math.min(...xs)-0.6,Math.max(...xs)+0.6], y:ym*1.08};}

function initFig(){
 const mids=S.mids.map((v,i)=>v), xs=[...Array(S.n).keys()];
 const av=S.aggr.filter(i=>i<S.n);
 Plotly.newPlot('fig',[
  {type:'bar',x:[],y:[],marker:{color:GRAY},width:0.4,xaxis:'x',yaxis:'y',hoverinfo:'x+y'},
  {type:'bar',x:[],y:[],marker:{color:[]},width:0.4,xaxis:'x2',yaxis:'y2',hoverinfo:'x+y'},
  {type:'scatter',x:xs,y:mids,mode:'lines',line:{color:'#1a1a1a',width:1},xaxis:'x3',yaxis:'y3',hoverinfo:'x+y'},
  {type:'scatter',x:av,y:av.map(i=>mids[i]),mode:'markers',marker:{color:'#2E7D52',size:6,symbol:'triangle-up'},xaxis:'x3',yaxis:'y3',hoverinfo:'x'}
 ],{
  template:'plotly_white',showlegend:false,height:430,margin:{l:50,r:15,t:34,b:34},bargap:0.1,
  xaxis:{domain:[0,0.3],title:'price'},yaxis:{title:'qty (ask + / bid −)'},
  xaxis2:{domain:[0.35,0.65],title:'price'},yaxis2:{anchor:'x2'},
  xaxis3:{domain:[0.72,1],title:'step'},yaxis3:{anchor:'x3',title:'mid'},
  annotations:[
   {text:'Было — t-1',x:0.15,y:1.06,xref:'paper',yref:'paper',showarrow:false,font:{size:12}},
   {text:'Стало — t',x:0.5,y:1.06,xref:'paper',yref:'paper',showarrow:false,font:{size:12}},
   {text:'Mid-price',x:0.86,y:1.06,xref:'paper',yref:'paper',showarrow:false,font:{size:12}}],
  shapes:[
   {type:'line',xref:'x3',yref:'paper',x0:0,x1:0,y0:0,y1:1,line:{color:'#888',width:1,dash:'dash'}},
   {type:'line',xref:'x3',yref:'paper',x0:S.junc,x1:S.junc,y0:0,y1:1,line:{color:'#C0392B',width:2}}]
 },{displayModeBar:false,responsive:true});
 document.getElementById('fig').on('plotly_click',e=>{
  if(e.points&&e.points[0].data.xaxis==='x3')setK(Math.round(e.points[0].x));});
}
function render(){
 const before=bars(rows[Math.max(0,k-1)],null,false), after=bars(rows[k],rows[Math.max(0,k-1)],true);
 const rg=rangeOf(before,after);
 Plotly.restyle('fig',{x:[before.x],y:[before.y]},[0]);
 Plotly.restyle('fig',{x:[after.x],y:[after.y],'marker.color':[after.c]},[1]);
 const up={'shapes[0].x0':k,'shapes[0].x1':k,
  'annotations[0].text':'Было — t='+(k-1),'annotations[1].text':'Стало — t='+k};
 if(rg){up['xaxis.range']=rg.x;up['xaxis2.range']=rg.x;up['yaxis.range']=[-rg.y,rg.y];up['yaxis2.range']=[-rg.y,rg.y];}
 Plotly.relayout('fig',up);
 document.getElementById('slider').value=k;
 document.getElementById('info').textContent='step '+k+' / '+(S.n-1);
 document.getElementById('prev').disabled=(k<=0); document.getElementById('next').disabled=(k>=S.n-1);
 const m=S.msgs[k],agg=AGG.has(k);
 // m[1] = the message's own side (for et=4 executions this is the RESTING side that got hit).
 const restSide=m[1]>0?'bid (buy-side)':'ask (sell-side)';
 const EXP=DATA.side.toUpperCase();   // the experiment / aggressive-order direction (buy or sell folder)
 document.getElementById('msgbar').innerHTML=
  '<span class="tag" style="background:'+(EVC[m[0]]||'#555')+'">'+(EV[m[0]]||('et'+m[0]))+'</span>'+
  '<span>'+m[2]+' @ <b>'+(m[3]/TICK).toFixed(TICK>=100?2:0)+'</b></span>'+
  '<span class="sub">side: '+restSide+'</span>'+
  '<span class="sub">order '+m[4]+' · '+hms(m[5])+'</span>'+
  (agg?'<span class="badge">◆ AGGRESSIVE '+EXP+'</span>':'');
 // bottom stat line (per-day calibration table values): day | child volume (p50) | msgs_btw (mb)
 document.getElementById('statbar').innerHTML=
  '<span>📅 день <b>'+S.day+'</b></span>'+
  '<span>объём (исполняемый/день, p50): <b>'+(S.child!=null?S.child:'?')+'</b> shares</span>'+
  '<span>msgs_btw (mb): <b>'+(S.mb!=null?S.mb:'?')+'</b></span>'+
  '<span class="sub">mid='+(S.mids[k]!=null?S.mids[k].toFixed(2):'?')+'</span>';
}
function setK(n){k=Math.max(0,Math.min(S.n-1,n));render();}
function nextAgg(){for(const i of S.aggr)if(i>k){setK(i);return;}}
function dayTable(curDay){
 const t=DATA.daytable||[];
 if(!t.length){document.getElementById('daytable').innerHTML='';return;}
 let h='<table><tr><th>day</th><th>child (p50 vol)</th><th>msgs_btw (mb)</th></tr>';
 for(const r of t){const cur=(r.day===curDay)?' class="cur"':'';
   h+='<tr'+cur+'><td>'+r.day+'</td><td>'+(r.child!=null?r.child:'?')+'</td><td>'+(r.mb!=null?r.mb:'?')+'</td></tr>';}
 document.getElementById('daytable').innerHTML=h+'</table>';
}
function load(idx){S=DATA.samples[idx];rows=reconstruct(S);AGG=new Set(S.aggr);
 const sl=document.getElementById('slider');sl.max=S.n-1;
 const day=S.day;
 document.getElementById('stock').textContent='📈 '+DATA.stock+' · '+day;
 // header = identity only; the REAL per-day numbers (child/mb) live in the stat line + day table below.
 document.getElementById('hsub').innerHTML=DATA.model+'/'+DATA.shape+' · '+DATA.side+
   ' · '+S.label.split(' · ')[1]+' · '+S.n+' msgs · '+S.aggr.length+' aggressive · '+
   'sample '+(idx+1)+'/'+DATA.samples.length+' ('+DATA.n_total+' on disk · '+(DATA.daytable||DATA.days).length+' days)';
 dayTable(day);
 // clickable list of THIS sample's aggressive insertions (step, gen-idx, executed size)
 let ins='<span class="sub">вставки ('+S.aggr.length+'):</span>';
 S.aggr.forEach((a,i)=>{const sz=S.msgs[a]?S.msgs[a][2]:'?';
   ins+=' <button onclick="setK('+a+')" title="gen '+(a-S.junc)+'" style="padding:2px 7px">#'+(i+1)+': шаг '+a+' <span class="sub">(gen '+(a-S.junc)+', '+sz+' sh)</span></button>';});
 document.getElementById('insbar').innerHTML=ins;
 initFig();k=Math.min(S.junc,S.n-1);render();}
const sel=document.getElementById('sample');
DATA.samples.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent='#'+i+' · '+DATA.stock+' · '+s.label;sel.appendChild(o);});
sel.onchange=()=>load(+sel.value);
document.getElementById('prev').onclick=()=>setK(k-1);
document.getElementById('next').onclick=()=>setK(k+1);
document.getElementById('junc').onclick=()=>setK(S.junc);
document.getElementById('nagg').onclick=nextAgg;
document.getElementById('slider').oninput=e=>setK(+e.target.value);
window.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){setK(k+1);e.preventDefault();}else if(e.key==='ArrowLeft'){setK(k-1);e.preventDefault();}});
load(0);
</script></body></html>
"""


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default=GRID)
    ap.add_argument('--exp', default='EA-Mamba3-beta')
    ap.add_argument('--side', default='buy', choices=['buy', 'sell'])
    ap.add_argument('--step', default='aggr', help="'aggr' (first insertion) | int step")
    ap.add_argument('--html', action='store_true',
                    help='write a self-contained interactive HTML player (no kernel needed)')
    ap.add_argument('--n_samples', type=int, default=6)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.html:
        export_html(a.grid, a.exp, a.side, a.n_samples, a.max_steps, a.out)
    else:
        _snapshot(a.grid, a.exp, a.side, a.step, a.out)
