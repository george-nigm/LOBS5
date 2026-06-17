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
import os, glob, re
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
    samples = list_samples(exp_dir)[:max_samples]
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


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default=GRID)
    ap.add_argument('--exp', default='EA-Mamba3-beta')
    ap.add_argument('--side', default='buy', choices=['buy', 'sell'])
    ap.add_argument('--step', default='aggr', help="'aggr' (first insertion) | int step")
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    _snapshot(a.grid, a.exp, a.side, a.step, a.out)
