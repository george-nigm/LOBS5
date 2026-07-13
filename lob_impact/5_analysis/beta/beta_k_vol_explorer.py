#!/usr/bin/env python3
"""
β–k VOLATILITY explorer: one self-contained HTML per stock.

For every model the ≤k (cumulative) pooling is fitted three ways at every k:
  free    — OLS of ln(I/σ) on ln(Q/V), free intercept, I>0 subset (the per-point
            estimator of the 3-views figure; carries the E[ln I | I>0] bias)
  origin  — same regression forced through the origin (intercept = 0)
  binned  — the signed-conditional-mean binned δ (18 quantile bins, NO I>0
            selection; the paper's headline estimator)
under all SIX σ normalisations: σ=1, parkinson, garman_klass, rogers_satchell,
close_to_close, yang_zhang.  Bottom panel: the free-fit intercept(k).

Why: on GOOG/NVDA the per-point curves park replays at β≈0.5 (pure-diffusion
scaling of |noise| ~ σ√T) while the signed binned δ puts them at ≈0 — switching
fit mode with the same points on screen shows exactly where the 0.5 comes from.

Also runs a day-clustered bootstrap (resample days with replacement) for the
full-panel binned δ under σ=parkinson and σ=1 → CI in the printed table + npz.

All plotted arrays are dumped next to the HTML as .npz (redraws need no grid).

  python beta_k_vol_explorer.py --grid <root> --stock GOOG --daily <daily.csv> \
      --models Historic,...,S5_4k [--kmax 100] [--bootstrap 400]
"""
import os, json, argparse
import numpy as np
from beta_grid import collect
from vol_estimators import daily_sigmas, METHODS
import plotly.graph_objects as go
import plotly.offline as pyo

ALL_METHODS = ['none'] + METHODS          # 6 normalisations
MODES = ['free', 'origin', 'binned']
NBINS = 18
MIN_PTS = 30

MODEL_COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
                'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
                'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
                'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E'}
MODE_DASH = {'free': 'solid', 'origin': 'dot', 'binned': 'dash'}


def delta_from_bins(x, y, nbins=NBINS):
    n = x.size
    if n < nbins * 3:
        nbins = max(6, n // 3)
    if n < 20:
        return np.nan
    order = np.argsort(x)
    xc, ym = [], []
    for chunk in np.array_split(order, nbins):
        xc.append(float(np.mean(x[chunk]))); ym.append(float(np.mean(y[chunk])))
    xc, ym = np.array(xc), np.array(ym)
    pos = ym > 0
    if pos.sum() < 3:
        return np.nan
    return float(np.polyfit(xc[pos], np.log(ym[pos]), 1)[0])


def fits_at_k(x, y, kc, ks):
    """Per k in ks: (beta_free, intercept_free, beta_origin, delta_binned)."""
    bf = np.full(len(ks), np.nan); ic = np.full(len(ks), np.nan)
    bo = np.full(len(ks), np.nan); bb = np.full(len(ks), np.nan)
    ly = np.where(y > 0, np.log(np.where(y > 0, y, 1.0)), np.nan)
    for i, k in enumerate(ks):
        m = kc <= k
        if m.sum() < MIN_PTS:
            continue
        xm, lym = x[m], ly[m]
        ok = np.isfinite(lym)
        if ok.sum() >= MIN_PTS and (xm[ok].max() - xm[ok].min()) > 1e-6:
            b, a = np.polyfit(xm[ok], lym[ok], 1)
            bf[i], ic[i] = b, a
            bo[i] = float(np.dot(xm[ok], lym[ok]) / np.dot(xm[ok], xm[ok]))
        bb[i] = delta_from_bins(xm, y[m])
    return bf, ic, bo, bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,Propagator,Hawkes,CST,NMZI,'
                                        'Mamba3,GDN,Mamba3_4k,S5_4k')
    ap.add_argument('--kmax', type=int, default=100)
    ap.add_argument('--kstep', type=int, default=1)
    ap.add_argument('--bootstrap', type=int, default=400)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    models = [m for m in args.models.split(',') if m]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'results', 'beta_k_vol')
    os.makedirs(outdir, exist_ok=True)
    out = args.out or os.path.join(outdir, f'beta_k_vol_{args.stock}.html')
    ks = np.arange(1, args.kmax + 1, args.kstep)

    cache = {'ks': ks}
    summary = {}
    rng = np.random.default_rng(42)
    traces = []
    for mi, model in enumerate(models):
        raw = collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'buy'), args.stock, +1) + \
              collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'sell'), args.stock, -1)
        if not raw:
            print(f'{model}: no data'); continue
        Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
        days = np.array([r[2] for r in raw]); kc = np.array([r[3] for r in raw], int) + 1
        V = np.array([sig.get((args.stock, d), {}).get('V', np.nan) for d in days], float)
        x_all = np.log(Q / V)
        col = MODEL_COLORS.get(model, '#555555')
        summary[model] = {}
        for method in ALL_METHODS:
            if method == 'none':
                y_all = I.copy()
            else:
                sg = np.array([sig.get((args.stock, d), {}).get(method, np.nan) for d in days], float)
                y_all = I / sg
            ok = np.isfinite(x_all) & np.isfinite(y_all)
            x, y, kk, dd = x_all[ok], y_all[ok], kc[ok], days[ok]
            bf, ic, bo, bb = fits_at_k(x, y, kk, ks)
            tag = f'{model}_{method}'
            cache[f'{tag}_free'] = bf; cache[f'{tag}_int'] = ic
            cache[f'{tag}_origin'] = bo; cache[f'{tag}_binned'] = bb
            for mode, arr in (('free', bf), ('origin', bo), ('binned', bb)):
                traces.append(dict(x=ks.tolist(), y=np.round(arr, 4).tolist(), model=model,
                                   method=method, mode=mode, color=col, panel='beta'))
            traces.append(dict(x=ks.tolist(), y=np.round(ic, 4).tolist(), model=model,
                               method=method, mode='intercept', color=col, panel='int'))
            # full-panel numbers (+ day-clustered bootstrap CI for the two headline sigmas)
            d_full = delta_from_bins(x, y)
            entry = {'binned': d_full, 'free': float(bf[-1]), 'origin': float(bo[-1]),
                     'intercept': float(ic[-1]), 'N': int(x.size)}
            if args.bootstrap and method in ('parkinson', 'none'):
                udays = np.unique(dd)
                idx_by_day = {d: np.where(dd == d)[0] for d in udays}
                bs = []
                for _ in range(args.bootstrap):
                    pick = rng.choice(udays, size=len(udays), replace=True)
                    sel = np.concatenate([idx_by_day[d] for d in pick])
                    bs.append(delta_from_bins(x[sel], y[sel]))
                bs = np.array(bs, float)
                entry['ci'] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
            summary[model][method] = entry
        p = summary[model].get('parkinson', {})
        s1 = summary[model].get('none', {})
        print(f"{model:10s} park: binned={p.get('binned', np.nan):+.3f} "
              f"CI[{p.get('ci', [np.nan, np.nan])[0]:+.2f},{p.get('ci', [np.nan, np.nan])[1]:+.2f}] "
              f"free={p.get('free', np.nan):+.3f} origin={p.get('origin', np.nan):+.3f} | "
              f"sigma1: binned={s1.get('binned', np.nan):+.3f} free={s1.get('free', np.nan):+.3f} "
              f"(N={p.get('N', 0):,})", flush=True)

    # ---------- figure: 2 stacked panels, custom JS radio controls ----------
    fig = go.Figure()
    for t in traces:
        fig.add_trace(go.Scatter(
            x=t['x'], y=t['y'], mode='lines',
            name=f"{t['model']}",
            legendgroup=t['model'], showlegend=(t['method'] == 'parkinson' and t['mode'] == 'free'),
            line=dict(color=t['color'], dash=MODE_DASH.get(t['mode'], 'solid'), width=2),
            yaxis='y2' if t['panel'] == 'int' else 'y',
            visible=(t['method'] == 'parkinson' and t['mode'] in ('free', 'binned')),
            meta=[t['method'], t['mode'], t['model']],
            hovertemplate=f"{t['model']} · {t['method']} · {t['mode']}<br>k=%{{x}} β=%{{y}}<extra></extra>"))
    fig.add_hline(y=0.5, line_dash='dash', line_color='#c0392b', opacity=0.7)
    fig.add_hline(y=0.0, line_color='#999999', line_width=0.8)
    fig.update_layout(
        height=850, template='plotly_white',
        title=(f'{args.stock} — β(≤k) under 6 σ-normalisations × 3 fit modes '
               f'(solid=per-point free-intercept · dot=through-origin · dash=signed binned δ; red 0.5 = √-law)'),
        yaxis=dict(domain=[0.32, 1.0], title='β(≤k)', range=[-0.6, 1.3]),
        yaxis2=dict(domain=[0.0, 0.24], title='free-fit intercept(k)'),
        xaxis=dict(title='insertion index k (pooling all points with k′ ≤ k)'),
        legend=dict(orientation='h', y=1.06, font=dict(size=10)))
    html = pyo.plot(fig, include_plotlyjs=True, output_type='div')

    controls = """
<div style="font-family:sans-serif;margin:8px 14px">
 <b>σ:</b> <span id="mth"></span> &nbsp;&nbsp; <b>fit:</b> <span id="mds"></span>
 <span style="color:#777;margin-left:18px">click legend entries to hide models; the intercept
 panel follows the σ choice (free fit only)</span>
</div>
<script>
const METHODS=%s, MODES=[['free+binned','fb'],['free','free'],['origin','origin'],['binned','binned'],['all','all']];
let curM='parkinson', curF='fb';
function apply(){
  const gd=document.querySelectorAll('.plotly-graph-div')[0];
  const vis=gd.data.map(t=>{
    if(!t.meta) return true;
    const [mth,mode]=t.meta;
    if(mth!==curM) return false;
    if(mode==='intercept') return true;
    if(curF==='all') return true;
    if(curF==='fb') return (mode==='free'||mode==='binned');
    return mode===curF;
  });
  Plotly.restyle(gd,{visible:vis});
}
function mkbtns(el,items,cur,cb){
  items.forEach(it=>{const [lab,val]=Array.isArray(it)?it:[it,it];
    const b=document.createElement('button');b.textContent=lab;b.dataset.v=val;
    b.style.cssText='margin:2px;padding:3px 10px;border:1px solid #bbb;border-radius:4px;cursor:pointer;background:'+(val===cur?'#2F5DA3':'#f5f5f5')+';color:'+(val===cur?'#fff':'#000');
    b.onclick=()=>{cb(val);[...el.children].forEach(c=>{const on=c.dataset.v===val;c.style.background=on?'#2F5DA3':'#f5f5f5';c.style.color=on?'#fff':'#000';});apply();};
    el.appendChild(b);});
}
window.addEventListener('load',()=>{
  mkbtns(document.getElementById('mth'),METHODS,'parkinson',v=>curM=v);
  mkbtns(document.getElementById('mds'),MODES,'fb',v=>curF=v);
});
</script>""" % json.dumps(ALL_METHODS)

    with open(out, 'w') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><title>%s β-k vol explorer</title></head>'
                '<body>%s%s</body></html>' % (args.stock, controls, html))
    np.savez(os.path.splitext(out)[0] + '.npz', **cache)
    with open(os.path.splitext(out)[0] + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'BETA_K_VOL_DONE -> {out} (+ .npz, _summary.json)')


if __name__ == '__main__':
    main()
