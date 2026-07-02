#!/usr/bin/env python3
"""
Multi-model interactive β explorer — ONE self-contained HTML (Plotly inlined, offline) with an
explicit k-mode toggle:

  ≤k  (CUMULATIVE) : at index k, pool ALL insertions k'≤k and fit  -> the smooth "paper" β(k)
                     (this is the beautiful curve; Mamba3 rises toward the 0.5 √-law)
  ==k (EXACT)      : only insertions exactly == k -> the noisy cross-section drill-down

Layout: 5 columns = σ-methods · row1 = impact cloud of the SELECTED model (highlight + int/org fit)
· row2 = β·intercept (ALL models overlaid) · row3 = β·origin (ALL models). One slider, one model
radio, one ≤k/==k toggle drive every panel in sync. Fit: log(I/σ)=[α+]β·log(Q/V).

  python beta_explorer_multimodel.py --grid <root> --daily <csv> --stock EA \
         --models Historic,Heuristic,CST,Mamba3 --out beta_explorer_multimodel_EA.html
"""
import os, json, argparse
import numpy as np
from plotly.offline import get_plotlyjs
from beta_grid import collect
from vol_estimators import daily_sigmas, METHODS

MODEL_COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'CST': '#27AE60',
                'Mamba3': '#2F5DA3', 'S5': '#E67E22', 'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22'}
DISPLAY_CAP = 24000


def fit(x, y, intercept):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 8 or (x.max() - x.min()) < 1e-6:
        return dict(b=None, a=0.0, se=None, x0=0.0, x1=0.0)
    if intercept:
        b, a = np.polyfit(x, y, 1)
    else:
        b = float(np.dot(x, y) / np.dot(x, x)); a = 0.0
    resid = y - (b * x + a)
    dof = n - 2 if intercept else n - 1
    sxx = np.sum((x - x.mean()) ** 2) if intercept else np.sum(x * x)
    se = float(np.sqrt((np.sum(resid ** 2) / dof) / sxx)) if (dof > 0 and sxx > 0) else None
    return dict(b=float(b), a=float(a), se=se, x0=float(x.min()), x1=float(x.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--models', default='Historic,Heuristic,CST,Mamba3')
    ap.add_argument('--kmax', type=int, default=100)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sig = daily_sigmas(args.daily)
    models = [m for m in args.models.split(',') if m]
    rng = np.random.default_rng(0)
    D = {'stock': args.stock, 'methods': METHODS, 'models': [], 'colors': {},
         'pts': {}, 'curves': {}, 'fits': {}, 'kmax': args.kmax}

    for model in models:
        exp = f'{args.stock}-{model}-beta'
        raw = collect(os.path.join(args.grid, exp, 'buy'), args.stock, +1) + \
              collect(os.path.join(args.grid, exp, 'sell'), args.stock, -1)
        if not raw:
            print(f'{exp}: no data — skip'); continue
        Q = np.array([r[0] for r in raw], float)
        I = np.array([r[1] for r in raw], float)
        days = [r[2] for r in raw]
        kc = np.array([r[3] for r in raw], int) + 1
        V = np.array([sig.get((args.stock, d), {}).get('V', np.nan) for d in days], float)
        x = np.log(Q / V); li = np.log(I)
        ok = np.isfinite(x) & np.isfinite(li)
        x, li, kc = x[ok], li[ok], kc[ok]
        days = [d for d, o in zip(days, ok) if o]
        kmax = int(min(args.kmax, kc.max()))

        curves = {m: {im: {'exact': {'k': [], 'b': [], 'lo': [], 'hi': []},
                           'cumul': {'k': [], 'b': [], 'lo': [], 'hi': []}}
                      for im in ('int', 'org')} for m in METHODS}
        fits = {m: {im: {'exact': {}, 'cumul': {}} for im in ('int', 'org')} for m in METHODS}
        for mth in METHODS:
            logsig = np.array([np.log(sig.get((args.stock, d), {}).get(mth, np.nan)) for d in days], float)
            y = li - logsig
            for k in range(1, kmax + 1):
                masks = {'exact': kc == k, 'cumul': kc <= k}
                for km, sel in masks.items():
                    for im, inter in (('int', True), ('org', False)):
                        f = fit(x[sel], y[sel], inter)
                        fits[mth][im][km][k] = dict(b=f['b'], a=f['a'], x0=f['x0'], x1=f['x1'])
                        c = curves[mth][im][km]
                        c['k'].append(k); c['b'].append(f['b'])
                        c['lo'].append(None if (f['b'] is None or f['se'] is None) else f['b'] - 1.96 * f['se'])
                        c['hi'].append(None if (f['b'] is None or f['se'] is None) else f['b'] + 1.96 * f['se'])

        n = len(x)
        idx = rng.choice(n, size=min(DISPLAY_CAP, n), replace=False)
        uday = sorted(set(days)); didx = {d: i for i, d in enumerate(uday)}
        D['pts'][model] = dict(x=[round(float(v), 4) for v in x[idx]],
                               li=[round(float(v), 4) for v in li[idx]],
                               k=[int(kc[i]) for i in idx],
                               d=[int(didx[days[i]]) for i in idx], days=uday)
        D['curves'][model] = curves; D['fits'][model] = fits
        D['colors'][model] = MODEL_COLORS.get(model, '#444'); D['models'].append(model)
        print(f'{exp}: {n} pts, kmax={kmax}, display {len(idx)}')

    alldays = sorted({d for m in D['models'] for d in D['pts'][m]['days']})
    D['lsig'] = {mth: {d: (float(np.log(sig[(args.stock, d)][mth]))
                           if (args.stock, d) in sig and np.isfinite(sig[(args.stock, d)].get(mth, np.nan))
                           and sig[(args.stock, d)][mth] > 0 else None)
                       for d in alldays} for mth in METHODS}

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'beta_explorer', f'beta_explorer_multimodel_{args.stock}.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(_TEMPLATE.replace('%%PLOTLYJS%%', get_plotlyjs()).replace('%%DATA%%', json.dumps(D)))
    print(f'  saved -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)')


_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>multi-model β explorer</title>
<script>%%PLOTLYJS%%</script>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
 #bar{padding:10px 16px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0;z-index:5}
 #bar h2{margin:0 0 6px;font-size:15px}
 .ctl{display:inline-flex;align-items:center;gap:8px;margin-right:18px}
 #kval{font-weight:700;color:#C0392B;min-width:2.5em;display:inline-block}
 button{border:1px solid #bbb;background:#f3f3f3;border-radius:5px;padding:3px 9px;cursor:pointer}
 button.on{color:#fff;border-color:#222}
 #kmode button.on{background:#16a085;border-color:#0e6b56}
 input[type=range]{width:420px;vertical-align:middle}
 #grid{display:grid;grid-template-columns:96px repeat(5,1fr);gap:2px;padding:6px}
 .hd{display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;background:#eef2f7;border-radius:4px;padding:4px;text-align:center}
 .rl{display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;writing-mode:vertical-rl;transform:rotate(180deg);background:#eef2f7;border-radius:4px}
 .cell{height:232px;min-width:0}
 #readout{font-size:13px;margin-top:6px;font-family:ui-monospace,monospace}
</style></head><body>
<div id="bar">
 <h2 id="title">multi-model β explorer</h2>
 <div class="ctl">k-mode: <span id="kmode"></span></div>
 <div class="ctl">model (cloud + bold): <span id="mbtns"></span></div>
 <div class="ctl">k = <input type="range" id="kslider" min="1" max="100" value="100"><span id="kval">100</span></div>
 <div class="ctl">quick: <button onclick="setk(1)">1</button><button onclick="setk(2)">2</button><button onclick="setk(3)">3</button><button onclick="setk(4)">4</button><button onclick="setk(10)">10</button><button onclick="setk(50)">50</button><button onclick="setk(100)">100</button></div>
 <div id="readout"></div>
</div>
<div id="grid"></div>
<script>
const D=%%DATA%%;
const METHODS=D.methods,MODELS=D.models,COL=D.colors,KMAX=D.kmax,LSIG=D.lsig;
const LABEL={parkinson:'Parkinson',garman_klass:'Garman-Klass',rogers_satchell:'Rogers-Satchell',close_to_close:'close-to-close',yang_zhang:'Yang-Zhang'};
let sel=MODELS[0], curk=KMAX, kmode='cumul';
document.getElementById('title').textContent=D.stock+'  —  multi-model β  ·  rows: cloud(selected) | β intercept (all models) | β origin (all)  ·  cols: 5 σ-methods';
document.getElementById('kslider').max=KMAX;

// k-mode toggle
const km=document.getElementById('kmode');
[['cumul','≤k  (cumulative — smooth)'],['exact','==k  (exact — noisy drill-down)']].forEach(([v,t])=>{
  const b=document.createElement('button');b.textContent=t;b.dataset.v=v;
  b.onclick=()=>{kmode=v;[...km.children].forEach(c=>c.classList.toggle('on',c.dataset.v===v));rebuildCurves();update(curk);};
  if(v===kmode)b.classList.add('on');km.appendChild(b);});

// model buttons
const mb=document.getElementById('mbtns');
MODELS.forEach(m=>{const b=document.createElement('button');b.textContent=m;b.style.borderLeft='6px solid '+COL[m];
  b.onclick=()=>{sel=m;[...mb.children].forEach(c=>{c.classList.toggle('on',c.textContent===m);c.style.background=c.textContent===m?COL[m]:'#f3f3f3';});redrawClouds();rebuildCurves();update(curk);};
  if(m===sel){b.classList.add('on');b.style.background=COL[m];}mb.appendChild(b);});

function yval(m,i,mth){const d=D.pts[m].days[D.pts[m].d[i]];const ls=LSIG[mth][d];return (ls==null)?NaN:(D.pts[m].li[i]-ls);}
const BYK={};MODELS.forEach(m=>{BYK[m]={};const k=D.pts[m].k;for(let i=0;i<k.length;i++){(BYK[m][k[i]]||=[]).push(i);}});

const grid=document.getElementById('grid');
const ROWS=[{k:'cloud',l:'cloud (selected)'},{k:'bint',l:'β · intercept'},{k:'borg',l:'β · origin'}];
const CLOUD={},BINT={},BORG={};
grid.appendChild(Object.assign(document.createElement('div'),{className:'hd'}));
METHODS.forEach(mth=>grid.appendChild(Object.assign(document.createElement('div'),{className:'hd',textContent:LABEL[mth]})));
ROWS.forEach(r=>{grid.appendChild(Object.assign(document.createElement('div'),{className:'rl',textContent:r.l}));
  METHODS.forEach(mth=>{const d=document.createElement('div');d.className='cell';d.id=r.k+'_'+mth;grid.appendChild(d);
    if(r.k==='cloud')CLOUD[mth]=d.id;else if(r.k==='bint')BINT[mth]=d.id;else BORG[mth]=d.id;});});
const AX={margin:{l:34,r:6,t:6,b:22},showlegend:false,font:{size:9}};

function makeCloud(id){Plotly.newPlot(id,[
  {x:[],y:[],mode:'markers',type:'scattergl',marker:{size:2,color:'#cfd8e3',opacity:0.3},hoverinfo:'skip'},
  {x:[],y:[],mode:'markers',type:'scattergl',marker:{size:4,color:'#C0392B',opacity:0.7},hoverinfo:'skip'},
  {x:[],y:[],mode:'lines',line:{color:'#000',width:2}},
  {x:[],y:[],mode:'lines',line:{color:'#888',width:1.5,dash:'dash'}}
],Object.assign({},AX,{xaxis:{title:''},yaxis:{title:''}}),{responsive:true,displayModeBar:false});}
function makeCurves(id){const t=[];MODELS.forEach(m=>t.push({x:[],y:[],mode:'lines',name:m,line:{color:COL[m],width:1.2}}));
  t.push({x:[1],y:[null],mode:'markers',marker:{size:11,color:'#000',symbol:'circle-open',line:{width:2.5}}});
  Plotly.newPlot(id,t,Object.assign({},AX,{yaxis:{range:[-0.6,1.0]},xaxis:{title:''},
    shapes:[{type:'line',x0:1,x1:KMAX,y0:0.5,y1:0.5,line:{color:'red',dash:'dash',width:0.7}}]}),{responsive:true,displayModeBar:false});}
METHODS.forEach(mth=>{makeCloud(CLOUD[mth]);makeCurves(BINT[mth]);makeCurves(BORG[mth]);});

function rebuildCurves(){
  METHODS.forEach(mth=>{
    [['int',BINT[mth]],['org',BORG[mth]]].forEach(([im,id])=>{
      MODELS.forEach((m,ti)=>{const c=D.curves[m][mth][im][kmode];
        Plotly.restyle(id,{x:[c.k],y:[c.b],'line.width':m===sel?2.6:1.1,opacity:m===sel?1:0.5},[ti]);});
    });
  });
}
function redrawClouds(){METHODS.forEach(mth=>{const ids=D.pts[sel].x.map((_,i)=>i);
  Plotly.restyle(CLOUD[mth],{x:[ids.map(i=>D.pts[sel].x[i])],y:[ids.map(i=>yval(sel,i,mth))]},[0]);});}

function update(k){curk=k;
  METHODS.forEach(mth=>{
    // cloud highlight: exact -> k==K ; cumul -> k<=K
    let idxs=[]; if(kmode==='exact'){idxs=BYK[sel][k]||[];}else{for(let kk=1;kk<=k;kk++){const a=BYK[sel][kk];if(a)idxs=idxs.concat(a);}}
    Plotly.restyle(CLOUD[mth],{x:[idxs.map(i=>D.pts[sel].x[i])],y:[idxs.map(i=>yval(sel,i,mth))]},[1]);
    const fi=D.fits[sel][mth].int[kmode][k], fo=D.fits[sel][mth].org[kmode][k];
    if(fi&&fi.b!=null)Plotly.restyle(CLOUD[mth],{x:[[fi.x0,fi.x1]],y:[[fi.b*fi.x0+fi.a,fi.b*fi.x1+fi.a]]},[2]);else Plotly.restyle(CLOUD[mth],{x:[[]],y:[[]]},[2]);
    if(fo&&fo.b!=null)Plotly.restyle(CLOUD[mth],{x:[[fo.x0,fo.x1]],y:[[fo.b*fo.x0,fo.b*fo.x1]]},[3]);else Plotly.restyle(CLOUD[mth],{x:[[]],y:[[]]},[3]);
    const ci=D.curves[sel][mth].int[kmode].b[k-1], co=D.curves[sel][mth].org[kmode].b[k-1];
    Plotly.restyle(BINT[mth],{x:[[k]],y:[[ci]]},[MODELS.length]);
    Plotly.restyle(BORG[mth],{x:[[k]],y:[[co]]},[MODELS.length]);
  });
  document.getElementById('kval').textContent=k;
  const fmt=v=>v==null?' — ':(v>=0?' ':'')+v.toFixed(2);
  document.getElementById('readout').textContent=(kmode==='cumul'?'≤k cumulative':'==k exact')+'   k='+k+
    '   intercept β @ '+LABEL[METHODS[0]]+':  '+MODELS.map(m=>m.slice(0,4)+'='+fmt(D.curves[m][METHODS[0]].int[kmode].b[k-1])).join('  ');
}
function setk(k){k=Math.max(1,Math.min(KMAX,k));document.getElementById('kslider').value=k;update(k);}
document.getElementById('kslider').addEventListener('input',e=>update(+e.target.value));
document.addEventListener('keydown',e=>{const s=document.getElementById('kslider');if(e.key==='ArrowRight')setk(+s.value+1);if(e.key==='ArrowLeft')setk(+s.value-1);});
redrawClouds();rebuildCurves();update(KMAX);
</script></body></html>"""


if __name__ == '__main__':
    main()
