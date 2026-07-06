#!/usr/bin/env python3
"""
Interactive EXACT-k beta explorer (self-contained HTML, Plotly inlined, offline).

Isolates the cross-section of points where the metaorder is EXACTLY k child orders (kc = k,
not ≤k / ≥k) and fits, live in the browser, under any of the 5 σ-estimators and intercept on/off:

    log(I / σ_method) = [α +] β_kc · log(Q / V)

β_kc answers "what would β be if the metaorder were exactly kc children". The HTML lets you:
  - drag a k-slider (or click 1/2/3/4) → only EXACT-kc points light up, the rest dim/vanish;
  - switch among Parkinson / Garman-Klass / Rogers-Satchell / close-to-close / Yang-Zhang;
  - toggle the free intercept (off = OLS through origin);
  - read β_kc (+ analytic 95% CI, n) with the live OLS fit line and the β_kc-vs-kc curve.

  python 5_analysis/beta/beta_explorer_html.py --grid <root> --exp EA-Historic-beta \
         --daily <daily_h_l_all.csv> --out beta_explorer_EA-Historic-beta.html
"""
import os, json, argparse
import numpy as np
from plotly.offline import get_plotlyjs
from beta_grid import collect                 # per-day-aware (Q, I, day, k)
from vol_estimators import daily_sigmas, METHODS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--exp', required=True)
    ap.add_argument('--daily', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--kmax', type=int, default=100)
    args = ap.parse_args()

    ticker = args.exp.split('-')[0]
    sig = daily_sigmas(args.daily)
    raw = collect(os.path.join(args.grid, args.exp, 'buy'), ticker, +1) + \
          collect(os.path.join(args.grid, args.exp, 'sell'), ticker, -1)
    if not raw:
        print(f'{args.exp}: no data'); return
    Q = np.array([r[0] for r in raw], float)
    I = np.array([r[1] for r in raw], float)
    days = [r[2] for r in raw]
    kc = np.array([r[3] for r in raw], int) + 1          # number of child orders (1-indexed)
    V = np.array([sig.get((ticker, d), {}).get('V', np.nan) for d in days], float)

    x = np.log(Q / V)
    li = np.log(I)
    ok = np.isfinite(x) & np.isfinite(li)
    x, li, kc = x[ok], li[ok], kc[ok]
    days = [d for d, o in zip(days, ok) if o]
    kmax = int(min(args.kmax, kc.max()))

    # per-day log(sigma) for each method, indexed by a compact day id
    uday = sorted(set(days))
    didx = {d: i for i, d in enumerate(uday)}
    d_of_pt = np.array([didx[d] for d in days], int)
    LSIG = {}
    for mth in METHODS:
        col = []
        for d in uday:
            s = sig.get((ticker, d), {}).get(mth, np.nan)
            col.append(float(np.log(s)) if (s is not None and np.isfinite(s) and s > 0) else None)
        LSIG[mth] = col

    print(f'{args.exp}: {len(x)} points, kmax={kmax}, days={len(uday)}, methods={METHODS}')

    PTS = dict(x=[round(float(v), 4) for v in x],
               li=[round(float(v), 4) for v in li],
               k=[int(v) for v in kc],
               d=[int(v) for v in d_of_pt])
    DATA = json.dumps(dict(pts=PTS, lsig=LSIG, methods=METHODS, exp=args.exp, kmax=kmax))

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'beta_explorer', f'beta_explorer_{args.exp}.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    html = _TEMPLATE.replace('%%PLOTLYJS%%', get_plotlyjs()).replace('%%DATA%%', DATA)
    with open(out, 'w') as f:
        f.write(html)
    print(f'  saved -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)')


_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>exact-k β grid</title>
<script>%%PLOTLYJS%%</script>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
 #bar{padding:10px 16px;background:#fff;border-bottom:1px solid #ddd;position:sticky;top:0;z-index:5}
 #bar h2{margin:0 0 6px;font-size:15px}
 .ctl{display:inline-flex;align-items:center;gap:8px;margin-right:18px}
 #kval{font-weight:700;color:#C0392B;min-width:2.5em;display:inline-block}
 button{border:1px solid #bbb;background:#f3f3f3;border-radius:5px;padding:3px 9px;cursor:pointer}
 button:hover{background:#e8e8e8}
 #readout{font-size:13px;margin-top:6px;font-family:ui-monospace,monospace}
 #readout b{color:#1A5276}
 input[type=range]{width:420px;vertical-align:middle}
 #grid{display:grid;grid-template-columns:90px repeat(5,1fr);gap:2px;padding:6px}
 .hd{display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;
     background:#eef2f7;border-radius:4px;padding:4px;text-align:center}
 .rl{display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;
     writing-mode:vertical-rl;transform:rotate(180deg);background:#eef2f7;border-radius:4px}
 .cell{height:230px;min-width:0}
</style></head><body>
<div id="bar">
 <h2 id="title">exact-k β grid</h2>
 <div class="ctl">k (child orders) =
   <input type="range" id="kslider" min="1" max="100" value="1">
   <span id="kval">1</span></div>
 <div class="ctl">quick:
   <button onclick="setk(1)">1</button><button onclick="setk(2)">2</button>
   <button onclick="setk(3)">3</button><button onclick="setk(4)">4</button>
   <button onclick="setk(10)">10</button><button onclick="setk(50)">50</button>
   <button onclick="setk(100)">100</button></div>
 <div id="readout"></div>
</div>
<div id="grid"></div>
<script>
const D = %%DATA%%;
const P = D.pts, LSIG = D.lsig, METHODS = D.methods, KMAX = D.kmax;
const LABEL = {parkinson:'Parkinson', garman_klass:'Garman-Klass', rogers_satchell:'Rogers-Satchell',
               close_to_close:'close-to-close', yang_zhang:'Yang-Zhang'};
let curk = 1;
document.getElementById('title').textContent = D.exp +
  '  —  EXACT-k β grid  ·  rows: impact cloud | β(k) intercept | β(k) origin  ·  cols: 5 σ-methods  ·  one slider';
document.getElementById('kslider').max = KMAX;

function yval(i,m){ const ls = LSIG[m][P.d[i]]; return ls===null? NaN : (P.li[i]-ls); }
const byk = {}; for(let i=0;i<P.k.length;i++){ (byk[P.k[i]] ||= []).push(i); }

// subsample of all points for the faint background (fits still use ALL points)
const BG=[]; const step=Math.max(1,Math.floor(P.x.length/14000));
for(let i=0;i<P.x.length;i+=step) BG.push(i);

function fit(idxs,m,intercept){
  let xs=[],ys=[];
  for(const i of idxs){ const yy=yval(i,m); if(isFinite(P.x[i])&&isFinite(yy)){ xs.push(P.x[i]); ys.push(yy);} }
  const n=xs.length;
  if(n<8) return {b:NaN,a:0,n:n,se:NaN,x0:0,x1:0};
  let sx=0,sy=0,sxx=0,sxy=0; for(let j=0;j<n;j++){sx+=xs[j];sy+=ys[j];sxx+=xs[j]*xs[j];sxy+=xs[j]*ys[j];}
  const xmin=Math.min(...xs), xmax=Math.max(...xs);
  if(xmax-xmin<1e-6) return {b:NaN,a:0,n:n,se:NaN,x0:xmin,x1:xmax};
  let b,a;
  if(intercept){ const den=n*sxx-sx*sx; b=(n*sxy-sx*sy)/den; a=(sy-b*sx)/n; }
  else { b=sxy/sxx; a=0; }
  let sse=0; for(let j=0;j<n;j++){ const r=ys[j]-(b*xs[j]+a); sse+=r*r; }
  const dof=intercept?n-2:n-1, sxx_c=intercept?(sxx-sx*sx/n):sxx;
  const se=(dof>0&&sxx_c>0)?Math.sqrt((sse/dof)/sxx_c):NaN;
  return {b:b,a:a,n:n,se:se,x0:xmin,x1:xmax};
}

// precompute β(k) curves + per-k fit lines for every method & both modes
const KS=[]; for(let k=1;k<=KMAX;k++)KS.push(k);
const CURVES={}, FITS={};
METHODS.forEach(m=>{
  CURVES[m]={int:{b:[],lo:[],hi:[]},org:{b:[],lo:[],hi:[]}}; FITS[m]={};
  for(let k=1;k<=KMAX;k++){
    const fi=fit(byk[k]||[],m,true), fo=fit(byk[k]||[],m,false);
    FITS[m][k]={int:fi,org:fo};
    for(const [mode,f] of [['int',fi],['org',fo]]){
      CURVES[m][mode].b.push(isFinite(f.b)?f.b:null);
      CURVES[m][mode].lo.push(isFinite(f.b)&&isFinite(f.se)?f.b-1.96*f.se:null);
      CURVES[m][mode].hi.push(isFinite(f.b)&&isFinite(f.se)?f.b+1.96*f.se:null);
    }
  }
});

// build the DOM grid
const grid=document.getElementById('grid');
const ROWS=[{k:'cloud',l:'impact cloud'},{k:'bint',l:'β · intercept'},{k:'borg',l:'β · origin'}];
const CLOUD={}, BINT={}, BORG={};
grid.appendChild(Object.assign(document.createElement('div'),{className:'hd',textContent:''}));
METHODS.forEach(m=>grid.appendChild(Object.assign(document.createElement('div'),{className:'hd',textContent:LABEL[m]})));
ROWS.forEach(row=>{
  grid.appendChild(Object.assign(document.createElement('div'),{className:'rl',textContent:row.l}));
  METHODS.forEach(m=>{
    const d=document.createElement('div'); d.className='cell'; d.id=row.k+'_'+m; grid.appendChild(d);
    if(row.k==='cloud') CLOUD[m]=d.id; else if(row.k==='bint') BINT[m]=d.id; else BORG[m]=d.id;
  });
});

const AX={margin:{l:34,r:6,t:6,b:22},showlegend:false,font:{size:9}};
function makeCloud(id,m){
  Plotly.newPlot(id,[
    {x:BG.map(i=>P.x[i]), y:BG.map(i=>yval(i,m)), mode:'markers',type:'scattergl',
      marker:{size:2,color:'#cfd8e3',opacity:0.30},hoverinfo:'skip'},
    {x:[],y:[],mode:'markers',type:'scattergl',marker:{size:4,color:'#C0392B',opacity:0.7},hoverinfo:'skip'},
    {x:[],y:[],mode:'lines',line:{color:'#000',width:2}},
    {x:[],y:[],mode:'lines',line:{color:'#888',width:1.5,dash:'dash'}}
  ],Object.assign({},AX,{xaxis:{title:''},yaxis:{title:''}}),{responsive:true,displayModeBar:false});
}
function makeCurve(id,m,mode){
  const C=CURVES[m][mode];
  Plotly.newPlot(id,[
    {x:KS.concat(KS.slice().reverse()),y:C.hi.concat(C.lo.slice().reverse()),fill:'toself',
      fillcolor:'rgba(31,93,163,0.12)',line:{color:'rgba(0,0,0,0)'},hoverinfo:'skip'},
    {x:KS,y:C.b,mode:'lines',line:{color:'#2F5DA3',width:1.5}},
    {x:[1],y:[C.b[0]],mode:'markers',marker:{size:10,color:'#C0392B',symbol:'circle-open',line:{width:2.5}}}
  ],Object.assign({},AX,{yaxis:{range:[-0.6,1.0]},xaxis:{title:''},
     shapes:[{type:'line',x0:1,x1:KMAX,y0:0.5,y1:0.5,line:{color:'red',dash:'dash',width:0.7}}]}),
   {responsive:true,displayModeBar:false});
}
METHODS.forEach(m=>{ makeCloud(CLOUD[m],m); makeCurve(BINT[m],m,'int'); makeCurve(BORG[m],m,'org'); });

function update(k){
  curk=k; const idxs=byk[k]||[];
  METHODS.forEach(m=>{
    Plotly.restyle(CLOUD[m],{x:[idxs.map(i=>P.x[i])],y:[idxs.map(i=>yval(i,m))]},[1]);
    const fi=FITS[m][k];
    if(fi.int&&isFinite(fi.int.b)) Plotly.restyle(CLOUD[m],{x:[[fi.int.x0,fi.int.x1]],
        y:[[fi.int.b*fi.int.x0+fi.int.a, fi.int.b*fi.int.x1+fi.int.a]]},[2]);
    else Plotly.restyle(CLOUD[m],{x:[[]],y:[[]]},[2]);
    if(fi.org&&isFinite(fi.org.b)) Plotly.restyle(CLOUD[m],{x:[[fi.org.x0,fi.org.x1]],
        y:[[fi.org.b*fi.org.x0, fi.org.b*fi.org.x1]]},[3]);
    else Plotly.restyle(CLOUD[m],{x:[[]],y:[[]]},[3]);
    Plotly.restyle(BINT[m],{x:[[k]],y:[[CURVES[m].int.b[k-1]]]},[2]);
    Plotly.restyle(BORG[m],{x:[[k]],y:[[CURVES[m].org.b[k-1]]]},[2]);
  });
  document.getElementById('kval').textContent=k;
  const fmt=(m,mode)=>{ const v=CURVES[m][mode].b[k-1]; return v==null?' — ':(v>=0?' ':'')+v.toFixed(2); };
  const ro = 'k='+k+'  n='+((byk[k]||[]).length)+
    '   intercept β:  '+METHODS.map(m=>LABEL[m].slice(0,4)+'='+fmt(m,'int')).join('  ')+
    '\norigin    β:  '+METHODS.map(m=>LABEL[m].slice(0,4)+'='+fmt(m,'org')).join('  ');
  document.getElementById('readout').textContent=ro;
}
function setk(k){ k=Math.max(1,Math.min(KMAX,k)); document.getElementById('kslider').value=k; update(k); }
document.getElementById('kslider').addEventListener('input',e=>update(+e.target.value));
document.addEventListener('keydown',e=>{ const s=document.getElementById('kslider');
  if(e.key==='ArrowRight')setk(+s.value+1); if(e.key==='ArrowLeft')setk(+s.value-1); });
update(1);
</script></body></html>"""


if __name__ == '__main__':
    main()
