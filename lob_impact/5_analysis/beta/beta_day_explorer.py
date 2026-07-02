#!/usr/bin/env python3
"""
Interactive β-cloud DAY explorer (EA).  One self-contained Plotly HTML where the impact regression
cloud  y = log(I/σ)  vs  x = log(Q/V)  is shown PER DAY as colour-coded clouds, with three day
display methods the user can switch between live:

  * SPECTRUM   — every day coloured on a Turbo gradient (day 1 → day 20), so the day-structure of the
                 cloud is visible at a glance ("interval 1..20 as a spectrum").
  * ACTIVE/RED — days inside the selected [d_lo, d_hi] range are RED (these build the regression),
                 the rest are grey.
  * dimmed     — days OUTSIDE the active range are always greyed/low-opacity (inactive).

Controls (JS-driven, no server): model buttons · σ-method (Parkinson / σ=1) · day-range slider
(d_lo..d_hi) · k-mode (all / ≤k / ==k) + k slider · colour-mode (spectrum / active-red).
The OLS β (free intercept) is recomputed live over the ACTIVE points and drawn as a line + printed.

  python beta_day_explorer.py --grid <root> --stock EA --daily <daily_h_l_all.csv> \
         --models Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k --out beta_day_explorer_EA.html
"""
import os, json, argparse
import numpy as np
import matplotlib
from beta_grid import collect
from vol_estimators import daily_sigmas
import plotly.offline as pyo

MODEL_COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Hawkes': '#D4AC0D',
                'CST': '#27AE60', 'Mamba3': '#2F5DA3', 'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22'}


def reds_hex(n):
    """pale (early day) -> saturated dark red (late day). Start at 0.22 so day 0 is faint-but-visible."""
    cm = matplotlib.colormaps['Reds']
    return [matplotlib.colors.to_hex(cm(0.22 + 0.78 * i / max(1, n - 1))) for i in range(n)]


def turbo_hex(n):
    cm = matplotlib.colormaps['turbo']
    return [matplotlib.colors.to_hex(cm(i / max(1, n - 1))) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--daily', required=True)
    ap.add_argument('--models', default='Historic,Heuristic,Hawkes,CST,Mamba3,Mamba3_4k')
    ap.add_argument('--maxpts', type=int, default=7000, help='subsample points per model for the cloud (slope is stable)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    models = [m for m in args.models.split(',') if m]
    rng = np.random.default_rng(0)

    alldays = sorted({d for (st, d) in sig.keys() if st == args.stock})
    didx = {d: i for i, d in enumerate(alldays)}

    D = {'stock': args.stock, 'models': [], 'colors': {}, 'days': alldays, 'pts': {}}
    for model in models:
        raw = collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'buy'), args.stock, +1) + \
              collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'sell'), args.stock, -1)
        if not raw:
            print(f'{model}: no data (skipped)'); continue
        Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
        days = [r[2] for r in raw]; k1 = np.array([r[3] for r in raw], int) + 1
        V = np.array([sig.get((args.stock, d), {}).get('V', np.nan) for d in days], float)
        sg = np.array([sig.get((args.stock, d), {}).get('parkinson', np.nan) for d in days], float)
        x = np.log(Q / V)
        yP = np.log(I / sg)            # σ = parkinson
        yS = np.log(I)                 # σ = 1
        di = np.array([didx.get(d, -1) for d in days], int)
        ok = np.isfinite(x) & np.isfinite(yP) & np.isfinite(yS) & (di >= 0)
        x, yP, yS, di, k1 = x[ok], yP[ok], yS[ok], di[ok], k1[ok]
        if x.size == 0:
            print(f'{model}: empty after filter'); continue
        if x.size > args.maxpts:                       # proportional subsample keeps every day visible
            idx = rng.choice(x.size, args.maxpts, replace=False)
            x, yP, yS, di, k1 = x[idx], yP[idx], yS[idx], di[idx], k1[idx]
        D['pts'][model] = {'x': [round(float(v), 4) for v in x],
                           'yP': [round(float(v), 4) for v in yP],
                           'yS': [round(float(v), 4) for v in yS],
                           'd': [int(v) for v in di], 'k': [int(v) for v in k1]}
        D['colors'][model] = MODEL_COLORS.get(model, '#444'); D['models'].append(model)
        print(f'{model}: {x.size} pts, days {di.min()}..{di.max()}')

    if not D['models']:
        print('no models with data'); return
    D['dayhex'] = reds_hex(len(alldays))      # pale day0 -> saturated red day20
    D['dayturbo'] = turbo_hex(len(alldays))   # alt rainbow palette (toggle)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'beta_day_explorer', f'beta_day_explorer_{args.stock}.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    html = PAGE.replace('%%PLOTLYJS%%', pyo.get_plotlyjs()).replace('%%DATA%%', json.dumps(D))
    open(out, 'w').write(html)
    print(f'saved -> {out}  ({len(D["models"])} models, {len(alldays)} days)')


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>β day explorer</title>
<script>%%PLOTLYJS%%</script>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
 #bar{padding:8px 12px;background:#fff;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:14px;align-items:center}
 .ctl{display:flex;gap:5px;align-items:center}
 button{border:1px solid #bbb;background:#f3f3f3;border-radius:5px;padding:3px 9px;cursor:pointer;font-size:13px}
 button.on{color:#fff;border-color:#222;background:#2F5DA3}
 #cmode button.on{background:#C0392B;border-color:#7d2519}
 #plot{width:100vw;height:86vh}
 b{color:#C0392B}
 input[type=range]{vertical-align:middle}
</style></head><body>
<div id="bar">
 <div class="ctl">model: <span id="models"></span></div>
 <div class="ctl">σ: <span id="smode"></span></div>
 <div class="ctl">days: <input type="range" id="dlo" min="0" value="0"> <b id="dlov"></b>
      .. <input type="range" id="dhi" min="0"> <b id="dhiv"></b></div>
 <div class="ctl">k-mode: <span id="kmode"></span> k=<input type="range" id="ksl" min="1" max="100" value="100"><b id="kv">100</b></div>
 <div class="ctl">colour: <span id="cmode"></span></div>
 <div class="ctl">β(active)=<b id="beta">–</b> <span id="npts" style="color:#888"></span></div>
</div>
<div id="plot"></div>
<script>
const D=%%DATA%%;
const MODELS=D.models, COL=D.colors, DAYS=D.days, DHEX=D.dayhex, DTURBO=D.dayturbo, ND=DAYS.length;
let sel=MODELS[0], smode='yP', dlo=0, dhi=ND-1, kmode='all', curk=100, cmode='reds';
function hexrgba(h,a){const n=parseInt(h.slice(1),16);return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;}

function mkbtns(host,items,cur,cb){const h=document.getElementById(host);h.innerHTML='';
 items.forEach(([v,t])=>{const b=document.createElement('button');b.textContent=t;b.dataset.v=v;
  if(v===cur)b.classList.add('on');
  b.onclick=()=>{[...h.children].forEach(c=>c.classList.toggle('on',c.dataset.v===v));cb(v);};h.appendChild(b);});}

mkbtns('models',MODELS.map(m=>[m,m]),sel,v=>{sel=v;draw();});
mkbtns('smode',[['yP','Parkinson'],['yS','σ=1']],smode,v=>{smode=v;draw();});
mkbtns('kmode',[['all','all'],['le','≤k'],['eq','==k']],kmode,v=>{kmode=v;draw();});
mkbtns('cmode',[['reds','red ramp (day0 faint→day20 deep)'],['turbo','rainbow']],cmode,v=>{cmode=v;draw();});

const dloS=document.getElementById('dlo'),dhiS=document.getElementById('dhi');
dloS.max=ND-1; dhiS.max=ND-1; dhiS.value=ND-1;
function dlbl(){document.getElementById('dlov').textContent=DAYS[dlo]; document.getElementById('dhiv').textContent=DAYS[dhi];}
dloS.oninput=()=>{dlo=Math.min(+dloS.value,dhi);dloS.value=dlo;dlbl();draw();};
dhiS.oninput=()=>{dhi=Math.max(+dhiS.value,dlo);dhiS.value=dhi;dlbl();draw();};
const ksl=document.getElementById('ksl');
ksl.oninput=()=>{curk=+ksl.value;document.getElementById('kv').textContent=curk;draw();};
dlbl();

function olsfit(xs,ys){const n=xs.length; if(n<8)return null;
 let sx=0,sy=0,sxx=0,sxy=0; for(let i=0;i<n;i++){sx+=xs[i];sy+=ys[i];sxx+=xs[i]*xs[i];sxy+=xs[i]*ys[i];}
 const dn=n*sxx-sx*sx; if(Math.abs(dn)<1e-9)return null;
 const b=(n*sxy-sx*sy)/dn, a=(sy-b*sx)/n; return {b,a};}

function draw(){
 const P=D.pts[sel], X=P.x, Y=P[smode], DD=P.d, KK=P.k, n=X.length;
 const kactive=(k)=> kmode==='all'?true : kmode==='le'? k<=curk : k===curk;
 // active = day in [dlo,dhi] AND k-mode satisfied
 const PAL = cmode==='turbo' ? DTURBO : DHEX;
 const ax=[],ay=[],acolor=[], gx=[],gy=[];   // active (day-coloured) vs grey (inactive)
 const fx=[],fy=[];
 for(let i=0;i<n;i++){
   const inday = DD[i]>=dlo && DD[i]<=dhi, ink=kactive(KK[i]);
   if(inday && ink){ ax.push(X[i]); ay.push(Y[i]);
     // weak(day0)->strong(dayN): alpha ramps with day index on top of the pale->deep palette
     const a = ND>1 ? 0.25 + 0.6*DD[i]/(ND-1) : 0.7;
     acolor.push(hexrgba(PAL[DD[i]], a));
     fx.push(X[i]); fy.push(Y[i]); }
   else { gx.push(X[i]); gy.push(Y[i]); }
 }
 const traces=[
   {x:gx,y:gy,mode:'markers',type:'scattergl',name:'inactive',
    marker:{color:'#cfcfcf',size:3,opacity:0.35},hoverinfo:'skip'},
   {x:ax,y:ay,mode:'markers',type:'scattergl',name:'active (by day)',
    marker:{color:acolor,size:5},hoverinfo:'skip'},
 ];
 const f=olsfit(fx,fy);
 document.getElementById('npts').textContent='('+fx.length+' active pts)';
 if(f){const xs=fx.slice().sort((p,q)=>p-q),x0=xs[0],x1=xs[xs.length-1];
   traces.push({x:[x0,x1],y:[f.a+f.b*x0,f.a+f.b*x1],mode:'lines',name:'β fit',
     line:{color:'#111',width:3}});
   document.getElementById('beta').textContent=f.b.toFixed(3);
 } else document.getElementById('beta').textContent='–';
 // √-law reference slope 0.5 line through the active cloud centroid
 if(f){const mx=fx.reduce((a,b)=>a+b,0)/fx.length, my=fy.reduce((a,b)=>a+b,0)/fy.length;
   const xs=fx.slice().sort((p,q)=>p-q),x0=xs[0],x1=xs[xs.length-1];
   traces.push({x:[x0,x1],y:[my+0.5*(x0-mx),my+0.5*(x1-mx)],mode:'lines',name:'√-law β=0.5',
     line:{color:'red',width:1.5,dash:'dash'}});}
 const siglab = smode==='yP' ? 'log(I/σ_parkinson)' : 'log(I)  (σ=1)';
 Plotly.react('plot',traces,{
   title:`${D.stock} · ${sel} — impact cloud by day (${cmode}), days ${DAYS[dlo]}…${DAYS[dhi]}, k-mode ${kmode}`,
   xaxis:{title:'log(Q / V)'},yaxis:{title:siglab},
   showlegend:true,margin:{l:70,r:20,t:40,b:50},
   uirevision:'keep',legend:{orientation:'h'}},{responsive:true});
}
draw();
</script></body></html>"""


if __name__ == '__main__':
    main()
