#!/usr/bin/env python3
"""
Interactive β–k explorer: ONE self-contained Plotly HTML, SIX panels in a 2×3 grid, one shared k slider.

  TOP ROW  (3 panels) — β(k) curves, ALL models overlaid: ≤k CUMULATIVE | ==k EXACT | ≥k REVERSE-CUMUL.
        A vertical cursor marks the current k.
  BOTTOM ROW (3 panels) — the impact regression CLOUD of the SELECTED model, one per view, filtered at the
        current k (kc≤k / kc==k / kc≥k); active points coloured by DAY (faint→deep red), the rest grey.

CRITICAL: β (the curves AND the per-panel β/fit-line) is computed in PYTHON on the FULL cloud (every
(sample,insertion) point, I>0), NOT on the displayed dots.  The scatter is a per-model SUBSAMPLE for
rendering only (a full 100k-point cloud is too heavy for a browser).  Every panel prints the REAL n it
was fit on (e.g. ==k uses ~1000-1600 pts/k for the big models, ~330/k for the small partial Mamba3_4k),
so the displayed sparse dots are never mistaken for the sample size.

Controls: model · σ (Parkinson / σ=1) · k slider · trim (0 / 5 % of impact I) · day palette (red / rainbow).

  python beta_k_explorer.py --grid <root> --stock EA --daily <daily_h_l_all.csv> --out beta_k_explorer_EA.html
"""
import os, json, argparse
import numpy as np
import matplotlib
from beta_grid import collect
from vol_estimators import daily_sigmas
import plotly.offline as pyo

MODEL_COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Hawkes': '#D4AC0D',
                'CST': '#27AE60', 'Mamba3': '#2F5DA3', 'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22'}
MINPTS = 8


def ramp_hex(name, n, lo, hi):
    cm = matplotlib.colormaps[name]
    return [matplotlib.colors.to_hex(cm(lo + (hi - lo) * i / max(1, n - 1))) for i in range(n)]


def slope(x, y, I, trim):
    """OLS slope on the FULL subset, optionally trimming the [trim,1-trim] tails of impact I first."""
    if trim > 0 and I.size >= 20:
        lo, hi = np.percentile(I, [trim * 100, (1 - trim) * 100])
        keep = (I >= lo) & (I <= hi)
        x, y = x[keep], y[keep]
    if x.size < MINPTS or (x.max() - x.min()) < 1e-6:
        return None
    return round(float(np.polyfit(x, y, 1)[0]), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--stock', default='EA')
    ap.add_argument('--daily', required=True)
    ap.add_argument('--models', default='Historic,Heuristic,OW,CST,Hawkes,QR,Mamba3,Mamba3_4k')
    ap.add_argument('--maxpts', type=int, default=9000, help='points per model EMBEDDED FOR DISPLAY only (β uses full data)')
    ap.add_argument('--kmax', type=int, default=100)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    sig = daily_sigmas(args.daily)
    models = [m for m in args.models.split(',') if m]
    rng = np.random.default_rng(0)
    alldays = sorted({d for (st, d) in sig.keys() if st == args.stock})
    didx = {d: i for i, d in enumerate(alldays)}
    KMAX = args.kmax
    ks = np.arange(1, KMAX + 1)

    D = {'stock': args.stock, 'models': [], 'colors': {}, 'days': alldays, 'kmax': KMAX,
         'pts': {}, 'curves': {}, 'nk': {}}
    for model in models:
        raw = collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'buy'), args.stock, +1) + \
              collect(os.path.join(args.grid, f'{args.stock}-{model}-beta', 'sell'), args.stock, -1)
        if not raw:
            print(f'{model}: no data (skipped)'); continue
        Q = np.array([r[0] for r in raw], float); I = np.array([r[1] for r in raw], float)
        days = [r[2] for r in raw]; k1 = np.array([r[3] for r in raw], int) + 1
        V = np.array([sig.get((args.stock, d), {}).get('V', np.nan) for d in days], float)
        sg = np.array([sig.get((args.stock, d), {}).get('parkinson', np.nan) for d in days], float)
        x = np.log(Q / V); yP = np.log(I / sg); yS = np.log(I)
        di = np.array([didx.get(d, -1) for d in days], int)
        ok = np.isfinite(x) & np.isfinite(yP) & np.isfinite(yS) & np.isfinite(I) & (di >= 0)
        x, yP, yS, di, k1, Iok = x[ok], yP[ok], yS[ok], di[ok], k1[ok], I[ok]
        if x.size == 0:
            print(f'{model}: empty'); continue

        # ---- β curves on FULL data, per σ-method × trim × view ----
        nk = [int((k1 == k).sum()) for k in ks]
        curves = {}
        for smk, yv in (('yP', yP), ('yS', yS)):
            curves[smk] = {}
            for tr in (0.0, 0.05):
                bc, be, br = [], [], []
                for k in ks:
                    for mask, acc in ((k1 <= k, bc), (k1 == k, be), (k1 >= k, br)):
                        acc.append(slope(x[mask], yv[mask], Iok[mask], tr))
                curves[smk][str(tr)] = {'le': bc, 'eq': be, 'ge': br}
        D['curves'][model] = curves
        D['nk'][model] = nk

        # ---- subsample ONLY for the scatter display ----
        xs, yPs, ySs, dis, k1s = x, yP, yS, di, k1
        if x.size > args.maxpts:
            idx = rng.choice(x.size, args.maxpts, replace=False)
            xs, yPs, ySs, dis, k1s = x[idx], yP[idx], yS[idx], di[idx], k1[idx]
        D['pts'][model] = {'x': [round(float(v), 4) for v in xs], 'yP': [round(float(v), 4) for v in yPs],
                           'yS': [round(float(v), 4) for v in ySs], 'd': [int(v) for v in dis], 'k': [int(v) for v in k1s]}
        D['colors'][model] = MODEL_COLORS.get(model, '#444'); D['models'].append(model)
        print(f'{model}: full n={x.size}, ==k median={int(np.median(nk))}/k, display subsample={len(xs)}')

    if not D['models']:
        print('no data'); return
    D['dayhex'] = ramp_hex('Reds', len(alldays), 0.22, 1.0)
    D['dayturbo'] = ramp_hex('turbo', len(alldays), 0.0, 1.0)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'beta_k_explorer', f'beta_k_explorer_{args.stock}.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w').write(PAGE.replace('%%PLOTLYJS%%', pyo.get_plotlyjs()).replace('%%DATA%%', json.dumps(D)))
    print(f'saved -> {out}  ({len(D["models"])} models, {len(alldays)} days)')


PAGE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>β–k explorer</title>
<script>%%PLOTLYJS%%</script>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
 #bar{padding:8px 12px;background:#fff;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:16px;align-items:center;position:sticky;top:0;z-index:9}
 .ctl{display:flex;gap:5px;align-items:center}
 button{border:1px solid #bbb;background:#f3f3f3;border-radius:5px;padding:3px 9px;cursor:pointer;font-size:13px}
 button.on{color:#fff;border-color:#222;background:#2F5DA3}
 #plot{width:100vw;height:90vh}
 b{color:#C0392B}
 input[type=range]{vertical-align:middle;width:240px}
</style></head><body>
<div id="bar">
 <div class="ctl">cloud model: <span id="models"></span></div>
 <div class="ctl">σ: <span id="smode"></span></div>
 <div class="ctl">trim: <span id="tmode"></span></div>
 <div class="ctl">colour: <span id="cmode"></span></div>
 <div class="ctl"><b>k = <span id="kv">100</span></b> <input type="range" id="ksl" min="1" max="100" value="100"></div>
</div>
<div id="plot"></div>
<script>
const D=%%DATA%%;
const MODELS=D.models, COL=D.colors, DAYS=D.days, DHEX=D.dayhex, DTURBO=D.dayturbo, ND=DAYS.length, KMAX=D.kmax;
const VIEWS=[['le','≤k  CUMULATIVE'],['eq','==k  EXACT'],['ge','≥k  REVERSE-CUMUL']];
let sel=MODELS[0], smode='yP', curk=KMAX, trim='0.05', cmode='reds';
function hexrgba(h,a){const n=parseInt(h.slice(1),16);return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;}
function mkbtns(host,items,cur,cb){const h=document.getElementById(host);h.innerHTML='';
 items.forEach(([v,t])=>{const b=document.createElement('button');b.textContent=t;b.dataset.v=v;
  if(v===cur)b.classList.add('on');
  b.onclick=()=>{[...h.children].forEach(c=>c.classList.toggle('on',c.dataset.v===v));cb(v);};h.appendChild(b);});}
mkbtns('models',MODELS.map(m=>[m,m]),sel,v=>{sel=v;drawAll();});
mkbtns('smode',[['yP','Parkinson'],['yS','σ=1']],smode,v=>{smode=v;drawAll();});
mkbtns('tmode',[['0','0%'],['0.05','5%']],trim,v=>{trim=v;drawAll();});
mkbtns('cmode',[['reds','red ramp'],['turbo','rainbow']],cmode,v=>{cmode=v;drawBottom();});
const ksl=document.getElementById('ksl');ksl.max=KMAX;
ksl.oninput=()=>{curk=+ksl.value;document.getElementById('kv').textContent=curk;drawAll();};

const keepFn={le:k=>(kk=>kk<=k),eq:k=>(kk=>kk===k),ge:k=>(kk=>kk>=k)};
function curveOf(model,view){return D.curves[model][smode][trim][view];}      // β(k) full-data, length KMAX
function nOf(model,view,k){const nk=D.nk[model];                              // REAL fit-n at this k
 if(view==='eq')return nk[k-1];
 if(view==='le'){let s=0;for(let i=0;i<k;i++)s+=nk[i];return s;}
 let s=0;for(let i=k-1;i<KMAX;i++)s+=nk[i];return s;}

const XD=[[0.0,0.30],[0.36,0.64],[0.70,1.0]];
const YT=[0.58,1.0], YB=[0.0,0.42];
function ax(col,row){const i=row*3+col+1; return i===1?'':String(i);}
function drawAll(){drawTop();drawBottom();}

function drawTop(){
 const tr=[];
 MODELS.forEach(m=>VIEWS.forEach(([v,_],c)=>{const id=ax(c,0);
   tr.push({x:Array.from({length:KMAX},(_,i)=>i+1),y:curveOf(m,v),mode:'lines',
     name:m,legendgroup:m,showlegend:(c===0),line:{color:COL[m],width:2},xaxis:'x'+id,yaxis:'y'+id});}));
 window._top=tr;reactAll();}

function drawBottom(){
 const P=D.pts[sel],X=P.x,Y=P[smode],DD=P.d,KK=P.k,N=X.length,PAL=cmode==='turbo'?DTURBO:DHEX;
 const tr=[];window._lbl=[];
 VIEWS.forEach(([v,_],c)=>{const id=ax(c,1),keep=keepFn[v](curk);
  const axx=[],ayy=[],ac=[],gx=[],gy=[];
  for(let i=0;i<N;i++){ if(keep(KK[i])){axx.push(X[i]);ayy.push(Y[i]);
     ac.push(hexrgba(PAL[DD[i]], ND>1?0.25+0.6*DD[i]/(ND-1):0.7));}
    else{gx.push(X[i]);gy.push(Y[i]);}}
  tr.push({x:gx,y:gy,mode:'markers',type:'scattergl',showlegend:false,hoverinfo:'skip',
     marker:{color:'#d2d2d2',size:3,opacity:0.3},xaxis:'x'+id,yaxis:'y'+id});
  tr.push({x:axx,y:ayy,mode:'markers',type:'scattergl',showlegend:false,hoverinfo:'skip',
     marker:{color:ac,size:5},xaxis:'x'+id,yaxis:'y'+id});
  // fit line uses the FULL-DATA β (not the subsample), drawn through the displayed active centroid
  const b=curveOf(sel,v)[curk-1], nreal=nOf(sel,v,curk);
  if(b!==null && axx.length>2){let mx=0,my=0,x0=axx[0],x1=axx[0];
    for(let i=0;i<axx.length;i++){mx+=axx[i];my+=ayy[i];if(axx[i]<x0)x0=axx[i];if(axx[i]>x1)x1=axx[i];}
    mx/=axx.length;my/=axx.length;
    tr.push({x:[x0,x1],y:[my+b*(x0-mx),my+b*(x1-mx)],mode:'lines',showlegend:false,
       line:{color:'#111',width:3},xaxis:'x'+id,yaxis:'y'+id});
    tr.push({x:[x0,x1],y:[my+0.5*(x0-mx),my+0.5*(x1-mx)],mode:'lines',showlegend:false,
       line:{color:'red',width:1.5,dash:'dash'},xaxis:'x'+id,yaxis:'y'+id});}
  window._lbl.push({c,b:(b===null?'–':b.toFixed(3)),n:nreal,shown:axx.length});
 });
 window._bot=tr;reactAll();}

function reactAll(){
 const tr=(window._top||[]).concat(window._bot||[]);
 const siglab=smode==='yP'?'log(I/σ_park)':'log(I) σ=1';
 const lay={margin:{l:55,r:15,t:64,b:40},showlegend:true,uirevision:'k',legend:{orientation:'h',y:1.07,x:0},
   title:`${D.stock} — β(k) on FULL data (top) · ${sel} cloud subsample filtered at k=${curk} (bottom) · σ=${smode==='yP'?'parkinson':'1'} · trim ${trim==='0'?'0':'5'}%`,
   shapes:[],annotations:[]};
 VIEWS.forEach(([v,t],c)=>{const tid=ax(c,0),bid=ax(c,1);
   lay['xaxis'+tid]={domain:XD[c],anchor:'y'+tid,range:[1,KMAX],title:(c===1?'insertion k':'')};
   lay['yaxis'+tid]={domain:YT,anchor:'x'+tid,range:[-0.2,1.0],title:(c===0?'β (full data)':'')};
   lay['xaxis'+bid]={domain:XD[c],anchor:'y'+bid,title:'log(Q/V)'};
   lay['yaxis'+bid]={domain:YB,anchor:'x'+bid,title:(c===0?siglab:'')};
   lay.shapes.push({type:'line',xref:'x'+tid,yref:'y'+tid,x0:1,x1:KMAX,y0:0.5,y1:0.5,line:{color:'red',dash:'dash',width:1}});
   lay.shapes.push({type:'line',xref:'x'+tid,yref:'y'+tid,x0:curk,x1:curk,y0:-0.2,y1:1.0,line:{color:'#111',width:1.5,dash:'dot'}});
   lay.annotations.push({xref:'paper',yref:'paper',x:(XD[c][0]+XD[c][1])/2,y:1.0,showarrow:false,text:'<b>'+t+'</b>',font:{size:12}});
 });
 (window._lbl||[]).forEach(L=>{
   lay.annotations.push({xref:'paper',yref:'paper',x:XD[L.c][0]+0.005,y:YB[1]-0.005,xanchor:'left',yanchor:'top',
     showarrow:false,align:'left',font:{size:11,color:'#111'},bgcolor:'rgba(255,255,255,0.7)',
     text:`β=${L.b}<br><b>n=${L.n.toLocaleString()}</b> (real)<br><span style="color:#999">${L.shown} shown</span>`});
 });
 Plotly.react('plot',tr,lay,{responsive:true});
}
drawAll();
</script></body></html>"""


if __name__ == '__main__':
    main()
