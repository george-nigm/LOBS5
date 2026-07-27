#!/usr/bin/env python3
"""
Figure 5 combined: 2x2 panel (rows = build-up / relaxation shapes; cols = k-clock
impact in bps / gated master curve) with ONE shared legend UNDER all four panels.
Pure cache re-render: mid_trajectory_<ST>_{beta,decay}.npz + master_curve_<ST>_
{beta,relaxation}_v2gated.npz. No grid access, nothing recomputed.

  python fig5_combined.py --stock NVDA [--copy_to <paper/Figures dir>]
"""
import os, argparse, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def _legend_below(fig, handles, labels, y=-0.04):
    """Shared legend rule (lob_impact/core/model_style.py): 3 entries per column, one size for
    every figure. Falls back to a plain bottom legend if the module is unreachable."""
    import os, importlib.util, math
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        p = os.path.join(d, 'core', 'model_style.py')
        if os.path.exists(p):
            sp = importlib.util.spec_from_file_location('_lob_model_style', p)
            m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
            return m.legend_below(fig, handles, labels, y=y)
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return fig.legend(handles, labels, loc='lower center', frameon=False, fontsize=9.5,
                      ncol=max(1, math.ceil(len(labels) / 3)), bbox_to_anchor=(0.5, y))


# OW (Obizhaeva-Wang exponential-resilience kernel) and QR (queue-reactive) sit with the
# other mechanical baselines, between Propagator and the point-process models.
ORDER = ['Historic', 'Heuristic', 'Propagator', 'OW', 'CST', 'NMZI', 'Hawkes', 'QR', 'S5', 'S5_120M', 'S5_4k', 'Mamba3', 'Mamba3_4k', 'GDN']
COLORS = {'Historic': '#C0392B', 'Heuristic': '#7F8C8D', 'Propagator': '#8B5E3C',
          'OW': '#6A1B9A', 'QR': '#00ACC1',
          'Hawkes': '#D4AC0D', 'CST': '#27AE60', 'NMZI': '#117864',
          'Mamba3': '#2F5DA3', 'GDN': '#D81B60', 'S5_120M': '#F06292',
          'Mamba3_4k': '#16A085', 'S5_4k': '#E67E22', 'S5': '#5D6D7E'}
GREY = '#666666'


def models_in(z, suffix):
    ms = {k[:-len(suffix)] for k in z.files if k.endswith(suffix)}
    return [m for m in ORDER if m in ms]


def n_of(z, m):
    """Samples behind model m's curve. `_cnt` is per-k; report the k=0 count (the
    full fleet) — later k can only lose samples to short runs."""
    if f'{m}_cnt' in z.files:
        c = np.atleast_1d(z[f'{m}_cnt']).ravel()
        if c.size:
            return int(c[0])
    if f'{m}_K' in z.files:
        return int(np.shape(z[f'{m}_K'])[0])
    # master-curve caches store the fleet size as `{m}_n` — without this a model that
    # exists only there (OW, QR) is legended as "n = 0".
    if f'{m}_n' in z.files:
        return int(np.atleast_1d(z[f'{m}_n']).ravel()[0])
    return 0


def robust_ylim(z, pad=0.10, blowup=20.0):
    """y-range that survives a pathological model.

    On AMD a handful of NMZI samples run the mid price to -4600 bps, which on a
    shared axis flattens every other curve onto y=0. Set the limits from the
    models whose extreme is within `blowup`x the median extreme; the outlier is
    still drawn (it just runs off the panel) and is named in the annotation, so
    nothing is silently hidden.
    """
    ext, span = {}, {}
    for m in models_in(z, '_k_mean'):
        y = np.asarray(z[f'{m}_k_mean'], float)
        y = y[np.isfinite(y)]
        if not y.size:
            continue
        ext[m] = float(np.max(np.abs(y)))
        span[m] = (float(np.min(y)), float(np.max(y)))
    if not ext:
        return None, []
    med = float(np.median(list(ext.values())))
    thr = max(blowup * med, 1e-9)
    keep = [m for m in ext if ext[m] <= thr]
    clipped = [m for m in ext if ext[m] > thr]
    if not keep:
        return None, []
    lo = min(span[m][0] for m in keep); hi = max(span[m][1] for m in keep)
    rng = max(hi - lo, 1e-9)
    return (lo - pad * rng, hi + pad * rng), clipped


# Bands are +-2 s.e. and deliberately faint: with 13 models overlapping, an alpha that reads well
# for one curve stacks into an opaque wash. They stay wide for Mamba3-4k and GDN because those
# distributions are genuinely heavy-tailed — that width is the honest uncertainty of a mean over a
# fat tail, and it is the reason the collapse threshold is not lowered (dropping the tail would
# shrink the band by changing the estimand, not by measuring better).
def draw_mid(ax, z, n_ins=None, ref=None, band_within=None):
    """band_within=(lo,hi): draw the +-2 s.e. ribbon only for curves that live inside that
    range. With --swap_zoom the three runaway curves leave the frame but their ribbons stay,
    filling the panel with translucent colour and washing out exactly the curves the zoom was
    meant to expose."""
    klen = 0
    for m in models_in(z, '_k_mean'):
        y = z[f'{m}_k_mean']; se = z[f'{m}_k_se']
        x = np.arange(len(y))
        klen = max(klen, len(y))
        c = COLORS.get(m, '#444444')
        ax.plot(x, y, color=c, lw=2.0)
        inside = True
        if band_within is not None and np.isfinite(y).any():
            pk = np.nanmax(np.abs(np.asarray(y, float)))
            inside = pk <= max(abs(band_within[0]), abs(band_within[1]))
        if inside:
            ax.fill_between(x, y - 2 * se, y + 2 * se, color=c, alpha=0.07, lw=0)
    if ref is not None:
        ax.plot(ref[0], ref[1], color=GREY, lw=2.2, ls='--')
    elif 'sqrt_x' in z.files and klen:
        # cache stores the reference against MESSAGE position; the model curves
        # are on the k-clock -> map the reference onto the same k axis
        ys = z['sqrt_y']
        xref = np.linspace(0, klen - 1, len(ys))
        ax.plot(xref, ys, color=GREY, lw=2.0, ls='--')
    if n_ins is not None:
        ax.axvline(n_ins, color='#999999', lw=1.0, ls=':')
    ax.axhline(0, color='#cccccc', lw=0.8)
    ax.margins(x=0)


def stamp_if_empty(ax, z):
    """A cache with no per-model curves renders as a BLANK panel inside an otherwise complete
    figure, legend and all — indistinguishable from a plotting glitch and easy to paste into a
    paper unnoticed (it happened to the GOOG build-up panel after the SHAPE/TAG bug wiped that
    cache on 2026-07-27). Testing ax.get_lines() does not work: the zero line and the reference
    curve are lines too. Test the cache."""
    if not models_in(z, '_k_mean'):
        ax.text(0.5, 0.5, 'no model curves in this cache', transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color='#B03A2E',
                bbox=dict(boxstyle='round', fc='#FDEDEC', ec='#B03A2E', alpha=0.92), zorder=10)
        return True
    return False




def warn_if_unfiltered(ax, z, bound=100.0):
    """Stamp the panel if a curve carries collapse-level values.

    mid_trajectory.py drops rollouts past 2000 bps at source and fix_collapsed_runs.py repairs
    caches already written, but a job launched before those fixes rewrites the cache unfiltered —
    which happened on AMD (a single GDN rollout of +8668 bps lifted the k=96 point from 7.07 to
    11.23 bps, and two NMZI rollouts put the whole curve at -4592). A mid-price impact past `bound`
    bps is not a market move, so say so on the figure rather than let it be pasted unnoticed.
    """
    bad = [m for m in models_in(z, '_k_mean')
           if np.isfinite(np.asarray(z[f'{m}_k_mean'], float)).any()
           and np.nanmax(np.abs(np.asarray(z[f'{m}_k_mean'], float))) > bound]
    if bad:
        ax.text(0.5, 0.94, 'UNFILTERED CACHE: ' + ', '.join(bad) + '  — run fix_collapsed_runs.py',
                transform=ax.transAxes, ha='center', va='top', fontsize=10, color='#B03A2E',
                bbox=dict(boxstyle='round', fc='#FDEDEC', ec='#B03A2E', alpha=0.95), zorder=12)
    return bad


def zoom_band(z, drop_top=3, pad=1.30):
    """Y-range that shows everything except the `drop_top` largest curves.

    The point of the swap is that the three runaway neural curves set the scale and flatten
    everything else: on AMD the build-up panel spans 0-72 bps while S5-120M peaks at 17 and the
    baselines sit under 1.5. Excluding the top few by peak leaves a range in which S5-120M, S5-4k,
    the reference law and the baseline band are all legible; the big picture moves to the inset.
    """
    peaks = []
    for m in models_in(z, '_k_mean'):
        y = np.asarray(z[f'{m}_k_mean'], float)
        if np.isfinite(y).any():
            peaks.append(np.nanmax(np.abs(y)))
    if not peaks:
        return None
    peaks.sort()
    keep = peaks[:-drop_top] if len(peaks) > drop_top else peaks
    hi = (keep[-1] if keep else peaks[-1]) * pad
    lo = -0.18 * hi
    return (lo, hi)


def inset_zoom(ax, z, ylim=(-4.5, 5.5), rect=(0.06, 0.52, 0.44, 0.44), ref=None, title=None):
    """Small axis inside `ax`. Default: the baseline band the neural curves dwarf.
    With --swap_zoom it carries the full-range view instead and the main axis is the zoom."""
    ins = ax.inset_axes(list(rect))
    for m in models_in(z, '_k_mean'):
        y = z[f'{m}_k_mean']; se = z[f'{m}_k_se']
        x = np.arange(len(y))
        c = COLORS.get(m, '#444444')
        ins.plot(x, y, color=c, lw=1.5)
        ins.fill_between(x, y - 2 * se, y + 2 * se, color=c, alpha=0.07, lw=0)
    if ref is not None:
        ins.plot(ref[0], ref[1], color=GREY, lw=1.8, ls='--')
    elif 'sqrt_x' in z.files:
        ys = z['sqrt_y']
        ins.plot(np.linspace(0, len(z[[k for k in z.files if k.endswith('_k_mean')][0]]) - 1, len(ys)),
                 ys, color=GREY, lw=1.8, ls='--')
    ins.set_ylim(*ylim)
    ins.axhline(0, color='#cccccc', lw=0.7)
    ins.set_title(title or 'zoom: baselines & reference (bps)', fontsize=9)
    ins.tick_params(labelsize=8)


def draw_master(ax, z, relax, zmid=None, n_end=None, faint_frac=0.05):
    """Master curve = I(v)/I(1). Normalising removes scale, which is the point — and also the
    hazard: a model whose absolute impact is near zero has its own measurement noise divided by
    that same near-zero number, so it arrives on the panel as a wildly oscillating line that looks
    like a finding. On AMD, QR peaks at 0.118 bps against GDN's 7.49 and its point-to-point jitter
    is 42% of its own signal where every other curve sits at 0.1--0.4% — 15x the smoothest line.
    It passes the significance gate (3.6 sigma: not zero) but significance is not magnitude.

    Curves whose |peak| falls below `faint_frac` of the panel's largest are therefore drawn thin
    and semi-transparent and named, with their absolute peak, in a note. Nothing is hidden; the
    reader is told which shapes are ratios of almost nothing.
    """
    gated_out, faint = [], []
    peaks = {m: abs(float(z[f'{m}_peak'])) for m in models_in(z, '_master') if f'{m}_peak' in z.files}
    pmax = max(peaks.values()) if peaks else 0.0
    for m in models_in(z, '_master'):
        if not bool(z[f'{m}_sig']):
            gated_out.append(m); continue
        v = z['vgrid']; y = z[f'{m}_master']
        c = COLORS.get(m, '#444444')
        weak = pmax > 0 and peaks.get(m, pmax) < faint_frac * pmax
        if weak:
            faint.append((m, peaks[m]))
            ax.plot(v, y, color=c, lw=1.0, alpha=0.45)
            continue
        ax.plot(v, y, color=c, lw=2.0)
        # +-2SE band via delta method: relative SE of the k-clock mean applied
        # to the master line (the k sampling matches the v grid one-to-one)
        if zmid is not None and f'{m}_k_mean' in zmid.files:
            km = zmid[f'{m}_k_mean']; ks = zmid[f'{m}_k_se']
            n = min(len(km), len(v), len(y))
            with np.errstate(all='ignore'):
                rel = np.abs(ks[:n] / km[:n])
            band = np.abs(y[:n]) * np.clip(rel, 0, 1.5)
            ax.fill_between(v[:n], y[:n] - 2 * band, y[:n] + 2 * band, color=c, alpha=0.06, lw=0)
    v = z['vgrid']
    if relax:
        # theory: sqrt build-up then power-law decay to the 2/3 permanent level
        vv = np.linspace(0.01, v[-1], 300)
        th = np.where(vv <= 1, np.sqrt(vv),
                      2/3 + (1/3) * (np.sqrt(vv) - np.sqrt(np.clip(vv - 1, 0, None))))
        ax.plot(vv, th, color=GREY, lw=2.4, ls='--')
        ax.axhline(2 / 3, color=GREY, lw=1.0, ls=':')
        ax.axvline(1.0, color='#999999', lw=1.0, ls=':')
    else:
        ax.plot(v, np.sqrt(np.clip(v, 0, None)), color=GREY, lw=1.4, ls='--')
    notes = []
    if gated_out:
        notes.append('gate-failed: ' + ', '.join(gated_out))
    if faint:
        notes.append('near-null impact, ratio is noise: '
                     + ', '.join(f'{m.replace("_", "-")} (peak {p:.2f} bps)' for m, p in faint))
    if notes:
        ax.text(0.02, 0.97, '\n'.join(notes), transform=ax.transAxes,
                fontsize=9.5, color='#888888', va='top')
    ax.axhline(0, color='#cccccc', lw=0.8)
    ax.margins(x=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--copy_to', default=None)
    ap.add_argument('--swap_zoom', action='store_true',
                    help='main panel = zoomed band, inset = full range (default is the reverse)')
    ap.add_argument('--drop_top', type=int, default=3,
                    help='--swap_zoom: how many largest curves the zoomed band may exclude')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, 'results')
    src = {
        'mid_beta': os.path.join(res, 'mid_impact', f'mid_trajectory_{args.stock}_beta.npz'),
        'mid_decay': os.path.join(res, 'mid_impact', f'mid_trajectory_{args.stock}_decay.npz'),
        'ms_beta': os.path.join(res, 'master_curve', f'master_curve_{args.stock}_beta_v2gated.npz'),
        'ms_relax': os.path.join(res, 'master_curve', f'master_curve_{args.stock}_relaxation_v2gated.npz'),
    }
    Z = {k: np.load(v, allow_pickle=True) for k, v in src.items()}

    plt.rcParams.update({'font.size': 12.5, 'axes.labelsize': 13, 'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    stamp_if_empty(axes[0][0], Z['mid_beta'])
    warn_if_unfiltered(axes[0][0], Z['mid_beta'])
    draw_mid(axes[0][0], Z['mid_beta'],
             band_within=(zoom_band(Z['mid_beta'], args.drop_top) if args.swap_zoom else None))
    if args.swap_zoom:
        # приближение — в большой панели, общий вид — во врезке
        band = zoom_band(Z['mid_beta'], args.drop_top)
        full = robust_ylim(Z['mid_beta'])[0]
        inset_zoom(axes[0][0], Z['mid_beta'], ylim=full, rect=(0.06, 0.55, 0.40, 0.41),
                   title='full range (bps)')
        if band:
            axes[0][0].set_ylim(*band)
    else:
        inset_zoom(axes[0][0], Z['mid_beta'])
    yl, clipped = robust_ylim(Z['mid_beta'])
    if yl:
        if not args.swap_zoom:
            axes[0][0].set_ylim(*yl)
    if clipped:
        axes[0][0].text(0.98, 0.03, 'off-scale: ' + ', '.join(m.replace('_', '-') for m in clipped),
                        transform=axes[0][0].transAxes, fontsize=10, color='#888888',
                        ha='right', va='bottom')
    axes[0][0].set_title('build-up: mid-price impact $I(k)$ (bps, antisymmetrised)', fontsize=13.5)
    axes[0][0].set_ylabel('build-up shape\n$I$, bps')
    draw_master(axes[0][1], Z['ms_beta'], relax=False, zmid=Z['mid_beta'])
    axes[0][1].set_ylim(-0.05, 1.12)
    axes[0][1].set_title(r'build-up: master curve $\langle I(v)\rangle/\langle I(1)\rangle$ (gated)', fontsize=13.5)
    zd = Z['mid_decay']
    theory = None
    if 'sqrt_y' in zd.files:
        P = float(np.nanmax(zd['sqrt_y']))
        klen = max(len(zd[k]) for k in zd.files if k.endswith('_k_mean'))
        kk = np.linspace(0.01, klen - 1, 400)
        vv = kk / 10.0
        th = np.where(vv <= 1, P * np.sqrt(np.clip(vv, 0, None)),
                      P * (2/3 + (1/3) * (np.sqrt(vv) - np.sqrt(np.clip(vv - 1, 0, None)))))
        theory = (kk, th)
    stamp_if_empty(axes[1][0], Z['mid_decay'])
    warn_if_unfiltered(axes[1][0], Z['mid_decay'])
    draw_mid(axes[1][0], Z['mid_decay'], n_ins=10, ref=theory,
             band_within=(zoom_band(Z['mid_decay'], args.drop_top) if args.swap_zoom else None))
    yl, clipped = robust_ylim(Z['mid_decay'])
    if yl is None:
        yl = axes[1][0].get_ylim()
    if not args.swap_zoom:
        axes[1][0].set_ylim(yl[0], yl[1] * 1.30 if yl[1] > 0 else yl[1])
    if clipped:
        axes[1][0].text(0.98, 0.03, 'off-scale: ' + ', '.join(m.replace('_', '-') for m in clipped),
                        transform=axes[1][0].transAxes, fontsize=10, color='#888888',
                        ha='right', va='bottom')
    if args.swap_zoom:
        band = zoom_band(Z['mid_decay'], args.drop_top)
        full = (yl[0], yl[1] * 1.30 if yl[1] > 0 else yl[1])
        inset_zoom(axes[1][0], Z['mid_decay'], ylim=full, rect=(0.05, 0.60, 0.38, 0.36),
                   ref=theory, title='full range (bps)')
        if band:
            axes[1][0].set_ylim(*band)
    else:
        inset_zoom(axes[1][0], Z['mid_decay'], ylim=(-0.8, 0.9), rect=(0.05, 0.62, 0.38, 0.36), ref=theory)
    axes[1][0].set_title('relaxation: $I(k)$ through execution end (dotted)', fontsize=13.5)
    axes[1][0].set_ylabel('relaxation shape\n$I$, bps')
    axes[1][0].set_xlabel('children executed $k$ (then cooling blocks)')
    draw_master(axes[1][1], Z['ms_relax'], relax=True, zmid=Z['mid_decay'])
    axes[1][1].set_ylim(-0.1, 1.9)
    axes[1][1].set_title(r'relaxation: master curve, $v>1$ = cooling (dashed: $2/3$ level)', fontsize=13.5)
    axes[1][1].set_xlabel('metaorder fraction executed $v$')

    # The legend must cover every model DRAWN anywhere in the figure. Deriving it from the
    # mid-trajectory caches alone silently dropped models that exist only in the (newer)
    # master-curve caches: OW and QR were drawn in the right-hand panels — OW in purple —
    # with no legend entry at all. Take the union over all four caches.
    present = [m for m in ORDER
               if any(f'{m}_k_mean' in Z[k].files for k in ('mid_beta', 'mid_decay'))
               or any(f'{m}_master' in Z[k].files for k in ('ms_beta', 'ms_relax'))]
    # sample count in every legend entry: how much data is behind each curve, so a
    # noisy-looking baseline can be read as "genuinely no impact" rather than "thin n".
    # build-up and relaxation fleets are separate runs -> show both when they differ.
    handles = []
    for m in present:
        nb, nd = n_of(Z['mid_beta'], m), n_of(Z['mid_decay'], m)
        if not (nb or nd):   # drawn only in the master-curve panels — take its count from there
            nb = nd = max(n_of(Z['ms_beta'], m), n_of(Z['ms_relax'], m))
        ns = f'{nb}' if nb == nd else '/'.join(str(v) for v in (nb, nd) if v)
        lbl = f"{m.replace('_', '-')} ($n{{=}}{ns}$)" if ns else m.replace('_', '-')
        handles.append(Line2D([], [], color=COLORS.get(m, '#444444'), lw=3.0, label=lbl))
    handles.append(Line2D([], [], color=GREY, lw=2.4, ls='--',
                          label=r'reference ($\sqrt{\cdot}$-law, $Y{=}0.5$ range-conv / $2/3$ level)'))
    _legend_below(fig, handles, [h.get_label() for h in handles])
    fig.suptitle(f'{args.stock}: metaorder impact — build-up and relaxation, '
                 'raw $k$-clock (left) and normalised master curves (right)',
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0.085, 1, 0.97])
    out = os.path.join(res, 'mid_impact', f'fig5_combined_{args.stock}.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    with open(out.replace('.png', '_sources.json'), 'w') as f:
        json.dump(src, f, indent=1)
    print(f'FIG5_DONE -> {out}', flush=True)
    if args.copy_to:
        import shutil
        shutil.copy(out, os.path.join(args.copy_to, os.path.basename(out)))
        print(f'copied -> {args.copy_to}', flush=True)


if __name__ == '__main__':
    main()
