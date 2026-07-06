#!/usr/bin/env python3
"""
Explainer doc: HOW paper Figure 4 (mid-price trajectory over generation) averages
samples of DIFFERENT LENGTHS (per-day mb -> per-day rollout length).

Two figures (one synthetic schematic, one from real EA data) + a Word doc.

  python 4_diagnostics/make_fig4_averaging_docx.py --out_dir results/fig4_averaging_<ts>
"""
import os, sys, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_triangle_docx import Doc  # noqa: E402

IMPACT = os.path.dirname(HERE)
PER_DAY_CSV = os.path.join(IMPACT, '1_data_prep/results/per_day_params/per_day_params_EA.csv')
FIG4_NPZ = os.path.join(IMPACT, '5_analysis/beta/results/mid_impact/mid_trajectory_EA_beta.npz')

C_A, C_B, C_C = '#2a78d6', '#1baf7a', '#eb6834'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRIDC, AXISC = '#e1e0d9', '#c3c2b7'
CRIT = '#d03b3b'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'text.color': INK, 'axes.edgecolor': AXISC, 'axes.labelcolor': INK2,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRIDC, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'legend.frameon': False,
})


def fig_schematic(path):
    """Three toy days with different mb -> different lengths; alive-count staircase."""
    rng = np.random.default_rng(7)
    days = [('day A  (m_b=110)', 110, C_A), ('day B  (m_b=125)', 125, C_B),
            ('day C  (m_b=145)', 145, C_C)]
    n_ins = 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.8, 6.4), dpi=200, sharex=True,
                                   height_ratios=[2.1, 1.0])
    ax1.grid(axis='x', visible=False)
    Ls = []
    for lab, mb, col in days:
        L = n_ins * mb
        Ls.append(L)
        t = np.arange(L)
        y = 40 * np.sqrt(t / L) + np.cumsum(rng.normal(0, 0.55, L)) * 0.35
        ax1.plot(t, y, color=col, lw=1.6, label=lab)
        ax1.axvline(L, color=col, lw=1.0, ls=(0, (2, 2)), alpha=0.8)
        ax1.annotate(f'{lab.split()[0]} {lab.split()[1]} ends\n(100·m_b = {L:,})',
                     xy=(L, 4), xytext=(L - 2600, -14), color=col, fontsize=8,
                     fontweight='bold')
        for k in range(10, n_ins + 1, 10):     # sparse insertion ticks at k·mb (day-specific!)
            ax1.axvline(k * mb, color=col, lw=0.5, alpha=0.18)
    ax1.set_ylabel('signed mid move (schematic)')
    ax1.set_title('Same experiment, three days: insertion k lands at step k·m_b — different step per day',
                  fontsize=11, fontweight='bold', loc='left', color=INK)
    ax1.legend(loc='upper left', fontsize=8.5)
    ax1.set_ylim(-20, 60)

    # alive-count staircase (say 52 samples per day, as in the EA grid)
    per_day_n = 52
    grid = np.arange(0, max(Ls) + 200, 10)
    cnt = np.zeros_like(grid)
    for L in Ls:
        cnt = cnt + (grid <= L) * per_day_n
    ax2.grid(axis='x', visible=False)
    ax2.step(grid, cnt, where='post', color=INK, lw=1.8)
    cut = max(0.5 * 3 * per_day_n, 30)
    ax2.axhline(cut, color=CRIT, lw=1.2, ls=(0, (4, 2)))
    ax2.annotate('cut: keep steps with cnt ≥ max(50% of samples, 30)',
                 xy=(200, cut + 6), color=CRIT, fontsize=8.5, fontweight='bold')
    Lkeep = grid[cnt >= cut].max()
    ax2.axvline(Lkeep, color=CRIT, lw=1.0, ls=(0, (2, 2)))
    ax2.fill_betweenx([0, 3 * per_day_n], Lkeep, grid[-1], color=CRIT, alpha=0.06, lw=0)
    ax2.annotate('dropped: only long-m_b days\nsurvive here → composition bias',
                 xy=(Lkeep + 300, 3 * per_day_n * 0.55), color=CRIT, fontsize=8.5)
    ax2.set_ylabel('samples alive  cnt(t)')
    ax2.set_xlabel('message step  t  (event time)')
    ax2.set_ylim(0, 3 * per_day_n * 1.12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def fig_real(path):
    """Real EA inputs: per-day m_b histogram + the actual alive-count staircase of Fig 4."""
    rows = list(csv.DictReader(open(PER_DAY_CSV)))
    mb = np.array([float(r['mb']) for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.0), dpi=200,
                                   width_ratios=[1.0, 1.6])
    ax1.grid(axis='x', visible=False)
    ax1.hist(mb, bins=12, color=C_A, edgecolor='white')
    ax1.set_xlabel('per-day m_b (messages between children)')
    ax1.set_ylabel('trading days')
    ax1.set_title(f'EA, Jan 2026: m_b varies {mb.min():.0f}–{mb.max():.0f} across '
                  f'{len(mb)} days\n→ build-up rollout length 100·m_b varies '
                  f'{100 * mb.min() / 1000:.1f}k–{100 * mb.max() / 1000:.1f}k msgs',
                  fontsize=10, fontweight='bold', loc='left', color=INK)

    z = np.load(FIG4_NPZ)
    cnt = z['Mamba3_cnt']
    n = int(cnt.max())
    t = np.arange(len(cnt))
    ax2.grid(axis='x', visible=False)
    ax2.step(t, cnt, where='post', color=INK, lw=1.5)
    cut = max(0.5 * n, 30)
    keep = cnt >= cut
    Lkeep = int(np.max(np.where(keep)[0])) + 1 if keep.any() else len(cnt)
    ax2.axhline(cut, color=CRIT, lw=1.2, ls=(0, (4, 2)))
    ax2.axvline(Lkeep, color=CRIT, lw=1.0, ls=(0, (2, 2)))
    ax2.fill_betweenx([0, n], Lkeep, t[-1], color=CRIT, alpha=0.06, lw=0)
    ax2.annotate(f'cnt ≥ max(50%·n, 30) → curve drawn to step {Lkeep:,}',
                 xy=(0.02, 0.10), xycoords='axes fraction', color=CRIT,
                 fontsize=8.5, fontweight='bold')
    ax2.annotate('each drop = the samples of one\n(shorter-m_b) day ending',
                 xy=(0.55, 0.6), xycoords='axes fraction', color=INK2, fontsize=8.5)
    ax2.set_xlabel('message step  t')
    ax2.set_ylabel('samples alive  cnt(t)')
    ax2.set_title(f'Actual Fig 4 staircase (Mamba3, n={n} samples pooled buy+sell)',
                  fontsize=10, fontweight='bold', loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def build_docx(out_dir, out_path):
    rows = list(csv.DictReader(open(PER_DAY_CSV)))
    mb = np.array([float(r['mb']) for r in rows])
    d = Doc()
    d.p([('How Figure 4 Averages Samples of Different Lengths', True, False, None, 40)],
        style='Title', space_after=40)
    d.p([('Event-time alignment, trimmed mean, and the alive-count guardrail — '
          'EA · grid_v2 · 2026-07-06', False, True, '52514E', 20)], space_after=240)

    d.h('1. Why the samples have different lengths', 1)
    d.p(f'The injection schedule is calibrated per trading day: each day d gets its own '
        f'm_b(d) — the number of model-generated messages between consecutive child orders — '
        f'chosen so the metaorder holds ~10% participation on that day’s trade rate. On EA '
        f'(Jan 2026) m_b ranges {mb.min():.0f}–{mb.max():.0f} across {len(mb)} days, so the '
        f'build-up rollout (100 children) is 100·m_b ≈ {100 * mb.min() / 1000:.1f}k–'
        f'{100 * mb.max() / 1000:.1f}k messages long, and the relaxation rollout (10 children '
        f'+ 100 cooling windows) is 110·m_b. Two samples from different days therefore never '
        f'have the same length, and insertion k lands at a different message step (k·m_b) on '
        f'each day. A naive matrix average is undefined; truncating everything to the shortest '
        f'day (an early version of the script) throws away up to a third of every long day.')
    d.image(os.path.join(out_dir, 'figA_schematic.png'),
            'Figure A. Schematic: three days with m_b = 110/125/145. Top: each sample ends at '
            'its own 100·m_b; insertion k sits at k·m_b (faint verticals). Bottom: the count of '
            'samples still alive at step t and the 50%-cut that ends the plotted curve.')

    d.h('2. What Figure 4 actually computes', 1)
    d.bullet([('Alignment: absolute message step (event time). ', True, False, None, None),
              ('All samples start at t=0 = the first generated message; the impact reference '
               'is the mid just before the first insertion. Sell runs are multiplied by −1 and '
               'pooled with buy runs.', False, False, None, None)])
    d.bullet([('Ragged stacking, not truncation. ', True, False, None, None),
              ('Samples are placed in a matrix padded with NaN to the longest day; the statistic '
               'at step t uses exactly the samples that are still running at t (cnt(t) of them).',
               False, False, None, None)])
    d.bullet([('Estimator: 10% two-sided trimmed mean per step, ', True, False, None, None),
              ('with a ±1.96 s.e. band of the kept samples. The trim protects the curve from '
               'rare book blow-ups (single samples reaching hundreds of bps) without the '
               'staircasing a median produces on tick-discrete mids.', False, False, None, None)])
    d.bullet([('Guardrail: the curve is drawn only while cnt(t) ≥ max(50% of samples, 30). ',
               True, False, None, None),
              ('Past that point the average would be dominated by the few longest-m_b days — a '
               'composition change, not model behaviour — so the tail is dropped.',
               False, False, None, None)])
    d.image(os.path.join(out_dir, 'figB_real.png'),
            'Figure B. Real EA inputs. Left: the per-day m_b distribution that causes the length '
            'spread. Right: the actual alive-count staircase behind Figure 4 (Mamba3); each step '
            'down is one day’s samples ending; the red cut ends the plotted curve.')

    d.h('3. What averaging at a fixed step means (and the visible artefacts)', 1)
    d.p('At a fixed step t, day d has executed k ≈ t/m_b(d) children — so the average at t '
        'mixes days at slightly different metaorder progress (±15% or so across the m_b '
        'spread). This is the price of event-time alignment, and it is also why two visible '
        'artefacts appear near the right edge of Figure 4: (i) small staircase jumps in the '
        'mean where a day’s samples end inside the kept window, and (ii) widening error '
        'bands as cnt(t) falls. The same composition effect explains the late-window upticks '
        'in the per-event response R(m) beyond m ≈ 105 in the triangle reports. None of these '
        'are model behaviour; they are sample-set composition.')

    d.h('4. Why not rescale time instead? We do — that is Figure 5', 1)
    d.table([
        ['Time axis', 'Definition', 'Question it answers', 'Where used'],
        ['Event time (this doc)', 't = message step; average over samples alive at t',
         'How much has the mid moved after t messages of trading?',
         'Paper Fig 4; √-law overlay at k·mean(m_b)'],
        ['Volume time', 'v = t / T_exec(sample); every sample interpolated to a common v-grid; '
         'v=1 = last child',
         'What is the SHAPE of the build-up, with execution progress aligned exactly?',
         'Paper Fig 5 master curves (then normalised by ⟨I(1)⟩)'],
        ['Rollout fraction', 'x = t / L(sample) interpolated to [0,1]',
         'Regime comparison over the whole rollout including cooling',
         'Control-triangle trajectory figures'],
    ], widths=[1400, 2800, 2600, 2560])
    d.p('Event time preserves the real message clock (so insertion ticks and the √-law '
        'reference can be drawn at actual steps) but mixes progress slightly; volume time '
        'aligns progress exactly but discards the clock and any level information once '
        'normalised. They are complementary — Figure 4 carries the LEVEL result (tens of bps '
        'versus a ~1 bps √-law expectation), Figure 5 carries the SHAPE result (convex, '
        'no relaxation). Neither is a different "correctness" of averaging.')

    d.h('5. The √-law reference on Figure 4', 1)
    d.p('The dashed reference is built from the same per-day calibration: for day d and child '
        'k, I_d(k) = σ_d · √(k·child_d / V_d); the curve shown is the across-day mean of '
        'I_d(k), placed at step k·mean(m_b). It inherits the per-day child sizes and daily '
        'volumes, so it is the apples-to-apples expectation for exactly this injection '
        'schedule — the 0.9 bps it ends at is what a √-law market would show for this '
        'metaorder, which is the honest yardstick for the neural models’ tens of bps.')

    d.h('6. Reproduction', 1)
    d.p('Averaging code: lob_impact/5_analysis/beta/mid_trajectory.py (estimator=tmean, '
        'trim=0.10, min_frac=0.5). This document: '
        'lob_impact/4_diagnostics/make_fig4_averaging_docx.py; Figure B reads the cached '
        'mid_trajectory_EA_beta.npz and 1_data_prep per_day_params_EA.csv.')
    d.save(out_path)
    print(f'DOCX_DONE -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fig_schematic(os.path.join(args.out_dir, 'figA_schematic.png'))
    fig_real(os.path.join(args.out_dir, 'figB_real.png'))
    build_docx(args.out_dir, os.path.join(args.out_dir, 'Fig4_Averaging_Explained.docx'))
    print(f'ALL_DONE -> {args.out_dir}')


if __name__ == '__main__':
    main()
