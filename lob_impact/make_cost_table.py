#!/usr/bin/env python3
"""
Compute the REAL compute cost of the impact campaign from SLURM accounting.

The hand-maintained EXPERIMENT_LEDGER.md records jobs at SUBMIT time, so its
Elapsed/State columns are frozen at "PENDING / 00:00:00" — useless for a cost
appendix. This script instead reads `sacct`, which carries what actually
happened: queue wait, wall time, allocated CPUs/GPUs, and peak RSS.

Outputs (to 5_analysis/results/cost/):
  cost_by_step.md    — Markdown, for EXPERIMENT_LEDGER.md / the repo
  cost_by_step.tex   — LaTeX booktabs table, for the paper appendix
  cost_jobs.csv      — one row per job, for any further slicing

Usage:
  python3 make_cost_table.py --since 2026-07-01 [--until now] [--user $USER]

Cost model notes (stated in the table caption so the paper is honest):
  * "GPU-hours" = sum over jobs of (gres/gpu allocated) x elapsed hours.
  * "CPU-hours" = sum of AllocCPUS x elapsed hours. On this cluster a CPU-only
    job is rounded up to a quarter node (72 cores), so CPU-hours reflect what
    was *allocated* (and therefore denied to others), not what was busy.
  * "Peak RSS" is the max MaxRSS over the step's jobs — the sizing constraint.
  * Queue wait = Start - Submit, median over the step's jobs.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

# job-name prefix -> pipeline step (first match wins; order matters)
STEP_RULES = [
    # 1. generation: scenario rollouts (incl. per-model/per-stock fleets and controls)
    (r'^(s0|hi_|as5|amam|m34k|s54k|lobs5_legacy|bench_|gen_|lobimp_a3)', '1. Generation (scenario rollouts)'),
    (r'.*_inv_|.*_noi',                           '1. Generation (scenario rollouts)'),
    # per-stock fleets are named <stock-letter><model>_<shape><side><slice>, e.g. nmam4_bb7,
    # ggdn_rs3, acst_bs1 — the _bb/_bs/_rb/_rs suffix (beta|relaxation x buy|sell) is the marker
    (r'.*_(bb|bs|rb|rs)\d*$',                     '1. Generation (scenario rollouts)'),
    (r'^[a-z]{1,2}(mam|s5|gdn|cst|haw|nmz|ow|qr|hist|heur|prop|knn)', '1. Generation (scenario rollouts)'),
    # 2. distributional scoring
    (r'^(lbb|lobbench)',                          '2. LOB-Bench distributional scoring'),
    # 3. impact estimators: beta variants, 3x3, sigma grids, per-stock parameter fits
    (r'^(beta|lobimp_a5_beta|lobimp_a5_deca|b33|b3l|b3v|bsg|bbin|bvk|bias|qrest|cstest|hawkest)',
                                                  '3. Impact estimators (beta, 3x3, params)'),
    # 4. master curves / relaxation / propagator
    (r'^(lobimp_master|master|mtd_|mresp|m3?4?k?_?relaxatio)', '4. Master curves / relaxation / response'),
    # 5. figures, diagnostics, causal triangle, HTML explorers, docs
    (r'^(fig|dkf|tri_|part_|emprc|Y_|y3x3|figsweep|midtraj|htmlx|causal|lobimp_triangl|lobimp_tricmp|lobimp_fig|lobimp_a4|lobimp_a5_html|lobimp_a5_pub)',
                                                  '5. Figures & diagnostics'),
    # 6. v4 external baselines
    (r'^(smk_|mgpt_probe|lob_export)',            '6. v4 external baselines (smoke/export)'),
]


def classify(name):
    for pat, step in STEP_RULES:
        if re.match(pat, name):
            return step
    return '7. Other'


def parse_tres(tres, key):
    m = re.search(rf'{re.escape(key)}=(\d+)', tres or '')
    return int(m.group(1)) if m else 0


def parse_rss(s):
    """sacct MaxRSS like '58297280K', '108M', '12G' -> bytes."""
    if not s:
        return 0
    m = re.match(r'^([\d.]+)([KMGT]?)$', s.strip())
    if not m:
        return 0
    v, u = float(m.group(1)), m.group(2)
    return int(v * {'': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}[u])


def ts(s):
    try:
        return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S')
    except Exception:
        return None


def human_h(hours):
    return f'{hours:,.1f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2026-07-01')
    ap.add_argument('--until', default='now')
    ap.add_argument('--user', default=os.environ.get('USER'))
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '5_analysis', 'results', 'cost'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fmt = 'JobID,JobName,State,Submit,Start,End,ElapsedRaw,Timelimit,ReqMem,MaxRSS,AllocTRES,NNodes'
    cmd = ['sacct', '-u', args.user, '-S', args.since, '-E', args.until,
           '-o', fmt, '-P', '-n', '--units=K']
    out = subprocess.run(cmd, capture_output=True, text=True).stdout

    # main rows carry Submit/Start/AllocTRES; the .batch child carries MaxRSS
    main, rss = {}, {}
    for line in out.strip().split('\n'):
        if not line:
            continue
        f = line.split('|')
        if len(f) < 12:
            continue
        jid = f[0]
        if jid.endswith('.batch') or jid.endswith('.extern') or '.' in jid:
            base = jid.split('.')[0]
            rss[base] = max(rss.get(base, 0), parse_rss(f[9]))
            continue
        main[jid] = f

    jobs = []
    for jid, f in main.items():
        (_, name, state, submit, start, end, eraw, tlim, reqmem, _mrss, tres, nn) = f[:12]
        if state.startswith('CANCELLED') and (not start or start == 'Unknown'):
            continue  # never ran: no cost
        elapsed_h = int(eraw or 0) / 3600.0
        cpus = parse_tres(tres, 'cpu')
        gpus = parse_tres(tres, 'gres/gpu')
        mem_k = parse_tres(tres, 'mem')  # not reliable across units; ReqMem kept as text
        sub_t, st_t = ts(submit), ts(start)
        wait_h = (st_t - sub_t).total_seconds() / 3600.0 if (sub_t and st_t) else None
        jobs.append(dict(
            jobid=jid, name=name, step=classify(name), state=state.split()[0],
            elapsed_h=elapsed_h, cpus=cpus, gpus=gpus,
            cpu_h=cpus * elapsed_h, gpu_h=gpus * elapsed_h,
            rss_gib=rss.get(jid, 0) / 1024**3, reqmem=reqmem, tlim=tlim,
            wait_h=wait_h, nodes=int(nn or 1)))

    if not jobs:
        print('no jobs found', file=sys.stderr)
        sys.exit(1)

    with open(os.path.join(args.out, 'cost_jobs.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(jobs[0].keys()))
        w.writeheader()
        w.writerows(jobs)

    agg = defaultdict(lambda: dict(n=0, ok=0, fail=0, elapsed=0.0, cpu_h=0.0,
                                   gpu_h=0.0, rss=0.0, waits=[]))
    for j in jobs:
        a = agg[j['step']]
        a['n'] += 1
        a['ok'] += j['state'] == 'COMPLETED'
        a['fail'] += j['state'] in ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY')
        a['elapsed'] += j['elapsed_h']
        a['cpu_h'] += j['cpu_h']
        a['gpu_h'] += j['gpu_h']
        a['rss'] = max(a['rss'], j['rss_gib'])
        if j['wait_h'] is not None:
            a['waits'].append(j['wait_h'])

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else 0.0

    steps = sorted(agg)
    tot = dict(n=0, ok=0, fail=0, elapsed=0.0, cpu_h=0.0, gpu_h=0.0)
    lines_md = ['| Pipeline step | Jobs | OK / failed | Wall-clock (h) | CPU-hours | GPU-hours | Peak RSS (GiB) | Median queue wait (h) |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
    rows_tex = []
    for s in steps:
        a = agg[s]
        for k in ('n', 'ok', 'fail', 'elapsed', 'cpu_h', 'gpu_h'):
            tot[k] += a[k]
        lines_md.append(f"| {s} | {a['n']} | {a['ok']} / {a['fail']} | {human_h(a['elapsed'])} | "
                        f"{human_h(a['cpu_h'])} | {human_h(a['gpu_h'])} | {a['rss']:.1f} | {med(a['waits']):.2f} |")
        rows_tex.append(f"{s.split('. ',1)[1]} & {a['n']} & {a['ok']}/{a['fail']} & {a['elapsed']:,.1f} & "
                        f"{a['cpu_h']:,.0f} & {a['gpu_h']:,.1f} & {a['rss']:.0f} & {med(a['waits']):.2f} \\\\")
    lines_md.append(f"| **Total** | **{tot['n']}** | **{tot['ok']} / {tot['fail']}** | **{human_h(tot['elapsed'])}** | "
                    f"**{human_h(tot['cpu_h'])}** | **{human_h(tot['gpu_h'])}** | | |")

    hdr = (f"# Compute cost of the impact campaign\n\n"
           f"Source: `sacct` for user `{args.user}`, {args.since} .. {args.until} "
           f"(generated by `make_cost_table.py`; jobs that never started are excluded).\n\n")
    open(os.path.join(args.out, 'cost_by_step.md'), 'w').write(hdr + '\n'.join(lines_md) + '\n')

    tex = ['\\begin{table*}[t]', '  \\centering', '  \\small',
           '  \\begin{tabular}{lrrrrrrr}', '    \\toprule',
           '    Pipeline step & Jobs & OK/fail & Wall (h) & CPU-h & GPU-h & Peak RSS (GiB) & Med.\\ queue (h) \\\\',
           '    \\midrule']
    tex += ['    ' + r for r in rows_tex]
    tex += ['    \\midrule',
            f"    \\textbf{{Total}} & \\textbf{{{tot['n']}}} & \\textbf{{{tot['ok']}/{tot['fail']}}} & "
            f"\\textbf{{{tot['elapsed']:,.1f}}} & \\textbf{{{tot['cpu_h']:,.0f}}} & \\textbf{{{tot['gpu_h']:,.1f}}} & & \\\\",
            '    \\bottomrule', '  \\end{tabular}',
            '  \\caption{Measured compute cost of the full evaluation campaign, from SLURM accounting '
            '(\\texttt{sacct}). CPU-hours count \\emph{allocated} cores $\\times$ elapsed time: on this '
            'cluster a CPU-only job is rounded up to a quarter node (72 cores), so CPU-hours reflect '
            'capacity denied to other users rather than cores kept busy. GPU-hours count allocated '
            'GPUs $\\times$ elapsed time. Peak RSS is the maximum observed resident set size across the '
            'step\'s jobs and is the memory-sizing constraint. Jobs that never started are excluded.}',
            '  \\label{tab:compute-cost}', '\\end{table*}']
    open(os.path.join(args.out, 'cost_by_step.tex'), 'w').write('\n'.join(tex) + '\n')

    print('\n'.join(lines_md))
    print(f"\nwrote -> {args.out}/{{cost_by_step.md,cost_by_step.tex,cost_jobs.csv}}")


if __name__ == '__main__':
    main()
