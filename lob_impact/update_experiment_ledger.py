#!/usr/bin/env python3
"""
Regenerate lob_impact/EXPERIMENT_LEDGER.md from SLURM accounting (sacct).

Purpose: the paper quotes compute statistics (queue wait, wall time, CPUs/GPUs,
GPU-hours per model) to give readers intuition about the cost of each generator.
Run this after every fleet completes:

    python3 lob_impact/update_experiment_ledger.py [--since 2026-06-30]

Login-node safe: one sacct call + trivial parsing, no filesystem walking.
n_samples is not in sacct — it comes from FLEET_NOTES below (extend by hand as
fleets are launched; unknown fleets just show the raw job rows).
"""
import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Hand-maintained notes keyed by job-name regex: (samples/output note, shown in the table).
FLEET_NOTES = [
    (r'^(bet|rel)_(historic|heuristic|cst|hawkes|prop|nmzi)', 'EA grid_v2 baseline: 64 samples/side (8/day x 8 days, per-day calib)'),
    (r'^lobimp_(bet|rel)', 'EA grid_v2: per-day calibrated fleet'),
    (r'^r2k_m3', 'Mamba3-500 EA grid_v2: 2048 samples target, bsz 52'),
    (r'^(m34k|s54k)_', '4k-context neural fleet: sample slices across GPU jobs'),
    (r'^noins_', 'no-insertion drift control: 64 samples, 13k clean msgs'),
    (r'^inv_', 'invisible-metaorder control: 8 samples/day (PER_DAY=1 N_PER_DAY=8)'),
    (r'^lobimp_tri', 'triangle report build: reads 64-sample regimes, renders figs+docx'),
    (r'^(hi|he|pr)_NVDA', 'NVDA grid_v2 replay: 52 samples/day x 20 days (1040/side)'),
    (r'^n(mam|gdn|s5)_[br][bs]\d', 'NVDA 500-ctx neural slice: 4 days/slice, 52 samples/day'),
    (r'^n(mam4|s5_4)', 'NVDA 4k-ctx neural slice: bsz 8, 104 samples/day, 10 slices'),
    (r'^(rn|f)(mam4|cst|nmz)', 'NVDA finisher: day-19 XLA-pressure / node-tmpfs resubmits'),
    (r'^(cst|nmz|haw)_N', 'NVDA grid_v2 param model: 52 samples/day x 20 days, 256G'),
    (r'^(N|G)(mast|midt|3v|agg|part)|^GNpart', 'figure job over grid_v2 (master/midtraj/3views/aggregation/participation)'),
    (r'^lobimp_beta_NVDA', 'NVDA beta analysis wave over grid_v2'),
    (r'^(hi|he|pr)_GOOG', 'GOOG grid_v2 replay: 52 samples/day x 20 days (1040/side)'),
    (r'^g(mam|gdn|s5)_[br][bs]\d', 'GOOG 500-ctx neural slice: 2 days/slice, 52 samples/day, bsz 52'),
    (r'^g(mam4|s5_4)', 'GOOG 4k-ctx neural slice: bsz 8, 104 samples/day, 16 slices'),
    (r'^(rf|f)(mam4|gmam4|s5_4|gs5_4)', 'GOOG 4k day-1 finisher: SAMPLE_SLICE=<day>/20 after 12h timeout'),
    (r'^(cst|nmz|haw)_G', 'GOOG grid_v2 param model: 52 samples/day x 20 days, 256G (tmpfs+XLA cgroup)'),
    (r'^(cst|hwk)_est_GOOG', 'GOOG param estimation from historic data_cond'),
    (r'^tw_day|^lobs5_legacy', 'twilight-sound-77 GOOG-2023 legacy grid: 104 samples/day (probe 8)'),
    (r'^goog23', 'GOOG 2023_Jan calibration from flair06 old-proc npy'),
    (r'^lobimp_beta', 'beta analysis pass over grid_v2'),
    (r'^lobimp_run|^run_', 'run_experiments.sh launch (smoke=64 samples unless noted)'),
    # 2026-07-13 post-crash recovery fleet (triangle/event-response for GOOG+NVDA papers)
    (r'^Rinv_', 'NVDA invisible finisher after 07-10 OOM: 8/day, day-slice k/2, 224G'),
    (r'^empresp_', 'real-data R(m) anchor from data_cond (400 files)'),
    (r'^basresp_', 'impact-blind baseline R(m) reference (1024 samples)'),
    (r'^tri[GN]_', 'control-triangle report: visible/invisible/noins x 160 samples'),
    (r'^figtri_', 'paper figures: triangle_bars + event_response per stock'),
    (r'^evresp_', 'event_response figure re-render (annotation fix)'),
    (r'^dur_NVDA', 'NVDA duration-independence slopes over grid_v2'),
    (r'^partover', 'participation cumulative overshoot printer (GOOG+NVDA Mamba3)'),
    (r'^html_[GN]_', 'interactive HTML explorer stage (12h resubmit of the 6h-TIMEOUT run)'),
    (r'^bkvol2?_', 'beta-k vol explorer + audit: 6 sigmas x 3 fit modes, bootstrap CI, cross-metaorder, Q-split'),
    (r'^esens_', 'estimator sensitivity: log-log/L2/L1, Almgren J=I/2+temp, duration covariate'),
    (r'^tr_audit', 'transient exponent + relaxation tail from master-curve caches (seconds)'),
    (r'^bexh_', 'bias exhibit figure: cloud + signed bins + per-point/binned/L2 fits, 2 sigma rows'),
]

TRES_RE = {
    'cpu': re.compile(r'(?:^|,)cpu=(\d+)'),
    'gpu': re.compile(r'gres/gpu=(\d+)'),
    'mem': re.compile(r'mem=([0-9.]+[MGT])'),
}


def note_for(name: str) -> str:
    for pat, note in FLEET_NOTES:
        if re.search(pat, name):
            return note
    return ''


def hhmmss_to_h(s: str) -> float:
    if not s or s == 'INVALID':
        return 0.0
    days, rest = (s.split('-', 1) + [''])[:2] if '-' in s else ('0', s)
    parts = [int(x) for x in rest.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return int(days) * 24 + h + m / 60 + sec / 3600


def wait_str(submit: str, start: str) -> str:
    fmt = '%Y-%m-%dT%H:%M:%S'
    try:
        d = (datetime.strptime(start, fmt) - datetime.strptime(submit, fmt)).total_seconds()
    except ValueError:
        return '-'
    if d < 0:
        return '-'
    if d < 120:
        return f'{d:.0f}s'
    if d < 7200:
        return f'{d / 60:.0f}m'
    return f'{d / 3600:.1f}h'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2026-06-30')
    ap.add_argument('--out', default=str(HERE / 'EXPERIMENT_LEDGER.md'))
    args = ap.parse_args()

    cmd = ['sacct', '-S', args.since, '-X', '-P', '--noheader',
           '--format=JobID,JobName%40,Submit,Start,Elapsed,AllocTRES%80,State%25,Partition']
    rows = []
    sacct_out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True).stdout
    for line in sacct_out.splitlines():
        f = line.split('|')
        if len(f) < 8:
            continue
        jid, name, submit, start, elapsed, tres, state, part = f[:8]
        if state.startswith('PENDING'):
            start, elapsed = 'None', '00:00:00'
        cpu = TRES_RE['cpu'].search(tres)
        gpu = TRES_RE['gpu'].search(tres)
        mem = TRES_RE['mem'].search(tres)
        rows.append(dict(
            jid=jid, name=name, submit=submit, start=start, elapsed=elapsed,
            cpu=int(cpu.group(1)) if cpu else 0,
            gpu=int(gpu.group(1)) if gpu else 0,
            mem=mem.group(1) if mem else '-',
            state=state.split()[0], part=part,
        ))

    gpu_h = sum(r['gpu'] * hhmmss_to_h(r['elapsed']) for r in rows)
    cpu_h = sum(r['cpu'] * hhmmss_to_h(r['elapsed']) for r in rows)
    gpu_h_ok = sum(r['gpu'] * hhmmss_to_h(r['elapsed']) for r in rows if r['state'] == 'COMPLETED')
    n_ok = sum(r['state'] == 'COMPLETED' for r in rows)

    out = [
        '# Experiment ledger (SLURM accounting, Isambard-AI)',
        '',
        f'Auto-generated by `update_experiment_ledger.py` on {datetime.now():%Y-%m-%d %H:%M} '
        f'(jobs since {args.since}). Regenerate after every fleet. All jobs are one node; '
        'AllocCPUS is the node allocation (72 = full node), GPUs are H100.',
        '',
        f'- **{len(rows)} jobs** ({n_ok} COMPLETED), wall-clock sums over all states.',
        f'- **GPU-hours: {gpu_h:.0f}** total / {gpu_h_ok:.0f} in completed jobs; **CPU-core-hours: {cpu_h:.0f}**.',
        '- Queue wait = Start − Submit. n_samples/fleet notes are hand-maintained in the script.',
        '',
        '| JobID | Name | Submitted | Wait | Wall | CPU | GPU | Mem | State | Notes |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for r in rows:
        out.append(
            f"| {r['jid']} | {r['name']} | {r['submit'][:16]} | {wait_str(r['submit'], r['start'])} "
            f"| {r['elapsed']} | {r['cpu']} | {r['gpu'] or ''} | {r['mem']} | {r['state']} | {note_for(r['name'])} |")
    Path(args.out).write_text('\n'.join(out) + '\n')
    print(f'{len(rows)} jobs -> {args.out}  (GPU-h {gpu_h:.0f}, CPU-core-h {cpu_h:.0f})')


if __name__ == '__main__':
    main()
