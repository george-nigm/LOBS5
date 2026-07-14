#!/usr/bin/env python3
"""
Five daily volatility estimators from OHLC, for the beta-regression normalization.

Reads the Action-2 daily file (ticker, day, open_price, highest_price, lowest_price,
close_price, execution_sum) and returns, per (ticker, day), a daily sigma under each of:

  parkinson        (H,L)           per-day, range-based  σ²=(h-l)²/(4ln2)
  range_raw        (H,L)           per-day, raw log range σ=h-l  — the metaorder-literature
                                   convention (Sato-Kanazawa, Maitrier TSE, Zarinelli all use the
                                   plain daily range, NOT the 1/sqrt(4ln2)≈0.60-scaled Parkinson);
                                   use this sigma when comparing amplitudes against Y≈0.5.
                                   Not in METHODS (kept out of the 5-method grids); extra dict key.
  garman_klass     (O,H,L,C)       per-day               σ²=0.5(h-l)² - (2ln2-1)(c-o)²
  rogers_satchell  (O,H,L,C)       per-day, drift-robust σ²=(h-c)(h-o)+(l-c)(l-o)
  close_to_close   (C series)      per-ticker window     σ=std(Δ ln C) over the month
  yang_zhang       (O,H,L,C series)per-ticker window     overnight + k·open + (1-k)·RS

h,l,o,c are LOG prices. close_to_close & yang_zhang are window estimators -> one value per
ticker, broadcast to all that ticker's days (they shift the regression intercept, not slope,
within a ticker — that is exactly what the 5-method comparison is meant to expose).
"""
import csv
from collections import defaultdict
import numpy as np

METHODS = ['parkinson', 'garman_klass', 'rogers_satchell', 'close_to_close', 'yang_zhang']
_LN2 = np.log(2.0)


def _rows(daily_csv):
    by_ticker = defaultdict(list)
    with open(daily_csv) as fh:
        for r in csv.DictReader(fh):
            try:
                O = float(r.get('open_price', 0)); H = float(r['highest_price'])
                L = float(r['lowest_price']); C = float(r.get('close_price', 0))
                V = float(r['execution_sum'])
            except (KeyError, ValueError):
                continue
            if min(O, H, L, C) <= 0 or H < L:
                # tolerate missing OHLC (old files): keep H/L only
                if H <= 0 or L <= 0 or H < L:
                    continue
            by_ticker[r['ticker']].append(dict(day=r['day'], O=O, H=H, L=L, C=C, V=V))
    for t in by_ticker:
        by_ticker[t].sort(key=lambda x: x['day'])
    return by_ticker


def daily_sigmas(daily_csv):
    """(ticker, day) -> dict(method -> sigma, plus 'V')."""
    by_ticker = _rows(daily_csv)
    out = {}
    for ticker, days in by_ticker.items():
        O = np.array([d['O'] for d in days]); H = np.array([d['H'] for d in days])
        L = np.array([d['L'] for d in days]); C = np.array([d['C'] for d in days])
        o, h, l, c = (np.log(np.where(x > 0, x, np.nan)) for x in (O, H, L, C))

        park = np.sqrt(np.clip((h - l) ** 2 / (4 * _LN2), 0, None))
        gk = np.sqrt(np.clip(0.5 * (h - l) ** 2 - (2 * _LN2 - 1) * (c - o) ** 2, 0, None))
        rs = np.sqrt(np.clip((h - c) * (h - o) + (l - c) * (l - o), 0, None))

        # close-to-close: std of daily log returns of close (one per ticker)
        if np.isfinite(c).sum() >= 3:
            dc = np.diff(c[np.isfinite(c)])
            c2c_val = float(np.std(dc, ddof=1)) if dc.size > 1 else np.nan
        else:
            c2c_val = np.nan

        # yang-zhang (window): σ_o² (overnight) + k σ_open² + (1-k) mean(RS²)
        yz_val = np.nan
        fin = np.isfinite(o) & np.isfinite(c)
        if fin.sum() >= 4:
            oc_overnight = o[1:] - c[:-1]          # ln(O_t / C_{t-1})
            co_open = c - o                         # ln(C_t / O_t)
            on = oc_overnight[np.isfinite(oc_overnight)]
            op = co_open[np.isfinite(co_open)]
            rs2 = rs[np.isfinite(rs)] ** 2
            if on.size > 1 and op.size > 1 and rs2.size > 0:
                n = min(on.size, op.size)
                sig_o2 = np.var(on, ddof=1)
                sig_c2 = np.var(op, ddof=1)
                sig_rs2 = np.mean(rs2)
                k = 0.34 / (1.34 + (n + 1) / (n - 1))
                yz_val = float(np.sqrt(max(sig_o2 + k * sig_c2 + (1 - k) * sig_rs2, 0)))

        for i, d in enumerate(days):
            out[(ticker, d['day'])] = dict(
                range_raw=float(h[i] - l[i]) if np.isfinite(h[i] - l[i]) else np.nan,
                parkinson=float(park[i]) if np.isfinite(park[i]) else np.nan,
                garman_klass=float(gk[i]) if np.isfinite(gk[i]) else np.nan,
                rogers_satchell=float(rs[i]) if np.isfinite(rs[i]) else np.nan,
                close_to_close=c2c_val,
                yang_zhang=yz_val,
                V=d['V'],
            )
    return out


if __name__ == '__main__':
    import sys
    d = daily_sigmas(sys.argv[1])
    for tk in ('EA', 'NVDA', 'AMD'):
        days = sorted(k for k in d if k[0] == tk)
        if not days:
            print(f'{tk}: no rows'); continue
        s = d[days[0]]
        print(f'{tk} {days[0][1]}: ' + '  '.join(f'{m}={s[m]:.4f}' for m in METHODS) + f'  V={s["V"]:.0f}')
