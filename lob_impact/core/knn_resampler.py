"""
K-NN message resampling core (Giegrich, Oomen, Reisinger, arXiv:2409.06514),
adapted from state-transition resampling to message-level resampling so it can
drive the JAX-LOB simulator.

Paper-faithful choices: state = volume profile around the mid (l levels/side),
plain Euclidean distance, K=20 neighborhood with k ~ U{1..K}, additive price
re-anchoring. Documented deviations (see MODEL_CATALOG.md / paper writeup):
  * we resample the historical NEXT MESSAGE(S) after the matched state instead
    of the next L2 state (we must emit messages, not states);
  * optional block resampling (`knn_block` > 1): adopt the neighbor's next m
    consecutive messages before re-querying (m=1 reproduces the paper's
    per-transition cadence at ~m x the query cost);
  * volume profile is re-gridded onto a fixed +-l_ticks tick grid around the
    tick-floored mid (our L10 book is price-sparse).

Pure numpy; pools are built per trading day from the same proc .npy files the
historic scenario reads (message: 14 cols; book: 43 cols = [dmid, ts, tns, 40 L10]).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as onp

# proc message columns (same as scenario constants)
EVENT_TYPE_i = 1
DIRECTION_i = 2
PRICE_ABS_i = 3
PRICE_REL_i = 4
SIZE_i = 5
DTs_i = 6
DTns_i = 7

DATE_RE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def vol_profile(l2_flat: onp.ndarray, tick_size: int, l_ticks: int) -> onp.ndarray:
    """
    Re-grid one flat L2 row [askP1,askV1,bidP1,bidV1,...] (n_levels*4) onto a
    fixed tick-offset grid around the tick-floored mid.

    Returns float32 vector of length 2*l_ticks:
      [bid vol at -l_ticks .. -1, ask vol at +1 .. +l_ticks] (missing levels = 0).
    """
    ask_p = l2_flat[0::4].astype(onp.int64)
    ask_v = l2_flat[1::4].astype(onp.int64)
    bid_p = l2_flat[2::4].astype(onp.int64)
    bid_v = l2_flat[3::4].astype(onp.int64)

    out = onp.zeros(2 * l_ticks, dtype=onp.float32)
    if ask_p[0] <= 0 or bid_p[0] <= 0:
        return out
    mid = ((ask_p[0] + bid_p[0]) // 2 // tick_size) * tick_size

    # bid side: offsets -1 .. -l_ticks -> slots l_ticks-1 .. 0
    off_b = (bid_p - mid) // tick_size          # negative
    ok = (bid_p > 0) & (off_b <= -1) & (off_b >= -l_ticks)
    slots = (off_b[ok] + l_ticks).astype(onp.int64)   # -l_ticks -> 0
    onp.add.at(out, slots, bid_v[ok].astype(onp.float32))

    # ask side: offsets +1 .. +l_ticks -> slots l_ticks .. 2*l_ticks-1
    off_a = (ask_p - mid) // tick_size          # positive
    ok = (ask_p > 0) & (off_a >= 1) & (off_a <= l_ticks)
    slots = (off_a[ok] - 1 + l_ticks).astype(onp.int64)
    onp.add.at(out, slots, ask_v[ok].astype(onp.float32))
    return out


def _vol_profiles_bulk(l2_rows: onp.ndarray, tick_size: int, l_ticks: int) -> onp.ndarray:
    """Vectorized vol_profile over (N, n_levels*4) -> (N, 2*l_ticks) float32."""
    n = l2_rows.shape[0]
    ask_p = l2_rows[:, 0::4].astype(onp.int64)
    ask_v = l2_rows[:, 1::4].astype(onp.float32)
    bid_p = l2_rows[:, 2::4].astype(onp.int64)
    bid_v = l2_rows[:, 3::4].astype(onp.float32)

    mid = ((ask_p[:, 0] + bid_p[:, 0]) // 2 // tick_size) * tick_size
    valid = (ask_p[:, 0] > 0) & (bid_p[:, 0] > 0)

    out = onp.zeros((n, 2 * l_ticks), dtype=onp.float32)
    rows_idx = onp.broadcast_to(onp.arange(n)[:, None], ask_p.shape)

    off_b = (bid_p - mid[:, None]) // tick_size
    ok = valid[:, None] & (bid_p > 0) & (off_b <= -1) & (off_b >= -l_ticks)
    onp.add.at(out, (rows_idx[ok], (off_b[ok] + l_ticks).astype(onp.int64)), bid_v[ok])

    off_a = (ask_p - mid[:, None]) // tick_size
    ok = valid[:, None] & (ask_p > 0) & (off_a >= 1) & (off_a <= l_ticks)
    onp.add.at(out, (rows_idx[ok], (off_a[ok] - 1 + l_ticks).astype(onp.int64)), ask_v[ok])
    return out


class KnnPool:
    """
    One trading day's resampling pool: state features at anchor indices + the
    day's full message array (mmap) for payload extraction.
    """

    def __init__(self, msg_path: str, book_path: str, tick_size: int = 100,
                 l_ticks: int = 10, pool_max: int = 1_000_000,
                 knn_block: int = 25):
        self.tick_size = tick_size
        self.l_ticks = l_ticks
        self.knn_block = knn_block
        self.msgs = onp.load(msg_path, mmap_mode='r')
        books = onp.load(book_path, mmap_mode='r')
        n = min(self.msgs.shape[0], books.shape[0])
        # anchors must leave room for a full payload block
        hi = n - knn_block - 1
        if hi <= 0:
            raise ValueError(f'day too short for knn_block={knn_block}: {n} rows')
        stride = max(1, hi // pool_max)
        self.anchors = onp.arange(0, hi, stride, dtype=onp.int64)
        # book row i = state BEFORE message i is applied -> transition payload starts at msg i
        self.features = _vol_profiles_bulk(
            onp.asarray(books[self.anchors, 3:43]), tick_size, l_ticks)
        self._feat_sqnorm = (self.features ** 2).sum(axis=1)

    def query(self, l2_flat: onp.ndarray, K: int, rng: onp.random.Generator) -> int:
        """Paper's rule: among the K nearest anchors pick k ~ U{1..K}. Returns anchor msg index."""
        q = vol_profile(onp.asarray(l2_flat), self.tick_size, self.l_ticks)
        d2 = self._feat_sqnorm - 2.0 * (self.features @ q)   # + ||q||^2 (constant, irrelevant)
        kk = min(K, d2.shape[0])
        top = onp.argpartition(d2, kk - 1)[:kk]
        pick = top[rng.integers(0, kk)]
        return int(self.anchors[pick])

    def take_block(self, anchor: int, m: int, cur_mid_ticked: int,
                   last_time_s: int, last_time_ns: int
                   ) -> Tuple[onp.ndarray, int, int]:
        """
        Take the neighbor's next m messages, re-anchored to the current book.

        Price re-anchoring (paper's additive rule, via the stored mid-relative
        price): price_abs = current tick-floored mid + price_rel * tick.
        Times are rebuilt from the historical dt sequence on top of the current
        clock. Returns (rows (m,6) int64 [etype, side01, size, price, ts, tns],
        new_last_time_s, new_last_time_ns).
        """
        seg = onp.asarray(self.msgs[anchor:anchor + m])
        etype = seg[:, EVENT_TYPE_i].astype(onp.int64)
        side01 = seg[:, DIRECTION_i].astype(onp.int64)
        size = onp.maximum(seg[:, SIZE_i].astype(onp.int64), 1)
        price = cur_mid_ticked + seg[:, PRICE_REL_i].astype(onp.int64) * self.tick_size

        dt_ns_total = seg[:, DTs_i].astype(onp.int64) * 1_000_000_000 \
            + seg[:, DTns_i].astype(onp.int64)
        dt_ns_total = onp.maximum(dt_ns_total, 1)
        t = last_time_s * 1_000_000_000 + last_time_ns + onp.cumsum(dt_ns_total)
        ts = t // 1_000_000_000
        tns = t % 1_000_000_000

        rows = onp.stack([etype, side01, size, price, ts, tns], axis=1)
        return rows, int(ts[-1]), int(tns[-1])


def build_day_pools(data_dir: str, tick_size: int, l_ticks: int,
                    pool_max: int, knn_block: int) -> Dict[str, KnnPool]:
    """Lazy dict date -> KnnPool over the day files in data_dir (built on first use)."""
    msg_files = sorted(Path(data_dir).glob('*message*.npy'))
    book_files = sorted(Path(data_dir).glob('*book*.npy'))
    by_date = {}
    for mf, bf in zip(msg_files, book_files):
        m = DATE_RE.search(mf.name)
        if m:
            by_date[m.group(1)] = (str(mf), str(bf))

    pools: Dict[str, KnnPool] = {}

    class _Lazy:
        def __getitem__(self, date: str) -> KnnPool:
            if date not in pools:
                mf, bf = by_date[date]
                pools[date] = KnnPool(mf, bf, tick_size=tick_size, l_ticks=l_ticks,
                                      pool_max=pool_max, knn_block=knn_block)
            return pools[date]

        def __contains__(self, date: str) -> bool:
            return date in by_date

    return _Lazy()
