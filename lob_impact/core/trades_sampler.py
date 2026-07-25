"""
TRADES sampler (transformer-DDPM per-order generator, TSLA/INTC Jan-2015
checkpoints from github.com/LeonardoBerti00/DeepMarket).

Batched next-message wrapper replicating WorldAgent's bookkeeping
(`_preprocess_orders_for_diff_cond`, `_z_score_orderbook`,
`_postprocess_generated_TRADES`) in vectorized numpy against the live JAX-LOB
state, instead of ABIDES.

Model I/O (verified against the repo):
  * cond_orders (B,255,6): [dt_s, event_code, size, price_cents, direction, depth]
    - event_code {0=LO,1=cancel,2=MO} (int; embedded to 3-dim inside the engine)
    - MO direction flipped to the aggressor side (direction * -1 for type 4)
    - depth = ticks from the SAME-side touch; for LOs measured against the
      POST-order book (index j+1), else the pre-order book (index j)
    - dt/size/price/depth z-scored with the hardcoded Jan-2015 stats
  * cond_lob (B,256,40): prices /100 then z-scored (LOB price stats), volumes
    z-scored (LOB size stats)
  * x (B,1,6) zeros (zeroed inside sample anyway; col1 must be a valid emb index)
  * output (B,1,8): [time_z, type_emb(3), size_z, price_z(IGNORED), direction, depth_z]
    decode: type = argmin L1 to the frozen embedder rows -> {1,3,4}; reject
    size<0 or >1000; LO/cancel price = touch -/+ depth*100 (book-relative);
    MO price = opposite touch, direction flips to resting side.

PRICE RE-ANCHORING (mandatory for the 2026 transfer): the conditioning z-scores
use 2015 absolute price means (TSLA $201.79, INTC $36.36); 2026 windows are
~20-40 sigma off. Per batch-sample we subtract a constant cent shift
anchor = round(mid0_cents - 2015_mean_cents) from every conditioning price
(messages + LOB), which lands the window's initial mid exactly on the 2015
train mean while preserving spreads/queues; decode is book-relative, so output
prices stay anchored to the real 2026 book.
"""

from __future__ import annotations

import sys
from typing import Tuple

import numpy as onp
import torch


class TradesSampler:
    SEQ = 256          # cond_lob rows
    N_ORD = 255        # cond_orders rows

    def __init__(self, deepmarket_root: str, ckpt_path: str, stock2015: str,
                 device: str = 'cuda'):
        assert stock2015 in ('TSLA', 'INTC')
        self.stock2015 = stock2015
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        sys.path.insert(0, deepmarket_root)
        try:
            import constants as dm  # noqa: F401
            from models.diffusers.diffusion_engine import DiffusionEngine
            _orig_load = torch.load
            torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
            try:
                self.engine = DiffusionEngine.load_from_checkpoint(
                    ckpt_path, map_location=self.device)
            finally:
                torch.load = _orig_load
            g = lambda name: getattr(dm, f'{stock2015}_{name}')
        finally:
            sys.path.remove(deepmarket_root)

        self.engine.eval()
        self.engine.to(self.device)
        for p in self.engine.parameters():
            p.requires_grad_(False)

        print(f"TRADES engine loaded: diffusionsteps={getattr(self.engine, 'num_diffusionsteps', '?')} "
              f"cond_size={getattr(self.engine, 'cond_size', '?')} device={self.device}")

        # 2015 stats ("normalization_terms": event=[m_size,s_size,m_price,s_price,m_time,s_time,m_depth,s_depth],
        # lob=[m_size,s_size,m_price,s_price])
        self.m_size, self.s_size = g('EVENT_MEAN_SIZE'), g('EVENT_STD_SIZE')
        self.m_price, self.s_price = g('EVENT_MEAN_PRICE'), g('EVENT_STD_PRICE')
        self.m_time, self.s_time = g('EVENT_MEAN_TIME'), g('EVENT_STD_TIME')
        self.m_depth, self.s_depth = g('EVENT_MEAN_DEPTH'), g('EVENT_STD_DEPTH')
        self.m_lob_size, self.s_lob_size = g('LOB_MEAN_SIZE_10'), g('LOB_STD_SIZE_10')
        self.m_lob_price, self.s_lob_price = g('LOB_MEAN_PRICE_10'), g('LOB_STD_PRICE_10')

        self.type_emb = self.engine.type_embedder.weight.data.detach().cpu().numpy()  # (3,3)

        self.orders = None       # (B,255,6) raw cents/seconds
        self.lob = None          # (B,256,40) raw price units (1e-4 $)
        self.anchor = None       # (B,) cent shift subtracted from every cond price

    # ------------------------------------------------------------------
    @staticmethod
    def _event_code(etype: onp.ndarray) -> onp.ndarray:
        # {1 -> 0 LO, 2/3 -> 1 cancel, 4 -> 2 MO}
        return onp.where(etype == 1, 0, onp.where(etype == 4, 2, 1))

    def _order_row(self, dt_s, etype, size, price_cents, dir_pm, depth):
        """raw order feature rows (…,6): [dt, code, size, price_cents, dir(with MO flip), depth]"""
        code = self._event_code(etype)
        d = onp.where(etype == 4, -dir_pm, dir_pm)
        return onp.stack([dt_s, code.astype(onp.float64), size.astype(onp.float64),
                          price_cents.astype(onp.float64), d.astype(onp.float64),
                          depth.astype(onp.float64)], axis=-1)

    @staticmethod
    def _depth_vs_book(price, dir_pm, book_row):
        """ticks from same-side touch (clipped at 0), book_row = flat 40-col L2."""
        bid = book_row[..., 2]
        ask = book_row[..., 0]
        d = onp.where(dir_pm == 1, (bid - price) // 100, (price - ask) // 100)
        return onp.maximum(d, 0)

    def init_state(self, cond_msgs: onp.ndarray, cond_books: onp.ndarray):
        """
        cond_msgs: (B, n_cond, 14) decoded messages; cond_books: (B, n_cond+1, 40).
        Builds the raw rings and the per-sample price anchor.
        """
        B, n_cond, _ = cond_msgs.shape
        assert n_cond >= self.N_ORD, f'need n_cond >= {self.N_ORD}'
        m = cond_msgs[:, -self.N_ORD:, :].astype(onp.float64)
        bpost = cond_books[:, -(self.N_ORD + 1):, :].astype(onp.float64)  # books around those msgs

        etype = m[:, :, 1].astype(onp.int64)
        dir_pm = 2 * m[:, :, 2].astype(onp.int64) - 1
        price_cents = m[:, :, 3] / 100.0
        size = m[:, :, 5]
        dt_s = m[:, :, 6] + m[:, :, 7] / 1e9

        # depth: LO vs post-order book (j+1), others vs pre-order book (j)
        book_pre = bpost[:, :-1, :]
        book_post = bpost[:, 1:, :]
        ref = onp.where((etype == 1)[:, :, None], book_post, book_pre)
        depth = onp.maximum(onp.where(dir_pm == 1,
                                      (ref[:, :, 2] - m[:, :, 3]) // 100,
                                      (m[:, :, 3] - ref[:, :, 0]) // 100), 0)

        self.orders = self._order_row(dt_s, etype, size, price_cents, dir_pm, depth)
        self.lob = cond_books[:, -self.SEQ:, :].astype(onp.float64)
        if self.lob.shape[1] < self.SEQ:
            pad = self.SEQ - self.lob.shape[1]
            self.lob = onp.concatenate(
                [onp.repeat(self.lob[:, :1, :], pad, axis=1), self.lob], axis=1)

        mid0 = (self.lob[:, -1, 0] + self.lob[:, -1, 2]) / 2.0 / 100.0  # cents
        self.anchor = onp.round(mid0 - self.m_price)                    # cents
        print(f"TRADES price anchors (cents): min={self.anchor.min():.0f} "
              f"max={self.anchor.max():.0f} (2015 mean={self.m_price:.1f})")

    def push(self, etype, dir01, size, price, dt_s, l2_pre, l2_new):
        """Append one applied message + its post-state to the rings.
        Depth convention mirrors upstream: LO vs post-order book, others vs pre-order book."""
        etype_a = onp.asarray(etype, onp.int64)
        dir_pm = 2 * onp.asarray(dir01, onp.int64) - 1
        price = onp.asarray(price, onp.float64)
        ref = onp.where((etype_a == 1)[:, None],
                        onp.asarray(l2_new, onp.float64),
                        onp.asarray(l2_pre, onp.float64))
        depth = onp.maximum(onp.where(dir_pm == 1,
                                      (ref[:, 2] - price) // 100,
                                      (price - ref[:, 0]) // 100), 0)
        row = self._order_row(onp.asarray(dt_s, onp.float64),
                              onp.asarray(etype, onp.int64),
                              onp.asarray(size, onp.float64),
                              price / 100.0, dir_pm, depth)
        self.orders = onp.concatenate([self.orders[:, 1:, :], row[:, None, :]], axis=1)
        self.lob = onp.concatenate(
            [self.lob[:, 1:, :], onp.asarray(l2_new, onp.float64)[:, None, :]], axis=1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_raw(self) -> onp.ndarray:
        """One reverse-diffusion pass for the batch -> raw (B,8) numpy."""
        B = self.orders.shape[0]
        o = self.orders.copy()
        o[:, :, 3] = o[:, :, 3] - self.anchor[:, None]          # price re-anchor (cents)
        cond_orders = onp.stack([
            (o[:, :, 0] - self.m_time) / self.s_time,
            o[:, :, 1],                                          # int event code
            (o[:, :, 2] - self.m_size) / self.s_size,
            (o[:, :, 3] - self.m_price) / self.s_price,
            o[:, :, 4],
            (o[:, :, 5] - self.m_depth) / self.s_depth,
        ], axis=2)

        lob = self.lob.copy()
        lob[:, :, 0::2] = lob[:, :, 0::2] / 100.0 - self.anchor[:, None, None]
        lob[:, :, 0::2] = (lob[:, :, 0::2] - self.m_lob_price) / self.s_lob_price
        lob[:, :, 1::2] = (lob[:, :, 1::2] - self.m_lob_size) / self.s_lob_size

        cond_orders_t = torch.from_numpy(cond_orders.astype(onp.float32)).to(self.device)
        cond_lob_t = torch.from_numpy(lob.astype(onp.float32)).to(self.device)
        x = torch.zeros(B, 1, 6, dtype=torch.float32, device=self.device)

        out = self.engine.sample(cond_orders=cond_orders_t, x=x, cond_lob=cond_lob_t)
        return out[:, 0, :].detach().cpu().numpy()

    # ------------------------------------------------------------------
    def decode(self, raw: onp.ndarray, l2: onp.ndarray
               ) -> Tuple[onp.ndarray, ...]:
        """
        Vectorized _postprocess_generated_TRADES against live L2 rows.
        raw: (B,8); l2: (B,40). Returns (etype, side01, size, price, dt_s, ok).
        """
        B = raw.shape[0]
        # type = argmin L1 distance to the frozen embedder rows
        d = onp.abs(raw[:, None, 1:4] - self.type_emb[None, :, :]).sum(axis=2)  # (B,3)
        code = d.argmin(axis=1) + 1                       # {1,2,3}
        etype = onp.where(code >= 2, code + 1, code)      # {1,3,4}

        direction = onp.where(raw[:, 6] < 0, -1, 1)
        size = onp.round(raw[:, 4] * self.s_size + self.m_size)
        depth = onp.round(raw[:, 7] * self.s_depth + self.m_depth)
        dt_s = raw[:, 0] * self.s_time + self.m_time
        dt_s = onp.where(dt_s <= 0, 1e-7, dt_s)

        ask_p = l2[:, 0::4].astype(onp.int64)
        bid_p = l2[:, 2::4].astype(onp.int64)

        ok = onp.ones(B, dtype=bool)
        ok &= (size >= 1) & (size <= 1000)
        ok &= (ask_p[:, 0] > 0) & (bid_p[:, 0] > 0)
        depth = onp.maximum(depth, 0).astype(onp.int64)

        # LO / cancel price: same-side touch -/+ depth ticks
        rel_price = onp.where(direction == 1,
                              bid_p[:, 0] - depth * 100,
                              ask_p[:, 0] + depth * 100)
        out_of_depth = onp.where(
            direction == 1,
            (rel_price < bid_p[:, 9]) & (bid_p[:, 9] > 0),
            (rel_price > ask_p[:, 9]) & (ask_p[:, 9] > 0))

        mo_price = onp.where(direction == 1, ask_p[:, 0], bid_p[:, 0])
        price = onp.where(etype == 4, mo_price, rel_price)
        eff_dir = onp.where(etype == 4, -direction, direction)
        side01 = ((eff_dir + 1) // 2).astype(onp.int64)

        ok &= onp.where(etype != 4, ~out_of_depth, True)
        ok &= price > 0

        return etype.astype(onp.int64), side01, size.astype(onp.int64), \
            price.astype(onp.int64), dt_s, ok
