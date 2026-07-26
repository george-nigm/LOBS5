"""
DeepMarket CGAN sampler (Coletta-style conditional GAN, reimplementation +
TSLA/INTC Jan-2015 checkpoints from github.com/LeonardoBerti00/DeepMarket).

Wraps the pretrained GANEngine as a batched next-message sampler for our impact
scenarios: we re-implement WorldAgent's feature bookkeeping
(`_preprocess_market_features_for_cgan`) and decode
(`_postprocess_generated_gan`) in vectorized numpy/torch against the live
JAX-LOB L2 state, instead of running ABIDES.

Key facts (verified against the repo):
  * conditioning y = (B, 256, 9): rolling windows over 511 L2 snapshots + 511
    order signs; features per row j (snapshot index j+255):
    [vol_imb_1, vol_imb_5, abs_vol_1, abs_vol_5, spread, osi_256, osi_128, ret_1, ret_50]
  * returns/vol-imbalances/abs-volumes/spread are z-scored with the HARDCODED
    Jan-2015 train stats in DeepMarket/constants.py (reused verbatim — they are
    part of the checkpoint contract); OSIs are just /256, /128.
  * generator output (B,1,7) tanh -> post_process_order quantizes col0 (order
    type, stock-specific thresholds), col2 (direction sign), col6 (qty type);
    decode maps to {LO=1, cancel=3, MO=4} with depth-relative prices from the
    LIVE book -> absolute 2026 price levels never touch the model.
  * interarrival times are NOT generated: gamma fit on the historical day's dt.

All 9 conditioning features are price-LEVEL-free (spread/returns/volumes), so
no price re-anchoring is needed for the 2026 transfer (unlike TRADES).
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as onp
import torch


class CganDmSampler:
    def __init__(self, deepmarket_root: str, ckpt_path: str, stock2015: str,
                 device: str = 'cuda'):
        assert stock2015 in ('TSLA', 'INTC')
        self.stock2015 = stock2015
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # DeepMarket's generic top-level module names (constants/configuration/utils/
        # models) collide with modules JAX and our tree already loaded — the bridge
        # stashes those for the duration of the import (see core/deepmarket_bridge.py).
        from lob_impact.core.deepmarket_bridge import load_engine
        self.engine, dm = load_engine(deepmarket_root, ckpt_path, 'gan', self.device)

        got_stock = getattr(self.engine, 'chosen_stock', None)
        print(f"CGAN engine loaded: chosen_stock={got_stock} "
              f"(expected {stock2015}), device={self.device}")
        if got_stock is not None and got_stock != stock2015:
            raise ValueError(f'checkpoint stock {got_stock} != requested {stock2015}')
        self.noise_dim = self.engine.generator_lstm_hidden_state_dim

        # 2015 z-score stats (constants.py, "normalization_terms['lob']" layout)
        S = stock2015
        g = lambda name: getattr(dm, f'{S}_{name}')
        self.mean_spread, self.std_spread = g('MEAN_SPREAD'), g('STD_SPREAD')
        self.mean_return, self.std_return = g('MEAN_RETURN'), g('STD_RETURN')
        self.mean_vol_imb, self.std_vol_imb = g('MEAN_VOL_IMB'), g('STD_VOL_IMB')
        self.mean_abs_vol, self.std_abs_vol = g('MEAN_ABS_VOL'), g('STD_ABS_VOL')
        self.mean_cancel_depth, self.std_cancel_depth = g('MEAN_CANCEL_DEPTH'), g('STD_CANCEL_DEPTH')
        self.mean_size_100, self.std_size_100 = g('MEAN_SIZE_100'), g('STD_SIZE_100')
        self.mean_depth, self.std_depth = g('EVENT_MEAN_DEPTH'), g('EVENT_STD_DEPTH')
        self.mean_size, self.std_size = g('EVENT_MEAN_SIZE'), g('EVENT_STD_SIZE')

        self.seq = 256
        self.ring_len = 2 * self.seq - 1  # 511

        self.books = None   # (B, 511, 40) float64
        self.dirs = None    # (B, 511) float32, +-1

    # ------------------------------------------------------------------
    def init_state(self, cond_books: onp.ndarray, cond_dirs01: onp.ndarray):
        """
        cond_books: (B, n_cond+1, 40) L2 snapshots; cond_dirs01: (B, n_cond) in {0,1}.
        Left-pads to ring length 511 with the earliest row (n_cond >= 510 avoids padding).
        """
        B = cond_books.shape[0]
        books = onp.asarray(cond_books, dtype=onp.float64)[:, -self.ring_len:, :]
        dirs = (2.0 * onp.asarray(cond_dirs01, dtype=onp.float32) - 1.0)[:, -self.ring_len:]
        if books.shape[1] < self.ring_len:
            pad = self.ring_len - books.shape[1]
            books = onp.concatenate(
                [onp.repeat(books[:, :1, :], pad, axis=1), books], axis=1)
        if dirs.shape[1] < self.ring_len:
            pad = self.ring_len - dirs.shape[1]
            dirs = onp.concatenate([onp.zeros((B, pad), onp.float32), dirs], axis=1)
        self.books = books
        self.dirs = dirs

    def push(self, l2_rows: onp.ndarray, dirs01: onp.ndarray):
        """Append one applied message's post-state (B,40) and direction01 (B,)."""
        self.books = onp.concatenate(
            [self.books[:, 1:, :], onp.asarray(l2_rows, onp.float64)[:, None, :]], axis=1)
        d = (2.0 * onp.asarray(dirs01, onp.float32) - 1.0)[:, None]
        self.dirs = onp.concatenate([self.dirs[:, 1:], d], axis=1)

    # ------------------------------------------------------------------
    def _features(self) -> torch.Tensor:
        """Vectorized replica of _preprocess_market_features_for_cgan -> (B,256,9)."""
        b = self.books                       # (B,511,40)
        prices = b[:, :, 0::2]               # sell1,buy1,sell2,buy2,... (B,511,20)
        sizes = b[:, :, 1::2]                # vsell1,vbuy1,...          (B,511,20)

        with onp.errstate(divide='ignore', invalid='ignore'):
            vi1 = sizes[:, :, 1] / (sizes[:, :, 1] + sizes[:, :, 0])
            buy5 = sizes[:, :, 1] + sizes[:, :, 3] + sizes[:, :, 5] + sizes[:, :, 7] + sizes[:, :, 9]
            tot5 = sizes[:, :, :10].sum(axis=2)
            vi5 = buy5 / tot5
        av1 = sizes[:, :, 1] + sizes[:, :, 0]
        av5 = tot5
        spread = prices[:, :, 0] - prices[:, :, 1]
        mid = (prices[:, :, 0] + prices[:, :, 1]) / 2.0

        seq, ring = self.seq, self.ring_len
        # rolling order-sign sums over the 511-long ring: window j -> dirs[j:j+256]
        c = onp.concatenate(
            [onp.zeros((self.dirs.shape[0], 1), onp.float64),
             onp.cumsum(self.dirs.astype(onp.float64), axis=1)], axis=1)  # (B,512)
        j = onp.arange(seq)
        osi256 = (c[:, j + 256] - c[:, j]) / 256.0
        osi128 = (c[:, j + 256] - c[:, j + 128]) / 128.0

        with onp.errstate(divide='ignore', invalid='ignore'):
            ret1 = mid[:, j + 255] / mid[:, j + 254] - 1.0
            ret50 = mid[:, j + 255] / mid[:, j + 205] - 1.0

        # snapshot-level features at row j+255
        sel = j + 255
        feats = onp.stack([
            (vi1[:, sel] - self.mean_vol_imb) / self.std_vol_imb,
            (vi5[:, sel] - self.mean_vol_imb) / self.std_vol_imb,
            (av1[:, sel] - self.mean_abs_vol) / self.std_abs_vol,
            (av5[:, sel] - self.mean_abs_vol) / self.std_abs_vol,
            (spread[:, sel] - self.mean_spread) / self.std_spread,
            osi256,
            osi128,
            (ret1 - self.mean_return) / self.std_return,
            (ret50 - self.mean_return) / self.std_return,
        ], axis=2)                           # (B,256,9)
        feats = onp.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(feats.astype(onp.float32)).to(self.device)

    @torch.no_grad()
    def sample_raw(self, gen: torch.Generator | None = None) -> onp.ndarray:
        """One generator call for the whole batch -> post-processed (B,7) numpy."""
        y = self._features()
        B = y.shape[0]
        noise = torch.randn(B, 1, self.noise_dim, generator=gen,
                            device=self.device, dtype=y.dtype)
        out = self.engine.sample(noise=noise, cond_market_features=y)  # (B,1,7)
        out = self.engine.post_process_order(out)
        return out[:, 0, :].detach().cpu().numpy()

    # ------------------------------------------------------------------
    def decode(self, raw: onp.ndarray, l2: onp.ndarray
               ) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray]:
        """
        Vectorized replica of _postprocess_generated_gan against live L2 rows.

        raw: (B,7) post-processed generator output; l2: (B,40) live JAX-LOB state.
        Returns (etype, side01, size, price, ok) int64/bool arrays. ok=False rows
        must be resampled or replaced by the caller (upstream returns None there).
        """
        B = raw.shape[0]
        otype = raw[:, 0].astype(onp.int64)          # {-1,0,1}
        etype = onp.where(otype == -1, 1, onp.where(otype == 0, 3, 4))
        direction = raw[:, 2].astype(onp.int64)      # {-1,+1}; +1 = buy
        qty_type = raw[:, 6].astype(onp.int64)

        depth = (raw[:, 3] * self.std_depth + self.mean_depth).astype(onp.int64)
        cancel_depth = (raw[:, 4] * self.std_cancel_depth + self.mean_cancel_depth).astype(onp.int64)
        size100 = raw[:, 5] * self.std_size_100 + self.mean_size_100
        size = (raw[:, 1] * self.std_size + self.mean_size).astype(onp.int64)
        size = onp.where(qty_type == -1, size100.astype(onp.int64) * 100, size)

        ask_p = l2[:, 0::4].astype(onp.int64)        # (B,10)
        bid_p = l2[:, 2::4].astype(onp.int64)

        ok = onp.ones(B, dtype=bool)
        ok &= size > 0
        ok &= (ask_p[:, 0] > 0) & (bid_p[:, 0] > 0)

        depth = onp.maximum(depth, 0)
        cd_ok = (cancel_depth >= 0) & (cancel_depth <= 9)
        cancel_depth = onp.clip(cancel_depth, 0, 9)

        # LO price: touch -/+ depth ticks; reject if beyond the visible 10th level
        lo_price_buy = bid_p[:, 0] - depth * 100
        lo_price_sell = ask_p[:, 0] + depth * 100
        lo_price = onp.where(direction == 1, lo_price_buy, lo_price_sell)
        last_buy = bid_p[:, 9]
        last_sell = ask_p[:, 9]
        lo_out = onp.where(
            direction == 1,
            (lo_price < last_buy) & (last_buy > 0),
            (lo_price > last_sell) & (last_sell > 0))

        # cancel price: the book price at level cancel_depth on the order's side
        rows = onp.arange(B)
        ca_price = onp.where(direction == 1,
                             bid_p[rows, cancel_depth], ask_p[rows, cancel_depth])

        # MO: price = opposite touch; direction flips to the resting side (LOBSTER convention)
        mo_price = onp.where(direction == 1, ask_p[:, 0], bid_p[:, 0])

        price = onp.where(etype == 1, lo_price,
                          onp.where(etype == 3, ca_price, mo_price))
        eff_dir = onp.where(etype == 4, -direction, direction)   # +-1
        side01 = ((eff_dir + 1) // 2).astype(onp.int64)

        ok &= onp.where(etype == 1, ~lo_out, True)
        ok &= onp.where(etype == 3, cd_ok & (ca_price > 0), True)
        ok &= price > 0

        return etype, side01, size, price, ok
