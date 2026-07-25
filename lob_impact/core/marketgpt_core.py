"""
MarketGPT sampler core (message-level transformer from github.com/aaron-wheeler/MarketGPT,
AAPL fine-tuned checkpoint `ckpt_finetune_AAPL_v3.pt`, ~95M params).

Wraps the repo's `equities.fast_model.Transformer` as a batched next-message
sampler for our impact scenarios: we replicate the token-by-token sampling loop
of `Transformer.generate` (KV cache, per-field `relevant_mask`, temperature
1.02 + nucleus top_p 0.98 as in `notebooks/simulate.ipynb`), batched over B
lockstep streams, and replace the author's 1-token cache refresh with a full
24-token refresh of the CORRECTED message so the cache always contains what was
actually applied to the book (including the injected metaorder).

Key facts (verified against the repo):
  * message = 24 tokens over 14 fields (`itch_encoding.Message_Tokenizer`):
    [ticker, type, side, price(sign+digit), fill_size, remain_size, delta_t_s,
    delta_t_ns x3, time_s x2, time_ns x3, price_ref(sign+digit), fill_size_ref,
    time_s_ref x2, time_ns_ref x3]; `encode_msg` consumes the 18-col proc ITCH
    row and SKIPS cols 1/4/12/17 (order_id, price_abs, old_id, old_price_abs).
  * the price token is RELATIVE: price_cents - prev_mid_cents, truncated +-999
    (preproc `_preproc_prices`); at generation the notebook de-relativizes with
    the CURRENT book mid: price = int(mid_cents) + msg[5].
  * ITCH event types: 1=A(add) 2=E(exec) 3=C(exec w/ price) 4=D(cancel,
    remain_size==0 -> full) 5=R(replace). Side: 0=BID 1=ASK for A/C/D/R; for
    E the stored side is FLIPPED (1 -> BID market order, i.e. side of the
    aggressor) which equals the LOBSTER resting-side convention; uniformly
    itch_dir == 1 - our_side01 for every event type.
  * cache roll (StreamingLLM): when the token buffer exceeds new_block_size the
    author crops 24 tokens from the front and passes roll=True once; KVCache
    keeps 1 sink token and rolls the rest left by 24. Keys are stored
    UNROTATED; RoPE is re-applied to the whole cache each forward, which is
    what makes position reuse after the roll consistent.
  * multi-token forwards with a warm cache are only valid at input_pos=0 (the
    causal mask slice `mask[:, :, :seqlen, :seqlen]` does not broadcast against
    scores of width cache_len+seqlen), so the corrected-message refresh feeds
    the 24 tokens one at a time.

ITCH<->LOBSTER mapping (this module, numpy, batched):
  LOBSTER 14-col decoded (our scenarios) -> 18-col proc ITCH for priming, and
  decoded ITCH messages -> JAX-LOB sim actions with the resting-order lookup
  cascade from `simulate.ipynb::find_matching_order`
  (exact size+time -> size-only -> time-only -> any order at price -> fail).
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import fields as dc_fields

import numpy as onp
import torch
import torch.nn.functional as F

MSG_LEN = 24          # tokens per encoded message
NA_VAL = -9999        # itch_encoding.NA_VAL == lob.encoding.NA_VAL
PRICE_REL_CLIP = 999  # vocab 'price' digit range (sign token separate)
SIZE_CLIP = 9999      # vocab 'size' range(10000)
DT_S_MAX = 10         # simulate.ipynb rejects |delta_t_s| > 10
TICKER_AAPL = 7       # line number of AAPL in dataset/symbols/custom_symbols.txt

# 18-col proc ITCH message layout (cols 1/4/12/17 not consumed by encode_msg)
IT_TICKER, IT_OID, IT_TYPE, IT_DIR, IT_PABS, IT_PRICE, IT_FILL, IT_REMAIN, \
    IT_DTS, IT_DTNS, IT_TS, IT_TNS, IT_OLDID, IT_PREF, IT_FREF, IT_TSREF, \
    IT_TNSREF, IT_OLDPABS = range(18)

# our 14-col decoded LOBSTER message layout (qr_scenario constants)
L_OID, L_TYPE, L_DIR, L_PABS, L_PRICE, L_SIZE, L_DTS, L_DTNS, L_TS, L_TNS, \
    L_PREF, L_SREF, L_TSREF, L_TNSREF = range(14)


def load_marketgpt(repo_root: str, ckpt_path: str, device: str = 'cuda'):
    """
    Load the MarketGPT Transformer + itch_encoding from the external repo.
    Returns (model, vocab, itch_encoding_module).
    """
    # MarketGPT uses a top-level `equities` package — resolve from its root,
    # then restore sys.path so the scenario's own imports are unaffected.
    sys.path.insert(0, repo_root)
    try:
        from equities.fast_model import Transformer, ModelArgs
        from equities.data_processing import itch_encoding
    finally:
        sys.path.remove(repo_root)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    margs = dict(ckpt['model_args'])
    # forward() reads params.n_kv_heads directly for the cache shape — resolve
    # the None default (MHA) explicitly.
    if margs.get('n_kv_heads') is None:
        margs['n_kv_heads'] = margs['n_heads']
    known = {f.name for f in dc_fields(ModelArgs)}
    dropped = {k: v for k, v in margs.items() if k not in known}
    if dropped:
        print(f"load_marketgpt: dropping unknown model_args {dropped}")
    model_args = ModelArgs(**{k: v for k, v in margs.items() if k in known})
    print(f"MarketGPT model_args: {model_args}")

    model = Transformer(model_args)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"load_marketgpt: missing={missing} unexpected={unexpected}")
    model.eval()
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    vocab = itch_encoding.Vocab()
    print(f"MarketGPT loaded: {model.get_num_params()/1e6:.1f}M params, "
          f"vocab={len(vocab)}, device={dev}")
    return model, vocab, itch_encoding


class MarketGPTSampler:
    """
    Batched KV-cache next-message sampler.

    Invariant: on entry to sample_next / refresh_context the token buffer
    self.x has length T <= new_block_size - MSG_LEN, so sampling writes cache
    slots up to T+22 and the refresh writes T..T+23, both < new_block_size.
    The crop (+ pending cache roll, consumed by the next forward) happens at
    the END of refresh_context, mirroring the author's post-append crop.
    """

    def __init__(self, model, vocab, itch_enc, device,
                 ctx_msgs: int = 111, new_block_size: int = 2688,
                 temperature: float = 1.02, top_p: float = 0.98,
                 ticker_id: int = TICKER_AAPL):
        self.model = model
        self.vocab = vocab
        self.enc = itch_enc
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.new_block_size = int(new_block_size)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.ticker_id = int(ticker_id)

        # capacity: sink + ctx*24 must leave one full message of headroom
        max_ctx = (self.new_block_size - MSG_LEN - 1) // MSG_LEN
        if ctx_msgs > max_ctx:
            print(f"MarketGPTSampler: ctx_msgs {ctx_msgs} -> {max_ctx} "
                  f"(sink + ctx*{MSG_LEN} + {MSG_LEN} headroom must fit "
                  f"new_block_size={self.new_block_size})")
            ctx_msgs = max_ctx
        self.ctx_msgs = int(ctx_msgs)

        self.x = None                # (B, T) long — corrected token history
        self._pending_roll = False   # cache roll owed to the next forward
        if self.device.type == 'cuda':
            self.amp = torch.amp.autocast(device_type='cuda', dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16))
        else:
            self.amp = nullcontext()

    # ------------------------------------------------------------------
    def _forward(self, tokens: torch.Tensor, input_pos: int, roll: bool = False):
        """One cached forward; returns float32 logits (B, vocab) of the last pos."""
        with torch.inference_mode(), self.amp:
            logits = self.model.forward(
                tokens, None, kv_cache=True,
                max_seq_length=self.new_block_size,
                input_pos=input_pos, roll=roll)
        return logits[:, -1, :].float()

    def prime(self, cond_msgs_itch18: onp.ndarray):
        """
        cond_msgs_itch18: (B, n_ctx, 18) pre-encoded proc-ITCH int rows.
        Encodes the last ctx_msgs messages, prepends the SINK token and
        prefills the KV cache with one batched forward (start=True path).
        """
        msgs = onp.asarray(cond_msgs_itch18)[:, -self.ctx_msgs:, :]
        B = msgs.shape[0]
        X = onp.stack([self.enc.encode_msgs(msgs[b], self.vocab.ENCODING)
                       for b in range(B)])                      # (B, n, 24)
        toks = X.reshape(B, -1)
        assert self.vocab.SINK_TOK == 1
        toks = onp.concatenate(
            [onp.ones((B, 1), dtype=onp.int64), toks], axis=1)  # prepend sink

        # fresh cache for this batch shape (lazily rebuilt inside forward)
        self.model.kv_cache = [None for _ in range(self.model.n_layers)]
        self._pending_roll = False
        self.x = torch.tensor(toks, dtype=torch.long, device=self.device)
        self._forward(self.x, input_pos=0, roll=False)          # prefill 0..T-1

    def sample_next(self) -> onp.ndarray:
        """
        Sample one 24-token message per stream (B, 24) — the token-by-token
        loop from Transformer.generate: 1-token cached forwards, per-position
        relevant_mask, temperature + nucleus top_p, batched multinomial.
        Does NOT advance self.x (resample rounds just overwrite the same cache
        slots; refresh_context finalizes them with the corrected tokens).
        """
        buf = self.x
        for tok_pos in range(MSG_LEN):
            roll = self._pending_roll
            self._pending_roll = False
            logits = self._forward(buf[:, -1:], input_pos=buf.shape[1] - 1, roll=roll)
            logits = logits / self.temperature
            logits = self.model.relevant_mask(
                tok_pos, logits, self.vocab.ENCODING, self.device)
            if self.top_p > 0.0:
                # nucleus filtering (verbatim from Transformer.generate)
                sorted_logits, sorted_indices = torch.sort(logits, descending=False)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits.masked_fill_(indices_to_remove, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            buf = torch.cat((buf, idx_next), dim=1)
        return buf[:, -MSG_LEN:].cpu().numpy()

    def refresh_context(self, corrected_tokens_24) -> None:
        """
        Feed the CORRECTED 24 tokens through the model at the cache slots the
        sampled tokens occupied (T..T+23), one token at a time (multi-token
        forwards are invalid with a warm cache, see module docstring), then
        append them to the buffer and crop/schedule a cache roll if needed.
        Replaces the author's 1-token refresh — after this call the cache
        contains exactly the applied message. Also used to append messages the
        model never sampled (the injected aggressive order).
        """
        tok = torch.as_tensor(onp.asarray(corrected_tokens_24),
                              dtype=torch.long, device=self.device)
        T = self.x.shape[1]
        for j in range(MSG_LEN):
            roll = self._pending_roll and j == 0
            self._forward(tok[:, j:j + 1], input_pos=T + j, roll=roll)
        self._pending_roll = False
        self.x = torch.cat((self.x, tok), dim=1)
        if self.x.shape[1] > self.new_block_size - MSG_LEN:
            # author's crop: drop the oldest message (incl. the sink token from
            # x — the cache keeps its sink K/V); cache rolls on the next forward
            self.x = self.x[:, MSG_LEN:]
            self._pending_roll = True

    # ------------------------------------------------------------------
    def decode_batch(self, tokens: onp.ndarray) -> onp.ndarray:
        """(B, 24) tokens -> (B, 18) decoded proc-ITCH rows (NA for cols 1/4/12/17)."""
        tokens = onp.asarray(tokens)
        return onp.stack([self.enc.decode_msg(tokens[b], self.vocab.ENCODING)
                          for b in range(tokens.shape[0])]).astype(onp.int64)

    def encode_batch(self, msgs18: onp.ndarray) -> onp.ndarray:
        """(B, 18) proc-ITCH rows -> (B, 24) tokens. Fields must be in-vocab
        (encode() is a searchsorted — out-of-range values map silently wrong)."""
        msgs18 = onp.asarray(msgs18)
        return onp.stack([self.enc.encode_msg(msgs18[b], self.vocab.ENCODING)
                          for b in range(msgs18.shape[0])]).astype(onp.int64)


# ======================================================================
# ITCH <-> LOBSTER mapping helpers (numpy, batched)
# ======================================================================

def _clip_or_na(x, lo, hi):
    """Clip to [lo, hi] but pass NA_VAL through untouched."""
    x = onp.asarray(x, dtype=onp.int64)
    return onp.where(x == NA_VAL, NA_VAL, onp.clip(x, lo, hi))


def _ts_clip(v) -> int:
    """time_s into the encodable range (encode() is a silent searchsorted)."""
    return int(min(max(int(v), 0), 999_999))


def _tns_clip(v) -> int:
    return int(min(max(int(v), 0), 999_999_999))


def _mid_cents(l2_or_book: onp.ndarray) -> onp.ndarray:
    """(…, >=4) L2 rows [ask_p, ask_v, bid_p, bid_v, …] in 1e-4$ -> mid in cents."""
    ask = l2_or_book[..., 0].astype(onp.int64)
    bid = l2_or_book[..., 2].astype(onp.int64)
    return ((ask + bid) // 2) // 100


def lobster14_to_itch18(msgs14: onp.ndarray, books_pre: onp.ndarray,
                        tick_size: int, ticker_id: int) -> onp.ndarray:
    """
    Convert our 14-col decoded LOBSTER conditioning messages to 18-col proc
    ITCH rows for priming.

    msgs14: (B, n, 14) — [oid, type, dir01, price_abs, price_rel_ticks, size,
        dt_s, dt_ns, ts, tns, price_ref_ticks, size_ref, ts_ref, tns_ref]
    books_pre: (B, n, 40) — L2 state BEFORE each message (1e-4$ units).

    Event map: LOBSTER {1:add, 2:partial cancel, 3:full delete, 4:execution}
    -> ITCH {1:A, 4:D(remain=NA), 4:D(remain=0), 2:E}. Directions map
    uniformly as itch_dir = 1 - side01 (E's flip == LOBSTER resting-side
    convention, see module docstring). remain_size is NA where LOBSTER does
    not carry it (adds, executions, partial cancels) — the NA token is legal
    for size fields (relevant_mask include_nan). Rel prices are recomputed in
    cents from price_abs and the pre-message mid (the crux of priming); the
    ref-price ticks are rescaled ticks->cents via tick_size (1 tick =
    tick_size 1e-4$ = tick_size/100 cents).
    """
    msgs14 = onp.asarray(msgs14, dtype=onp.int64)
    B, n, _ = msgs14.shape
    et = msgs14[..., L_TYPE]
    side01 = msgs14[..., L_DIR]

    itch_type = onp.select(
        [et == 1, et == 2, et == 3, et == 4],
        [1, 4, 4, 2], default=1)
    remain = onp.where(et == 3, 0, NA_VAL)
    itch_dir = 1 - side01

    price_cents = msgs14[..., L_PABS] // 100
    mid = _mid_cents(onp.asarray(books_pre))
    rel = onp.clip(price_cents - mid, -PRICE_REL_CLIP, PRICE_REL_CLIP)

    fill = onp.clip(msgs14[..., L_SIZE], 0, SIZE_CLIP)
    dt_s = onp.clip(msgs14[..., L_DTS], 0, 999)
    dt_ns = onp.clip(msgs14[..., L_DTNS], 0, 999_999_999)
    ts = onp.clip(msgs14[..., L_TS], 0, 999_999)
    tns = onp.clip(msgs14[..., L_TNS], 0, 999_999_999)

    # ref fields only exist for modification events (2/3/4); NA elsewhere.
    # our price_ref semantics == theirs: the original add's price rel to the
    # mid at ITS OWN time (both preprocs merge the A row's rel price).
    has_ref = (et == 2) | (et == 3) | (et == 4)
    pref_ticks = msgs14[..., L_PREF]
    pref = onp.where(
        has_ref & (pref_ticks != NA_VAL),
        onp.clip(pref_ticks * tick_size // 100, -PRICE_REL_CLIP, PRICE_REL_CLIP),
        NA_VAL)
    sref = onp.where(has_ref, _clip_or_na(msgs14[..., L_SREF], 0, SIZE_CLIP), NA_VAL)
    tsref = onp.where(has_ref, _clip_or_na(msgs14[..., L_TSREF], 0, 999_999), NA_VAL)
    tnsref = onp.where(has_ref, _clip_or_na(msgs14[..., L_TNSREF], 0, 999_999_999), NA_VAL)

    out = onp.empty((B, n, 18), dtype=onp.int64)
    out[..., IT_TICKER] = ticker_id
    out[..., IT_OID] = NA_VAL          # not consumed by encode_msg
    out[..., IT_TYPE] = itch_type
    out[..., IT_DIR] = itch_dir
    out[..., IT_PABS] = price_cents    # not consumed by encode_msg
    out[..., IT_PRICE] = rel
    out[..., IT_FILL] = fill
    out[..., IT_REMAIN] = remain
    out[..., IT_DTS] = dt_s
    out[..., IT_DTNS] = dt_ns
    out[..., IT_TS] = ts
    out[..., IT_TNS] = tns
    out[..., IT_OLDID] = NA_VAL        # not consumed
    out[..., IT_PREF] = pref
    out[..., IT_FREF] = sref
    out[..., IT_TSREF] = tsref
    out[..., IT_TNSREF] = tnsref
    out[..., IT_OLDPABS] = NA_VAL      # not consumed
    return out


def _match_resting(side_arr: onp.ndarray, price: int, ref_size: int,
                   ref_time_ns: int):
    """
    find_matching_order cascade against one JAX-LOB L3 side array
    (rows [price, qty, orderid, traderid, time_s, time_ns]):
    exact size+time -> size-only -> time-only -> any order at price -> None.
    Candidates are FIFO-ordered (time asc) to approximate the book iteration
    order of the author's simulator.
    """
    rows = side_arr[(side_arr[:, 0] == price) & (side_arr[:, 1] > 0)]
    if rows.shape[0] == 0:
        return None
    t_ns = rows[:, 4].astype(onp.int64) * 1_000_000_000 + rows[:, 5].astype(onp.int64)
    order = onp.argsort(t_ns, kind='stable')
    rows, t_ns = rows[order], t_ns[order]
    size_eq = rows[:, 1].astype(onp.int64) == ref_size
    time_eq = t_ns == ref_time_ns
    for mask in (size_eq & time_eq, size_eq, time_eq):
        idx = onp.nonzero(mask)[0]
        if idx.size:
            return rows[idx[0]]
    return rows[0]


def _best_resting(side_arr: onp.ndarray, is_bid: bool):
    """Earliest order at the best price on one L3 side (None if empty)."""
    rows = side_arr[side_arr[:, 1] > 0]
    if rows.shape[0] == 0:
        return None
    best_p = rows[:, 0].max() if is_bid else rows[:, 0].min()
    return _match_resting(side_arr, int(best_p), -1, -1)


def itch_to_lobster_action(decoded18: onp.ndarray, l2_np: onp.ndarray,
                           asks_np: onp.ndarray, bids_np: onp.ndarray,
                           tick_size: int, ticker_id: int,
                           order_id: int):
    """
    Decode one sampled ITCH message per stream into a JAX-LOB sim action.

    decoded18: (B, 18) decode_msg output; l2_np: (B, n_levels*4) live L2;
    asks_np/bids_np: (B, N, 6) live L3 sides; order_id: descending counter
    for new orders (adds / market orders), same convention as the other
    scenarios.

    Returns (act, ok, corrected):
      act: dict of (B,) int64 arrays — primary action
        [etype, side01, size, price(1e-4$), oid] and the R-add second leg
        [is_r, size2, price2] (cancel first, add second);
      ok: (B,) bool — False -> resample this stream;
      corrected: (B, 18) proc-ITCH rows with the ACTUAL matched ref-order
        fields written back (fill_size_ref/time refs; sampled price_ref kept,
        as in the notebook's new_msg construction) for re-encoding into the
        KV cache. time_s/time_ns are left as sampled — the scenario overwrites
        them from the per-stream clock before encoding.

    Event map (ITCH -> our LOBSTER/JAX-LOB types):
      A -> 1 (LO); E -> 4 (MO; stored side01 = resting side, the sim flips
      type-4 internally); C -> treated as E (counted as an execution, the
      modify leg is dropped); D -> 3 (full, remain==0) / 2 (partial);
      R -> cancel old (3) + add new (1), two sim ops in one message slot.
    Prices de-relativize against the LIVE mid: price_cents = mid_cents + rel.
    """
    decoded18 = onp.asarray(decoded18, dtype=onp.int64)
    B = decoded18.shape[0]
    etype = onp.zeros(B, onp.int64)
    side = onp.zeros(B, onp.int64)
    size = onp.ones(B, onp.int64)
    price = onp.zeros(B, onp.int64)
    oid = onp.full(B, order_id, dtype=onp.int64)
    is_r = onp.zeros(B, bool)
    size2 = onp.ones(B, onp.int64)
    price2 = onp.zeros(B, onp.int64)
    ok = onp.zeros(B, bool)
    corrected = decoded18.copy()
    corrected[:, IT_TICKER] = ticker_id
    corrected[:, IT_OID] = order_id

    mid = _mid_cents(l2_np)

    for b in range(B):
        m = decoded18[b]
        it = int(m[IT_TYPE])
        if int(m[IT_TICKER]) != ticker_id:
            continue  # symbol error -> resample (notebook behavior)
        if it < 1 or it > 5:
            continue
        if int(mid[b]) <= 0:
            continue  # one-sided/empty book: no valid mid reference
        rel = int(m[IT_PRICE])
        dt_s = int(m[IT_DTS])
        dt_ns = int(m[IT_DTNS])
        if rel == NA_VAL or dt_s == NA_VAL or dt_ns == NA_VAL or dt_s > DT_S_MAX:
            continue  # price/time error -> resample (notebook behavior)
        s01 = 1 - int(m[IT_DIR])       # uniform for all event types
        if s01 not in (0, 1):
            continue
        p_1e4 = (int(mid[b]) + rel) * 100
        if p_1e4 <= 0:
            continue
        fill = int(m[IT_FILL])
        remain = int(m[IT_REMAIN])
        side_arr = bids_np[b] if s01 == 1 else asks_np[b]

        if it == 1:                                     # A -> add LO
            if fill == NA_VAL or fill <= 0:
                continue
            etype[b], side[b], size[b], price[b] = 1, s01, fill, p_1e4
            corrected[b, IT_PABS] = int(mid[b]) + rel
            corrected[b, IT_OLDID] = NA_VAL
            corrected[b, IT_PREF] = NA_VAL
            corrected[b, IT_FREF] = NA_VAL
            corrected[b, IT_TSREF] = NA_VAL
            corrected[b, IT_TNSREF] = NA_VAL
            ok[b] = True
        elif it in (2, 3):                              # E / C -> market order
            if fill == NA_VAL or fill <= 0:
                continue
            etype[b], side[b], size[b], price[b] = 4, s01, fill, p_1e4
            # executed resting order = earliest at the best on the resting side
            row = _best_resting(side_arr, is_bid=(s01 == 1))
            if row is None:
                continue
            corrected[b, IT_PABS] = int(mid[b]) + rel
            corrected[b, IT_FREF] = min(int(row[1]), SIZE_CLIP)
            corrected[b, IT_TSREF] = _ts_clip(row[4])
            corrected[b, IT_TNSREF] = _tns_clip(row[5])
            ok[b] = True
        elif it == 4:                                   # D -> cancel/delete
            if fill == NA_VAL or fill <= 0 or remain == NA_VAL:
                continue
            if int(m[IT_PREF]) == NA_VAL:
                continue  # price reference error -> resample (notebook)
            ref_size = fill if remain == 0 else remain + fill
            ref_t = (int(m[IT_TSREF]) * 1_000_000_000 + int(m[IT_TNSREF])
                     if m[IT_TSREF] != NA_VAL and m[IT_TNSREF] != NA_VAL else -1)
            row = _match_resting(side_arr, p_1e4, ref_size, ref_t)
            if row is None:
                continue
            matched_qty = int(row[1])
            if remain == 0:
                canc = matched_qty                      # full deletion
                etype[b] = 3
            else:
                canc = fill
                if matched_qty != ref_size:
                    # preserve intent of the partial cancel (notebook ratio)
                    canc = int(matched_qty * fill / ref_size)
                canc = max(1, min(canc, matched_qty))
                etype[b] = 3 if canc >= matched_qty else 2
            side[b], size[b], price[b] = s01, canc, int(row[0])
            oid[b] = int(row[2])
            corrected[b, IT_PABS] = int(row[0]) // 100
            corrected[b, IT_FILL] = min(canc, SIZE_CLIP)
            corrected[b, IT_REMAIN] = min(max(matched_qty - canc, 0), SIZE_CLIP) \
                if remain != 0 else 0
            corrected[b, IT_FREF] = min(matched_qty, SIZE_CLIP)
            corrected[b, IT_TSREF] = _ts_clip(row[4])
            corrected[b, IT_TNSREF] = _tns_clip(row[5])
            ok[b] = True
        else:                                           # R -> cancel old + add new
            if fill == NA_VAL or fill <= 0:
                continue
            pref = int(m[IT_PREF])
            sref = int(m[IT_FREF])
            if pref == NA_VAL or sref == NA_VAL:
                continue
            # process_message: ref_price = mid + price_ref (rel to CURRENT mid)
            ref_p_1e4 = (int(mid[b]) + pref) * 100
            ref_t = (int(m[IT_TSREF]) * 1_000_000_000 + int(m[IT_TNSREF])
                     if m[IT_TSREF] != NA_VAL and m[IT_TNSREF] != NA_VAL else -1)
            row = _match_resting(side_arr, ref_p_1e4, sref, ref_t)
            if row is None:
                continue
            etype[b], side[b], size[b], price[b] = 3, s01, int(row[1]), int(row[0])
            oid[b] = int(row[2])
            is_r[b], size2[b], price2[b] = True, fill, p_1e4
            corrected[b, IT_PABS] = int(mid[b]) + rel
            corrected[b, IT_OLDID] = int(row[2])
            corrected[b, IT_FREF] = min(int(row[1]), SIZE_CLIP)
            corrected[b, IT_TSREF] = _ts_clip(row[4])
            corrected[b, IT_TNSREF] = _tns_clip(row[5])
            corrected[b, IT_OLDPABS] = int(row[0]) // 100
            ok[b] = True

    act = {'etype': etype, 'side01': side, 'size': size, 'price': price,
           'oid': oid, 'is_r': is_r, 'size2': size2, 'price2': price2}
    return act, ok, corrected


def build_aggressive_itch18(msg_dec14: onp.ndarray, l2_pre: onp.ndarray,
                            asks_pre: onp.ndarray, bids_pre: onp.ndarray,
                            ticker_id: int) -> onp.ndarray:
    """
    Build the proc-ITCH E row for the injected aggressive market order so the
    model's KV cache sees the metaorder.

    msg_dec14: (B, 14) create_aggressive_order decoded output (side01 stored =
    resting side); l2/asks/bids: book state BEFORE the insertion. Rel price =
    executed price (cents) - pre-event mid (cents). remain_size=0 (the order
    is capped at the available best-level volume), dt = 0s/1ns. Ref fields
    from the earliest resting order at the aggressed best; price_ref = NA
    (the original add's own rel price is unknowable here).
    """
    msg_dec14 = onp.asarray(msg_dec14, dtype=onp.int64)
    B = msg_dec14.shape[0]
    mid = _mid_cents(l2_pre)
    out = onp.full((B, 18), NA_VAL, dtype=onp.int64)
    out[:, IT_TICKER] = ticker_id
    out[:, IT_TYPE] = 2                                 # E
    side01 = msg_dec14[:, L_DIR]                        # resting side
    out[:, IT_DIR] = 1 - side01
    price_cents = msg_dec14[:, L_PABS] // 100
    out[:, IT_PABS] = price_cents
    out[:, IT_PRICE] = onp.clip(price_cents - mid, -PRICE_REL_CLIP, PRICE_REL_CLIP)
    out[:, IT_FILL] = onp.clip(msg_dec14[:, L_SIZE], 1, SIZE_CLIP)
    out[:, IT_REMAIN] = 0
    out[:, IT_DTS] = 0
    out[:, IT_DTNS] = 1
    out[:, IT_TS] = onp.clip(msg_dec14[:, L_TS], 0, 999_999)
    out[:, IT_TNS] = onp.clip(msg_dec14[:, L_TNS], 0, 999_999_999)
    for b in range(B):
        side_arr = bids_pre[b] if side01[b] == 1 else asks_pre[b]
        row = _best_resting(side_arr, is_bid=(side01[b] == 1))
        if row is not None:
            out[b, IT_FREF] = min(int(row[1]), SIZE_CLIP)
            out[b, IT_TSREF] = _ts_clip(row[4])
            out[b, IT_TNSREF] = _tns_clip(row[5])
    return out
