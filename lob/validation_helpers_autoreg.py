"""
Autoregressive validation helpers for inference with hidden state support.

This module provides functions that work with hidden state accumulation,
specifically the apply_model function that passes hidden state through the model.
"""
from typing import Optional, Tuple
from lob import encoding
from lob.encoding import Message_Tokenizer, Vocab
import jax
from jax import nn
import flax
from flax.training.train_state import TrainState
import jax.numpy as np
from functools import partial

# Import common functions from the original module
from lob.validation_helpers import (
    syntax_validation_matrix,
    get_valid_mask,
    get_idx_from_field,
    mask_last_msg_in_seq,
    filter_valid_pred,
    sample_pred,
    mask_n_highest,
)


def repeat_book(msg, book, shift_start):
    """
    Repeat book states to match message sequence length.

    This function ensures the book sequence has the same leading dimension
    as the message sequence by repeating book states.

    Args:
        msg: Message sequence array
        book: Book state sequence array
        shift_start: Whether to shift the book (currently unused)

    Returns:
        Tuple of (msg, book) with matching dimensions
    """
    if msg.shape[0] > book.shape[0]:
        book = np.repeat(book, (msg.shape[0]) // book.shape[0], axis=0)
    return (msg, book)


def get_first_time(m_seq_cond, encoder):
    """Extract the time from the last message in a conditioning sequence."""
    last_msg = m_seq_cond[-Message_Tokenizer.MSG_LEN:]
    with jax.ensure_compile_time_eval():
        time_s_start_i, time_s_end_i = get_idx_from_field('time_s')
        time_ns_start_i, time_ns_end_i = get_idx_from_field('time_ns')
    time_init_s, time_init_ns = encoding.decode_time(
        last_msg[time_s_start_i:time_ns_end_i],
        encoder
    )
    return (time_init_s, time_init_ns)


@partial(jax.jit, static_argnums=(4, 5, 6))
def apply_model(
        hidden_state: Tuple,
        m_seq: jax.Array,
        b_seq: jax.Array,
        state: TrainState,
        model: flax.linen.Module,
        batchnorm: bool,
        shift_start: bool,
    ):
    """
    Apply model with hidden state support for autoregressive inference.

    This function works with UNBATCHED inputs (1D m_seq, 2D b_seq) that come from
    generate_batched's vmap. It uses PaddedLobPredModel (unbatched) directly.

    IMPORTANT: This function expects an UNBATCHED model (PaddedLobPredModel).
    The outer vmap in generate_batched handles batching.

    Args:
        hidden_state: Tuple of hidden states for each layer (unbatched - single sample)
        m_seq: Message token sequence (1D array - L_m)
        b_seq: Book state sequence (2D array - L_b x d_book)
        state: TrainState with model parameters
        model: Flax model module (should be PaddedLobPredModel - UNBATCHED)
        batchnorm: Whether to use batch normalization
        shift_start: Whether to shift the book sequence for alignment

    Returns:
        Tuple of (updated_hidden_state, logits)
    """
    # Repeat book to match message length if needed (unbatched)
    m_seq, b_seq = repeat_book(m_seq, b_seq, shift_start)

    # Create reset signals (zeros = no reset) - unbatched
    d_m = np.zeros_like(m_seq, dtype=bool)  # (L_m,)
    d_b = np.zeros((b_seq.shape[0],), dtype=bool)  # (L_b,)
    d_f = np.zeros((m_seq.shape[0] + b_seq.shape[0],), dtype=bool)  # (L_m+L_b,)

    # Integration timesteps - unbatched
    msg_integration_timesteps = np.ones((m_seq.shape[0],))  # (L_m,)
    book_integration_timesteps = np.ones((b_seq.shape[0],))  # (L_b,)

    # Apply unbatched model with __call_rnn__ method
    # Signature: __call_rnn__(hiddens_tuple, x_m, x_b, d_m, d_b, d_f, msg_int_ts, book_int_ts)
    if batchnorm:
        hidden_state_out, logits = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats},
            hidden_state,
            m_seq,
            b_seq,
            d_m,
            d_b,
            d_f,
            msg_integration_timesteps,
            book_integration_timesteps,
            method="__call_rnn__"
        )
    else:
        hidden_state_out, logits = model.apply(
            {"params": state.params},
            hidden_state,
            m_seq,
            b_seq,
            d_m,
            d_b,
            d_f,
            msg_integration_timesteps,
            book_integration_timesteps,
            method="__call_rnn__"
        )

    return hidden_state_out, logits


@partial(jax.jit, static_argnums=(1,))
def fill_predicted_tok(
        pred_logits: jax.Array,
        top_n: int = 1,
        rng: Optional[jax.dtypes.prng_key] = None,
    ) -> jax.Array:
    """
    Get the predicted token from logits.

    When top_n=1, the argmax is used, otherwise a random sample
    from the top_n highest scores is used (proportional to the score).

    Args:
        pred_logits: Predicted logits array
        top_n: Number of top predictions to sample from (1 = argmax)
        rng: Random key for sampling (required when top_n > 1)

    Returns:
        Predicted token value
    """
    if top_n == 1:
        vals = pred_logits.argmax(axis=-1)
    else:
        vals = sample_pred(pred_logits, top_n, rng)
    return vals
