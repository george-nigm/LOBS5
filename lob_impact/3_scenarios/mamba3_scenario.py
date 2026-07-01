#!/usr/bin/env python
"""
Aggressive Scenario with Continuous Hidden State -- Mamba3 variant.

Adapted from 1.aggressive_scenario_s5.py for Mamba3 checkpoints (e.g.
j3417629) trained in the NEW codebase under exp_R1_Mamba3, using the
26-token encoding.

Key differences from the S5 scenario:
- Resolves lob.*/s5.* to the NEW codebase (exp_R1_Mamba3) via sys.path.
- Installs the 26-token encoding as 'lob.encoding' before any lob.* import.
- Sets MAMBA3_LEGACY_NORM=1 and TOKEN_MODE=26tok (required by the ckpt).
- Builds the model via init_train_state with args.ssm_type=='mamba3'.
- Uses the mamba3 hidden-carry recipe (model.initialize_carry with the
  mamba3 head/state/rope args).
- Imports the grafted generator inference_w_insertions_mamba3.

Everything else (config keys, insertion_schedule construction, batch loop,
AOT compile of generate_batched, output writing) is unchanged so that
run_experiments.sh can call it identically.

Usage:
    python -u lob_impact/3_scenarios/1.aggressive_scenario_mamba3.py --config <config.yaml>
"""

import os
import sys

# ── Step 0: Mamba3 / encoding environment (BEFORE any lob.*/s5.* import) ──
os.environ.setdefault("MAMBA3_LEGACY_NORM", "1")  # REQUIRED for the j3417629 ckpt
os.environ.setdefault("TOKEN_MODE", "26tok")

# NEW model codebase (Mamba3-capable). lob.*/s5.* must resolve here.
EXP = "/lus/lfs1aip2/projects/public/s5e/quant_team/quant/AlphaTrade/experiments/exp_R1_Mamba3"
sys.path.insert(0, EXP)

# Install the 26-token encoding as 'lob.encoding' so the whole downstream
# stack (inference, validation_helpers, ...) uses 26-tok transparently.
import lob.encoding_26tok as _enc26
sys.modules["lob.encoding"] = _enc26

import argparse
import yaml
from datetime import datetime
from pathlib import Path
from functools import partial
from typing import Any, Dict, Optional, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import jax
import jax.numpy as jnp
import numpy as onp
from tqdm import tqdm

# Add repo root to path so lob_impact.* resolves (and as a fallback for any
# repo-local modules). EXP stays ahead of it on sys.path for lob.*/s5.*.
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_folder_path)

# Add AlphaTrade submodule (mounted at /AlphaTrade in container) so that
# gymnax_exchange resolves. Also try the repo-relative path for local runs.
if os.path.exists('/AlphaTrade'):
    sys.path.insert(0, '/AlphaTrade')
else:
    sys.path.insert(0, os.path.join(parent_folder_path, 'Alphatrade'))

from gymnax_exchange.jaxob.jorderbook import OrderBook, LobState
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
import gymnax_exchange.jaxob.jaxob_constants as cst
import gymnax_exchange.jaxob.JaxOrderBookArrays as job

from lob.encoding import Vocab, Message_Tokenizer
from lob_impact.core import inference_w_insertions_mamba3 as inference
import lob.validation_helpers as valh
import lob.encoding as encoding
import preproc
from lob.init_train import init_train_state, load_checkpoint, load_metadata

# Indices for DECODED message fields (from inference_no_errcorr.py)
ORDER_ID_i = 0
EVENT_TYPE_i = 1
DIRECTION_i = 2
PRICE_ABS_i = 3
PRICE_i = 4
SIZE_i = 5
DTs_i = 6
DTns_i = 7
TIMEs_i = 8
TIMEns_i = 9
PRICE_REF_i = 10
SIZE_REF_i = 11
TIMEs_REF_i = 12
TIMEns_REF_i = 13

AGGRESSIVE_ORDER_ID = 77777777


class TeeLogger:
    """Duplicates output to both console and log file."""
    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', buffering=1)  # line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def setup_logging(save_folder: Path, log_name: str = 'experiment.log') -> TeeLogger:
    """Set up logging to both console and file."""
    log_file = save_folder / log_name
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee


def create_experiment_folder(base_dir: str) -> Path:
    """Create experiment folder with sequential number and timestamp."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Find max experiment index
    max_idx = 0
    for entry in base.iterdir():
        if entry.is_dir() and entry.name.startswith("exp_"):
            parts = entry.name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))

    next_idx = max_idx + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder = base / f"exp_{next_idx}_{timestamp}"
    new_folder.mkdir(parents=True, exist_ok=False)

    return new_folder


@jax.jit
def create_aggressive_order(
    sim: OrderBook,
    sim_state: LobState,
    last_msg_decoded: jax.Array,
    tick_size: int,
    event_type: int,
    direction: int,
    order_volume: int,
) -> Tuple[jax.Array, jax.Array]:
    """
    Create an aggressive market order based on current book state.

    Args:
        sim: OrderBook simulator
        sim_state: Current LOB state
        last_msg_decoded: Last generated message (for time info)
        tick_size: Tick size
        event_type: Order type (4 = market order)
        direction: 0 = buy (hit ask), 1 = sell (hit bid)
        order_volume: Order size

    Returns:
        sim_msg: Message for simulator
        msg_decoded: Decoded message for storage
    """
    # Get best price on the side we will aggress
    price = jax.lax.cond(
        direction == 0,
        lambda: sim.get_best_ask(sim_state),
        lambda: sim.get_best_bid(sim_state)
    )

    # Get available volume at best level
    best_bid_ask = sim.get_best_bid_and_ask_inclQuants(sim_state)
    avail = jax.lax.cond(
        direction == 0,
        lambda: best_bid_ask[1][1],  # ask volume for buy
        lambda: best_bid_ask[0][1],  # bid volume for sell
    ).astype(jnp.int32)

    # Cap order size at available volume
    quantity = jnp.minimum(jnp.int32(order_volume), avail)

    # Use time from last message + small increment
    time_s = last_msg_decoded[TIMEs_i].astype(jnp.int32)
    time_ns = (last_msg_decoded[TIMEns_i] + 1).astype(jnp.int32)

    # Build simulator message
    sim_msg = inference.construct_sim_msg(
        event_type,
        direction,
        quantity,
        price,
        AGGRESSIVE_ORDER_ID,
        time_s,
        time_ns,
    )

    # Build decoded message for storage (14 fields)
    mid_price = (sim.get_best_ask(sim_state) + sim.get_best_bid(sim_state)) // 2
    mid_price = (mid_price // tick_size) * tick_size
    rel_price = (price - mid_price) // tick_size

    msg_decoded = jnp.array([
        AGGRESSIVE_ORDER_ID,  # order_id
        event_type,           # event_type
        direction,            # direction
        price,                # price_abs
        rel_price,            # price (relative)
        quantity,             # size
        0,                    # delta_t_s
        1,                    # delta_t_ns
        time_s,               # time_s
        time_ns,              # time_ns
        0,                    # price_ref
        0,                    # size_ref
        0,                    # time_s_ref
        0,                    # time_ns_ref
    ], dtype=jnp.int32)

    return sim_msg, msg_decoded


def roll_msg_through_hidden(
    hidden: Tuple,
    msg_decoded: jax.Array,
    book_state: jax.Array,
    train_state,
    model,
    batchnorm: bool,
    encoder: Dict,
) -> Tuple:
    """
    Roll a message through hidden state so model "sees" it.

    This is crucial for maintaining hidden state continuity after
    inserting an aggressive order.
    """
    # Encode message to tokens
    msg_tokens = encoding.encode_msg(msg_decoded, encoder)

    # Roll through hidden state
    # Note: We pass the full token sequence at once (like conditioning roll)
    hidden, _ = valh.apply_model(
        hidden,
        msg_tokens,
        book_state,
        train_state,
        model,
        batchnorm,
        True  # inference mode
    )

    return hidden


# Create batched version of create_aggressive_order
create_aggressive_order_batched = jax.jit(
    jax.vmap(
        create_aggressive_order,
        in_axes=(None, 0, 0, None, None, None, None)
    ),
    static_argnums=(0,)
)

# Create batched version of roll_msg_through_hidden
roll_msg_through_hidden_batched = jax.jit(
    jax.vmap(
        roll_msg_through_hidden,
        in_axes=(0, 0, 0, None, None, None, None)
    ),
    static_argnums=(3, 4, 5)
)


# NOTE: Using inference.generate_batched directly (the Mamba3-grafted version).
# This version includes insertion_schedule support for single-call generation.


def sample_aggressive_scenario(
    cfg: Dict[str, Any],
    save_folder: Path,
):
    """
    Main function for aggressive scenario generation.

    Flow:
    1. Load data and model
    2. Initialize with conditioning
    3. For each insertion: generate + insert aggressive order
    4. For each cooling: generate only
    5. Save results
    """
    # Unpack config
    n_gen_msgs = cfg['n_gen_msgs']
    num_insertions = cfg['num_insertions']
    num_coolings = cfg['num_coolings']
    n_cond_msgs = cfg['n_cond_msgs']
    n_samples = cfg['n_samples']
    batch_size = cfg['batch_size']
    rng_seed = cfg['rng_seed']
    stock = cfg['stock']
    data_dir = cfg['data_dir']
    ckpt_path = cfg['ckpt_path']
    tick_size = cfg['tick_size']
    sample_top_n = cfg['sample_top_n']
    n_vol_series = cfg['n_vol_series']
    book_dim = cfg['book_dim']
    test_split = cfg['test_split']
    event_type = cfg['event_type']
    direction = cfg['direction']
    n_eval_msgs_dataset = cfg.get('n_eval_msgs_dataset', 500)  # for dataset loading (match run_inference.py)
    order_volume = cfg['order_volume']
    checkpoint_step = cfg.get('checkpoint_step', None)  # None = latest checkpoint
    chunk_size = cfg.get('chunk_size', 1)

    # Derived parameters
    # Total messages to generate including aggressive orders
    total_gen_msgs = (num_insertions + num_coolings) * n_gen_msgs
    n_msg_todo_total = total_gen_msgs + num_insertions  # +insertions for aggressive orders
    cond_seq_len = n_cond_msgs * Message_Tokenizer.MSG_LEN

    # Initialize
    v = Vocab()
    n_classes = len(v)
    rng = jax.random.key(rng_seed)

    # Load model (Mamba3)
    print(f"Loading model from {ckpt_path}")
    args = load_metadata(ckpt_path)
    args.num_devices = 1
    args.bsz = 1
    args.micro_bsz = 1
    args.opt_config = "standard"
    # Single-GPU inference: collapse any tensor-parallel sharding from training. Long-context ckpts
    # (e.g. j4163888) were trained with tp_size=4 -> nh_local = n_heads//4 = 8, but the hidden-carry
    # recipe builds rope-angle state with the FULL n_heads (32). Without this the conditioning roll dies
    # with "add got incompatible shapes (32,32) vs (8,32)". Mamba3 weights are tp-interchangeable.
    args.tp_size = 1

    # Install legacy-norm shim for mamba3 BEFORE building the model.
    from mamba3_legacy_norm import maybe_install_mamba3_legacy_norm
    maybe_install_mamba3_legacy_norm('mamba3')

    new_train_state, model_cls = init_train_state(
        args,
        n_classes=n_classes,
        seq_len=cond_seq_len,
        book_dim=503,
        book_seq_len=n_cond_msgs,
    )

    ckpt = load_checkpoint(
        new_train_state, ckpt_path, step=checkpoint_step,
        train=False, partial_restore=True,
    )
    print(f"Loaded checkpoint step: {ckpt.get('step', checkpoint_step)}")
    train_state = ckpt['model']
    model = model_cls(training=False, step_rescale=1.0)

    # Per-day mode: restrict to specific day
    day_index = cfg.get('day_index', None)
    day_indeces = [day_index] if day_index is not None else None

    # Load dataset
    print(f"Loading dataset from {data_dir} (day_indeces={day_indeces})")
    import glob as _gg, os as _oo
    _root = _oo.path.dirname(str(data_dir))
    print(f"[DIAG] main-proc: data_dir={data_dir!r} isdir={_oo.path.isdir(str(data_dir))} "
          f"msg_glob={len(_gg.glob(str(data_dir)+'/*message*.npy'))} | "
          f"PARENT={_root} parent_isdir={_oo.path.isdir(_root)} "
          f"parent_ls={_oo.listdir(_root)[:6] if _oo.path.isdir(_root) else 'NOPARENT'} "
          f"TMPDIR={_oo.environ.get('TMPDIR')}", flush=True)
    ds = inference.get_dataset(
        data_dir,
        n_cond_msgs,
        n_eval_msgs_dataset,  # use fixed value to match run_inference.py sample selection
        test_split=test_split,
        day_indeces=day_indeces,
    )
    print(f"Dataset length: {len(ds)}")

    # Create output folders (same structure as run_inference.py)
    (save_folder / 'data_cond').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_real').mkdir(exist_ok=True, parents=True)
    (save_folder / 'data_gen').mkdir(exist_ok=True, parents=True)

    # Long-context conditioning (n_cond=4000) yields FEW windows on short days (EA ~20). To reach the
    # requested n_samples/day we sample WITH REPLACEMENT when n_samples exceeds the window count — each
    # reused window is generated with a different RNG so the neural model produces a DISTINCT sample.
    assert len(ds) >= batch_size, f'dataset too small ({len(ds)} windows) for batch_size {batch_size}'

    # Sample indices
    assert n_samples % batch_size == 0, f'n_samples ({n_samples}) must be divisible by batch_size ({batch_size})'
    _replace = n_samples > len(ds)
    if _replace:
        print(f"  [replace] n_samples {n_samples} > {len(ds)} windows -> sampling WITH replacement (distinct gen RNG)")
    # NOTE: Need TWO splits to match run_inference.py behavior:
    # run_inference.py splits once in main script (line 125), then sample_new splits again (line 1197)
    rng, _ = jax.random.split(rng)  # First split (matches run_inference.py main script)
    rng, rng_ = jax.random.split(rng)  # Second split (matches sample_new)
    sample_i = jax.random.choice(
        rng_,
        jnp.arange(len(ds), dtype=jnp.int32),
        shape=(n_samples // batch_size, batch_size),
        replace=_replace
    ).tolist()

    # SAMPLE_SLICE: run only one batch (slice) of the deterministic partition, so N jobs fan out
    # across GPUs and merge into one consolidated folder (slices are disjoint -> distinct real_ids).
    slice_k = cfg.get('sample_slice', None)
    if slice_k is not None:
        slice_k = int(slice_k)
        assert 0 <= slice_k < len(sample_i), f'sample_slice {slice_k} out of range 0..{len(sample_i)-1}'
        batches = [(slice_k, sample_i[slice_k])]
        print(f'>>> SAMPLE_SLICE {slice_k}/{len(sample_i)} -> batch of {batch_size} samples')
    else:
        batches = list(enumerate(sample_i))

    # Initialize hidden state template (Mamba3 recipe).
    # Mirrors sample_new (inference_no_errcorr.py) ssm_type=='mamba3' branch.
    m3_n_heads = (args.mamba3_expand * args.d_model) // args.mamba3_headdim
    init_hidden = model.initialize_carry(
        1,
        hidden_size=0,
        ssm_type='mamba3',
        n_message_layers=args.n_message_layers,
        n_book_pre_layers=args.n_book_pre_layers,
        n_book_post_layers=args.n_book_post_layers,
        n_fused_layers=args.n_layers,
        h_size_ema=args.d_model,
        n_heads=m3_n_heads,
        headdim=args.mamba3_headdim,
        d_state=args.mamba3_d_state,
        num_rope_angles=int(args.mamba3_d_state * args.mamba3_rope_fraction) // 2,
        d_book=503,
    )

    # Replicate for batch
    init_hidden_batched = jax.tree_util.tree_map(
        lambda x: jnp.resize(x, (batch_size,) + x.shape),
        init_hidden
    )

    # Initialize simulator
    sim = OrderBook(cfg=JAXLOB_Configuration(cancel_mode=cst.CancelMode.CANCEL_UNIFORM_AND_LARGE.value))

    # Transform function for book states (use GPU version)
    transform_L2_state_batch = jax.jit(
        jax.vmap(preproc.transform_L2_state_gpu, in_axes=(0, None, None)),
        static_argnums=(1, 2)
    )

    # AOT compile generate function (once)
    print("Compiling generate function (this happens once)...")
    generate_compiled = None

    # Split RNG before loop (matches sample_new line 1207)
    rng, rng_ = jax.random.split(rng)

    # Process batches
    for batch_idx, batch_i in tqdm(batches, desc="Batches"):
        print(f'\n=== BATCH {batch_idx}: samples {batch_i} ===')

        # Load data and put on GPU
        gpu_device = jax.devices('gpu')[0]
        m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[batch_i]
        m_seq = jax.device_put(jnp.array(m_seq), gpu_device)
        b_seq_pv = jax.device_put(jnp.array(b_seq_pv), gpu_device)
        msg_seq_raw = jax.device_put(jnp.array(msg_seq_raw), gpu_device)
        book_l2_init = jax.device_put(jnp.array(book_l2_init), gpu_device)

        # Transform book to volume representation
        b_seq = transform_L2_state_batch(b_seq_pv, n_vol_series, tick_size)
        init_time_batched = b_seq_pv[:, 0, 1:3]

        # Split conditioning and evaluation
        m_seq_cond = m_seq[:, :cond_seq_len + 1]
        b_seq_cond = b_seq[:, :n_cond_msgs + 1]
        m_seq_raw_cond = msg_seq_raw[:, :n_cond_msgs]
        b_seq_pv_cond = onp.array(b_seq_pv[:, :n_cond_msgs + 1, 3:])

        # Initialize simulators
        sim_states = inference.get_sims_vmap(
            book_l2_init,
            m_seq_raw_cond,
            init_time_batched,
            sim,
        )

        # === CREATE INSERTION SCHEDULE ===
        # Format: [flag, event_type, direction, volume, 0, 0, 0, 0, 0]
        # flag=1 means insert aggressive order after this message
        insertion_schedule = jnp.zeros((n_msg_todo_total, 9), dtype=jnp.int32)

        # Calculate insertion positions:
        # Aggressive orders are inserted after every n_gen_msgs messages during the first num_insertions blocks
        # Positions: after message 49, 100, 151, 202, 253 (0-indexed), accounting for cumulative offset
        offset = 0
        for i in range(num_insertions):
            # Insert after (i+1)*n_gen_msgs messages, adjusted by cumulative offset from previous insertions
            insert_step = (i + 1) * n_gen_msgs + offset - 1
            insertion_schedule = insertion_schedule.at[insert_step, 0].set(1)  # flag
            insertion_schedule = insertion_schedule.at[insert_step, 1].set(event_type)
            insertion_schedule = insertion_schedule.at[insert_step, 2].set(direction)
            insertion_schedule = insertion_schedule.at[insert_step, 3].set(order_volume)
            offset += 1  # Each insertion adds to the cumulative offset

        # Log insertion positions
        insertion_positions = jnp.where(insertion_schedule[:, 0] == 1)[0]
        print(f"  Insertion positions (0-indexed): {insertion_positions.tolist()}")

        # Replicate for batch
        insertion_schedule_batched = jnp.tile(insertion_schedule[None, :, :], (batch_size, 1, 1))

        # === SINGLE-CALL GENERATION ===
        # Use rng_ from pre-loop split (or from previous iteration's post-generation split)
        rng_batch = jax.random.split(rng_, batch_size)

        # AOT compile on first call
        if generate_compiled is None:
            print("  AOT compiling generate function...")
            # Trace with all arguments (including insertion_schedule).
            # NOTE: the Mamba3-grafted generate adds valid_mask_array (pos 17,
            # non-static) ahead of insertion_schedule (pos 18, batched) and
            # chunk_size (pos 19, static).
            generate_traced = inference.generate_batched.trace(
                sim,                        # static (0)
                train_state,                # non-static
                model,                      # static (2)
                args.batchnorm,             # static (3)
                v.ENCODING,                 # non-static
                sample_top_n,               # static (5)
                tick_size,                  # static (6)
                m_seq_cond,                 # non-static, batched
                b_seq_cond,                 # non-static, batched
                n_msg_todo_total,           # static (9) - FULL COUNT
                sim_states,                 # non-static, batched
                rng_batch,                  # non-static, batched
                init_hidden_batched,        # non-static, batched
                True,                       # static (13) - conditional
                init_time_batched,          # non-static, batched
                False,                      # static (15) - debug_book
                None,                       # non-static - b_seq_real (for debug)
                None,                       # non-static - valid_mask_array (auto)
                insertion_schedule_batched, # non-static, batched
                chunk_size,                 # static (19) - chunk_size for conditioning
            )
            generate_lowered = generate_traced.lower()
            generate_compiled = generate_lowered.compile()

        # Generate ALL messages in single call.
        # Note: static args (chunk_size) are baked into the compiled function,
        # not passed here.
        all_msgs, all_books, num_errors, msgs_tokens = generate_compiled(
            train_state,
            v.ENCODING,
            m_seq_cond,
            b_seq_cond,
            sim_states,
            rng_batch,
            init_hidden_batched,
            init_time_batched,
            None,  # b_seq_real (debug)
            None,  # valid_mask_array (auto)
            insertion_schedule_batched,
        )

        # Filter out zero-filled placeholder rows (aggressive order placeholders where no insertion happened)
        # A row is valid if ORDER_ID != 0 OR EVENT_TYPE != 0
        # Output: 2N slots (interleaved regular + aggressive), filter zeros
        # all_msgs shape: (batch, n_msg_todo*2, 14)
        # all_books shape: (batch, n_msg_todo*2, book_dim)
        # Filter out zero-filled placeholder rows (aggressive slots where no insertion happened)
        valid_mask = (all_msgs[:, :, ORDER_ID_i] != 0) | (all_msgs[:, :, EVENT_TYPE_i] != 0)
        n_valid_per_batch = valid_mask.sum(axis=1)
        print(f"  Generated {all_msgs.shape[1]} raw slots, {n_valid_per_batch[0]} valid messages, errors: {num_errors.sum()}")

        # === SAVE RESULTS ===
        # Save for each sample in batch (same format as run_inference.py)
        for i, sample_idx in enumerate(batch_i):
            date = ds.get_date(sample_idx)

            # Conditioning data
            inference.msg_to_lobster_format(m_seq_raw_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_message_real_id_{sample_idx}.csv',
                index=False, header=False
            )
            inference.book_to_lobster_format(b_seq_pv_cond[i]).to_csv(
                save_folder / 'data_cond' / f'{stock}_{date}_orderbook_real_id_{sample_idx}.csv',
                index=False, header=False
            )

            # Filter valid messages and books for this batch item
            valid_msgs_i = all_msgs[i][valid_mask[i]]
            valid_books_i = all_books[i][valid_mask[i]]

            # Generated data (filtered - messages aligned 1:1 with book states)
            inference.msg_to_lobster_format(valid_msgs_i).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_message_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )
            inference.book_to_lobster_format(valid_books_i).to_csv(
                save_folder / 'data_gen' / f'{stock}_{date}_orderbook_real_id_{sample_idx}_gen_id_0.csv',
                index=False, header=False
            )

            # For now, no "real" data in aggressive scenario - we're generating counterfactual
            # Save empty placeholder or skip

        # Split RNG for next iteration (matches sample_new line 1382)
        rng, rng_ = jax.random.split(rng)

        # Save aggressive indices. PER-DAY file (aggressive_indices_<date>.csv): in per-day mode each
        # day has its own insertion positions, so one shared file is wrong for all but one day (the
        # old `if not exists` guard kept only the FIRST day). Keep the shared file for back-compat.
        original_indices = onp.arange(all_msgs.shape[1])
        is_aggressive = (original_indices % 2 == 1)[valid_mask[0]]
        aggressive_indices = onp.where(is_aggressive)[0]
        onp.savetxt(save_folder / f'aggressive_indices_{date}.csv', aggressive_indices, fmt='%d')
        if not (save_folder / 'aggressive_indices.csv').exists():
            onp.savetxt(save_folder / 'aggressive_indices.csv', aggressive_indices, fmt='%d')

    print(f"\nResults saved to: {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggressive Scenario with Continuous Hidden State (Mamba3)")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='lob_impact/3_scenarios/1.aggressive_scenario_config.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument('--n_gen_msgs', type=int, default=None, help='Override n_gen_msgs from config')
    parser.add_argument('--direction', type=int, default=None, choices=[0, 1], help='Override direction (0=buy, 1=sell)')
    return parser.parse_args()


def main():
    # Print initial info before logging is set up
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")

    args = parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.n_gen_msgs is not None:
        cfg['n_gen_msgs'] = args.n_gen_msgs
    if args.direction is not None:
        cfg['direction'] = args.direction

    print(f"Configuration: {cfg}")

    # Experiment folder. In slice mode all slices share ONE exact folder (so they merge); otherwise
    # a fresh exp_N is created. The consolidated launcher pre-clears the path on non-slice runs.
    if cfg.get('sample_slice', None) is not None or cfg.get('exact_save_dir', False):
        save_folder = Path(cfg['save_dir']); save_folder.mkdir(parents=True, exist_ok=True)
    else:
        save_folder = create_experiment_folder(cfg['save_dir'])
    print(f"Experiment folder: {save_folder}")

    # Set up logging (per-slice log so concurrent slices don't clobber each other)
    _logname = f"experiment_slice{cfg['sample_slice']}.log" if cfg.get('sample_slice', None) is not None else 'experiment.log'
    logger = setup_logging(save_folder, log_name=_logname)
    print(f"\n{'='*60}")
    print(f"Aggressive Scenario Experiment (Mamba3)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Configuration: {cfg}")
    print(f"Experiment folder: {save_folder}")

    # Save config to experiment folder
    with open(save_folder / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    try:
        # Per-day mode: loop over days from per_day_params CSV
        if cfg.get('per_day_params'):
            import pandas as pd
            per_day_csv = cfg['per_day_params']
            print(f"\nPer-day mode: loading {per_day_csv}")
            pd_df = pd.read_csv(per_day_csv)
            mult_target = cfg.get('order_volume_mult', 1.0)
            pd_df = pd_df[pd_df['mult'] == mult_target].reset_index(drop=True)
            print(f"  {len(pd_df)} days to process (mult={mult_target})")

            bsz = cfg['batch_size']
            n_samples_per_day = cfg.get('n_samples_per_day', max(bsz, cfg['n_samples'] // len(pd_df)))
            n_samples_per_day = max((n_samples_per_day // bsz) * bsz, bsz)
            print(f"  n_samples_per_day = {n_samples_per_day} (batch_size={bsz})")

            for day_idx, row in pd_df.iterrows():
                cfg_d = dict(cfg)
                cfg_d['order_volume'] = int(row['child'])
                cfg_d['n_gen_msgs'] = int(row['mb'])
                cfg_d['day_index'] = int(day_idx)
                cfg_d['n_samples'] = n_samples_per_day
                cfg_d.pop('per_day_params', None)
                print(f"\n--- Day {day_idx}: {row['day']}, child={row['child']}, mb={row['mb']} ---")
                sample_aggressive_scenario(cfg_d, save_folder)
        else:
            sample_aggressive_scenario(cfg, save_folder)

        print(f"\n{'='*60}")
        print(f"Experiment completed!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {save_folder}")
        print(f"{'='*60}")
    finally:
        # Restore stdout/stderr and close log file
        sys.stdout = logger.terminal
        sys.stderr = logger.terminal
        logger.close()
        print(f"Log saved to: {save_folder / 'experiment.log'}")


if __name__ == "__main__":
    main()
