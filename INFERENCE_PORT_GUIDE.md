# Inference Port Guide — Multi-Architecture Support

**Branch**: `inference_port` (based on `george/lob_impact` at `a69413c8`)
**Date**: 2026-03-11
**Status**: Ready for testing

## What Changed

This branch ports shard-map's inference improvements into lob_impact. All changes
are **additive** — existing S5 inference paths are untouched; new features activate
only when the checkpoint metadata contains `ssm_type` or `model_type` fields.

### Summary of Changes

| Area | What | Backward Compatible? |
|------|------|---------------------|
| New model files | `s5/gdn.py`, `s5/transformer.py`, `s5/moe.py`, + 4 more | Yes — unused unless imported |
| `initialize_carry` | Extended with `ssm_type`, `is_transformer` params | Yes — defaults = S5 behavior |
| `init_train_state` | GDN/KDA and Transformer SSM init branches | Yes — falls through to S5 HiPPO |
| `load_checkpoint` | TensorStore direct-read fallback for inference | Yes — only triggers on StandardRestore failure |
| `get_sim_msg` | L1→L2 progressive cancel order-ID resolution | Yes — L1 still tried first |
| `syntax_validation_matrix` | `block_start_tok` param for transformers | Yes — default `False` |
| `generate()` | `valid_mask_array` replaces `chunk_size` param | **Breaking** — see below |
| `sample_new()` | `sample_indices`, `wide_levels` replace `chunk_size` | **Breaking** — see below |
| `get_dataset()` | `.npy` validation + `wide_book_dir` param | Yes — both optional |
| `lobster_dataloader` | `wide_book_files` param for deeper L2 init | Yes — default `None` |
| `run_inference.py` | New CLI args, multi-GPU rank splitting | Yes — all args have defaults |

### Breaking Change: `chunk_size` Removed

The `chunk_size` parameter has been removed from both `generate()` and `sample_new()`.
The conditioning sequence is now **always** chunked at 1 message per step internally
(N = total_tokens / MSG_LEN). This is required for transformer compatibility (avoids
O(L^2) attention OOM) and is harmless for S5.

**If you were passing `chunk_size=N` to `sample_new()`, remove that argument.**
The old `--chunk_size` CLI flag is also removed from `run_inference.py`.

## Usage

### S5 Inference (No Change Required)

Existing S5 checkpoints work exactly as before:

```bash
python run_inference.py --stock GOOG --batch_size 32 --n_sequences 1024
```

### GDN/KDA Inference

Requires a checkpoint trained with `ssm_type='gdn'` or `ssm_type='kda'`. The
checkpoint metadata must contain these fields (set automatically during training):

```bash
python run_inference.py \
    --ckpt_path /path/to/gdn_checkpoint \
    --data_dir /path/to/preprocessed_data \
    --save_dir ./results/gdn_inference \
    --batch_size 16 \
    --n_sequences 512
```

### Transformer Inference

Requires `model_type='transformer'` in checkpoint metadata:

```bash
python run_inference.py \
    --ckpt_path /path/to/transformer_checkpoint \
    --data_dir /path/to/preprocessed_data \
    --save_dir ./results/transformer_inference \
    --batch_size 16
```

### New CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n_gen_msgs` | int | 500 | Messages to generate per sequence |
| `--data_dir` | str | None | Override data directory (bypasses stock lookup) |
| `--ckpt_path` | str | None | Override checkpoint path (bypasses stock lookup) |
| `--save_dir` | str | None | Override output directory |
| `--sample_indices_file` | str | None | `.npy` file with pre-determined sample indices |
| `--wide_book_dir` | str | None | Directory with wider L2 book files for deeper sim init |
| `--wide_levels` | int | 10 | Book levels for simulator init (increase for wide books) |
| `--rank` | int | 0 | GPU rank for multi-GPU inference |
| `--world_size` | int | 1 | Total GPU count for multi-GPU inference |

When `--ckpt_path` is provided, the hardcoded stock paths (GOOG/AMZN/INTC) are
bypassed entirely. You must also provide `--data_dir`.

### Multi-GPU Inference

Split work across N GPUs using interleaved index assignment:

```bash
# GPU 0
python run_inference.py --ckpt_path ... --data_dir ... --rank 0 --world_size 4

# GPU 1
python run_inference.py --ckpt_path ... --data_dir ... --rank 1 --world_size 4

# etc.
```

Each rank gets every Nth sample (interleaved, not contiguous). Results are saved
to the same `--save_dir` with different sample IDs, so they can be merged after.

### Wide Book Support

For deeper simulator L2 initialization (e.g. 100-level books instead of 10):

```bash
python run_inference.py \
    --ckpt_path /path/to/checkpoint \
    --data_dir /path/to/data \
    --wide_book_dir /path/to/L100_books \
    --wide_levels 100
```

The `wide_book_dir` must contain files matching `*{YYYY-MM-DD}*orderbook*proc.npy`.
Dates are matched from the standard book files. The simulator's `nOrders` and
`book_depth` are automatically sized based on `--wide_levels`.

## Architecture Detection

The model architecture is detected from checkpoint metadata:

```
ssm_type = getattr(args, 'ssm_type', 's5')     # 'gdn', 'kda', or 's5'
model_type = getattr(args, 'model_type', 's5')  # 'transformer' or 's5'
```

If neither field exists in the metadata, S5 is assumed. This means **all existing
S5 checkpoints work without any metadata changes**.

## Files Added

| File | Description |
|------|-------------|
| `s5/gdn.py` | Gated Delta Networks / KDA SSM module |
| `s5/gdn_triton_kernels.py` | Fused WY correction Triton kernel (optional) |
| `s5/fla_kernels.py` | FLA Triton kernels |
| `s5/fla_solve_tril.py` | FLA solve_tril JAX wrapper |
| `s5/transformer.py` | Transformer block with KV-cache inference |
| `s5/moe.py` | Mixture of Experts FFN module |
| `lob/encoding_23tok.py` | 23-token encoding variant |

These files are only imported when the corresponding architecture is requested
via `ssm_type` or `model_type`. They have no effect on S5 inference.

## TensorStore Checkpoint Fallback

When loading a checkpoint for inference, if `StandardRestore` fails (common when
the optax/Orbax version differs from training), the loader now falls back to
reading `params` (and `batch_stats` if present) directly via TensorStore. This
handles both OCDBT and zarr3 checkpoint formats.

This only activates for `train=False` (inference mode). Training restores still
raise on mismatch.

## Progressive Cancel Fallback

Cancel order resolution now uses two levels:

1. **L1**: Exact timestamp match via `sim.get_order_at_time()` (original behavior)
2. **L2**: Price-based closest-time match via `find_order_at_price_closest_time()`

L2 only activates when L1 returns `NEGATIVE_RETURN_ID`. This reduces cancel
failures when the simulator's order timestamps don't exactly match the model's
predicted reference timestamps.
