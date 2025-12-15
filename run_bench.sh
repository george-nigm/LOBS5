#!/bin/bash

# Quick benchmark script for checkpoint evaluation

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lob

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export JAX_PLATFORMS=cuda

CHECKPOINT_PATH="/lus/lfs1aip2/home/s5e/kangli.s5e/AlphaTrade/LOBS5/checkpoints/brisk-violet-111_x514rgnq"
TEST_DATA="/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc"

echo "========================================="
echo "Running LOB Benchmark"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Test Data: $TEST_DATA"
echo "========================================="

python run_eval.py \
  --restore=$CHECKPOINT_PATH \
  --restore_step=2 \
  --epochs=1 \
  --dir_name=$TEST_DATA \
  --bsz=8 \
  --num_devices=1 \
  --n_data_workers=4 \
  --USE_WANDB=True \
  --wandb_project=LOBS5-Bench \
  --wandb_entity=kang-oxford \
  --curtail_epoch=500 \
  --ignore_times=True

echo "========================================="
echo "Benchmark completed"
echo "========================================="
