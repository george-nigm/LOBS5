python3 run_train.py --USE_WANDB=True \
                    --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
                    --blocks=16 --bsz=32 --d_model=1024 --dataset=lobster-prediction --merging=padded \
                    --dir_name='/home/myuser/processed_data/GOOG/2022' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=30 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.00032 --ssm_size_base=1024 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True  \
                    --masking=none \
                    --num_devices=8 --n_data_workers=8 \
                    --debug_loading=False \
                    --enable_profiler=False \
                    --random_offsets_train=True \
                    --shuffle_train=True \
                    --debug_overfit=False \
                    --lr_patience=3 \
                    --restore='/home/myuser/checkpoints/robust-darkness-68_ci5f3jww' \
                    --restore_step=10
                    #--restore='checkpoints/eager-shadow-750_af39bb9u/'
                    #5135
                    # --curtail_epochs=5135 \


