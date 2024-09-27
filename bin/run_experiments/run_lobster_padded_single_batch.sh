python run_train.py --USE_WANDB=True \
                    --C_init=trunc_standard_normal --prenorm=True --batchnorm=False --bidirectional=False \
                    --blocks=16 --bsz=4 --d_model=512 --dataset=lobster-prediction --merging=padded \
                    --dir_name='/data1/sascha/data/lobster_preproced/GOOG2019' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=10000 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True  \
                    --masking=none \
                    --num_devices=1 --n_data_workers=0 \
                    --debug_loading=False \
                    --enable_profiler=False \
                    --curtail_epochs=0 \
                    --random_offsets_train=False --shuffle=False\
                    # --restore='/homes/80/kang/LOBS5/checkpoints/honest-oath-159_3kn3xbd5/' \
                    # --restore_step=63
                    #--restore='checkpoints/eager-shadow-750_af39bb9u/'
                    


