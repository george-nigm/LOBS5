python run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=8 --bsz=32 --d_model=128 --dataset=lobster-prediction \
                    --dir_name='./data/GOOG' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=6 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=256 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=8 --n_data_workers=4 \
                    --restore='/nfs/home/peern/LOBS5/checkpoints/spring-armadillo-756_4g99ra9s/'
