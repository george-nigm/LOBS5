python run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=16 --bsz=128 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='./data/INTC' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=8 --n_data_workers=4

# --n_data_workers=2 or 4

# python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
#                     --blocks=16 --bsz=128 --d_model=512 --dataset=lobster-prediction \
#                     --dir_name='/homes/80/kang/lob_bench/GOOG_small' --clip_eigs=True --activation_fn=half_glu1 \
#                     --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
#                     --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
#                     --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
#                     --use_book_data=True --use_simple_book=False --book_transform=True \
#                     --masking=causal \
#                     --num_devices=8 --n_data_workers=4

# python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
#                     --blocks=16 --bsz=128 --d_model=512 --dataset=lobster-prediction \
#                     --dir_name='/homes/80/kang/LOBS5/GOOG_unzip' --clip_eigs=True --activation_fn=half_glu1 \
#                     --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
#                     --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
#                     --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
#                     --use_book_data=True --use_simple_book=False --book_transform=True \
#                     --masking=causal \
#                     --num_devices=8 --n_data_workers=4


python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=16 --bsz=128 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='/homes/80/kang/LOBS5/GOOG' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=8 --n_data_workers=4


python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=16 --bsz=128 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='/homes/80/kang/LOBS5/GOOG' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=8 --n_data_workers=1
        
python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=16 --bsz=128 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='/homes/80/kang/LOBS5/GOOG_tiny' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=4 --n_data_workers=4

python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=16 --bsz=128 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='/homes/80/kang/LOBS5/GOOG_small' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=4 --n_data_workers=1



python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=16 --bsz=128 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='/homes/80/kang/LOBS5/GOOG_small' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=4 --n_data_workers=1



python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=16 --bsz=16 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='/homes/80/kang/LOBS5/GOOG_small' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=1 --n_data_workers=1
                    # --num_devices=4 --n_data_workers=1


python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=16 --bsz=16 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='/homes/80/kang/LOBS5/GOOG_small' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=8 --n_data_workers=8


python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=8 --bsz=128 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='/homes/80/kang/LOBS5/GOOG_tiny' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=8 --n_data_workers=1

                    # --blocks=16 --bsz=128 --d_model=512 --dataset=lobster-prediction \


python3 run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=True \
                    --blocks=8 --bsz=128 --d_model=512 --dataset=lobster-prediction \
                    --dir_name='/homes/80/kang/LOBS5/GOOG_tiny' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=12 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=512 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=causal \
                    --num_devices=4 --n_data_workers=1