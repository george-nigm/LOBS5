#!/bin/bash
#SBATCH --job-name=lob_autoreg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm_logs/autoreg_%A_%a.out
#SBATCH --error=slurm_logs/autoreg_%A_%a.err

# 128 batches / 16 GPUs = 8 batches per GPU
BATCHES_PER_GPU=8
START_BATCH=$((SLURM_ARRAY_TASK_ID * BATCHES_PER_GPU))
END_BATCH=$((START_BATCH + BATCHES_PER_GPU))

echo "Job array task: $SLURM_ARRAY_TASK_ID"
echo "Processing batches: $START_BATCH to $END_BATCH"
echo "GPU: $CUDA_VISIBLE_DEVICES"

cd /scratch/local/homes/80/georgenigm/LOBS5

# Using Singularity/Apptainer (common on HPC)
# Adjust the container path as needed
singularity exec --nv /path/to/georgenigm_docker.sif \
    conda run -n myenv python -u 1_run_exp_aggressive_scenario_whole_lvl_autoreg_26_01_14.py \
    --config 1_run_exp_aggresive_scenario_autoreg_bs8 \
    --start-batch $START_BATCH \
    --end-batch $END_BATCH

# Alternative: if using modules + conda directly (no container)
# module load cuda/12.0
# module load anaconda3
# conda activate myenv
# python -u 1_run_exp_aggressive_scenario_whole_lvl_autoreg_26_01_14.py \
#     --config 1_run_exp_aggresive_scenario_autoreg_bs8 \
#     --start-batch $START_BATCH \
#     --end-batch $END_BATCH
