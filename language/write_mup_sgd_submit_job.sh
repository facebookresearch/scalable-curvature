echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:59:00
#SBATCH --job-name=gpt-$1
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
###SBATCH --gpus=h100:1

# Environment setup
source /home/dayal/miniconda3/bin/activate
conda activate torch

cd \$SLURM_SUBMIT_DIR

time srun python3 train_gpt_mup_sgd_func.py \
    --dataset_name fineweb \
    --num_layers 4 \
    --num_heads 4 \
    --init_var 1.0 \
    --batch_size 32 \
    --gradient_accumulation_steps 20 \
    --lr_peak $1 \
    --lr_min_factor inf \
    --weight_decay 0.0 \
    --momentum 0.0 \
    --warmup_steps 100 \
    --num_steps 1000 \
    --eval_interval 200" > submit_job.sh
