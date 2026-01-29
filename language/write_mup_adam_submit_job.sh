# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=32:59:00
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

time srun python3 train_gpt_mup_adam.py \
    --dataset_name fineweb \
    --num_layers 12 \
    --num_heads 12 \
    --init_var 1.0 \
    --use_flash True \
    --compile True \
    --batch_size 32 \
    --gradient_accumulation_steps 40 \
    --lr_peak $1 \
    --lr_min_factor inf \
    --weight_decay 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup_steps 2_000 \
    --num_steps 10_000 \
    --eval_interval 200" > submit_job.sh
