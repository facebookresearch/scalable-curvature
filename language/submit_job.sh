#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH -A MY_ACCT
#SBATCH -q MY_QOS

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=90:00:00

#SBATCH --job-name=gpt-3e-02
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out

# Environment setup
conda activate curvature

cd $SLURM_SUBMIT_DIR

time srun python3 train_gpt_adam_dir_sharp.py  	--dataset_name fineweb       	--num_layers 12        	--num_heads 12       	--init_var 1.0       	--batch_size 16       	--gradient_accumulation_steps 64       	--lr_peak 3e-02     	--lr_min_factor inf  	--grad_clip 0.0    	--weight_decay 0.1     	--beta1 0.9  	--beta2 0.95    	--warmup_steps 1000      	--stable_steps 8000     	--decay_schedule_name cosine  	--decay_exponent 1.0      	--num_steps 10_000       	--eval_interval 200 	--sharpness_interval 10
