# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:59:00
#SBATCH --job-name=cifar-$1-$2-$3
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Environment setup
source /home/dayal/miniconda3/bin/activate
conda activate torch

cd \$SLURM_SUBMIT_DIR

time srun python3 train_fcn_image_adamw_dir_sharp.py \
	--loss_name $1 \
	--lr_peak $2 \
	--batch_size $3 \
	--warmup_steps $4 \
	--stable_steps $5 \
	--num_steps $6 \
	--beta1 0.9 \
	--beta2 0.99 " > submit_job.sh
