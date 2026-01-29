# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo "#!/bin/bash

#SBATCH -A MY_ACCT
#SBATCH -q MY_QOS

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00

#SBATCH --job-name=cifar-$1-$2
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out


# Environment setup
conda activate curvature

cd \$SLURM_SUBMIT_DIR

time srun python3 train_fcn_image_sgd_dir_sharp.py \
	--loss_name $1 \
	--lr_peak $2 \
	--batch_size $3 \
	--warmup_steps 0 \
	--stable_steps 10_000 \
	--num_steps 10_000 \
	--momentum 0.0 " > submit_job.sh
