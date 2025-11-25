echo "#!/bin/bash

#SBATCH -A maui_sft
#SBATCH -q h200_maui_sft_high

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00

#SBATCH --job-name=cifar-$1-$2
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out


# Environment setup
eval \"\$(micromamba shell hook --shell bash)\"
micromamba activate instruct_sft

cd \$SLURM_SUBMIT_DIR

time srun python3 train_fcn_image_sgd_dir_sharp.py \
	--loss_name $1 \
	--lr_peak $2 \
	--batch_size $3 \
	--warmup_steps 0 \
	--stable_steps 10_000 \
	--num_steps 10_000 \
	--momentum 0.0 " > submit_job.sh
