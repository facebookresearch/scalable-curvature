echo "#!/bin/bash
#SBATCH -A maui_sft
#SBATCH -q h200_maui_sft_high

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=90:00:00

#SBATCH --job-name=gpt-$1
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out

# Environment setup
eval \"\$(micromamba shell hook --shell bash)\"
micromamba activate instruct_sft

cd \$SLURM_SUBMIT_DIR

time srun python3 train_gpt_adam_dir_sharp.py  \
	--dataset_name fineweb  \
     	--num_layers 12  \
      	--num_heads 12  \
     	--init_var 1.0  \
     	--batch_size 16  \
     	--gradient_accumulation_steps 64  \
     	--lr_peak $1  \
   	--lr_min_factor inf  \
	--grad_clip $2 \
   	--weight_decay $3  \
   	--beta1 0.9  \
	--beta2 0.95 \
   	--warmup_steps $4  \
    	--stable_steps $5  \
   	--decay_schedule_name cosine  \
	--decay_exponent $7 \
     	--num_steps $6  \
     	--eval_interval 200 \
	--sharpness_interval 10 " > submit_job.sh
