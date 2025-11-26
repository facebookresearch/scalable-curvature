echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=150:59:00
#SBATCH --job-name=fwd-$2-$4
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
###SBATCH --gpus=h100:1

# Environment setup
source /home/dayal/miniconda3/bin/activate
conda activate torch

cd \$SLURM_SUBMIT_DIR

time srun python3 train_gpt_adam_forward_ckpts.py  \
	--dataset_name fineweb  \
     	--num_layers 12  \
      	--num_heads 12  \
     	--init_var 1.0  \
     	--batch_size 16  \
     	--gradient_accumulation_steps $1  \
     	--lr_peak $2  \
   	--lr_min_factor inf  \
   	--weight_decay 0.1  \
   	--beta1 0.9  \
	--beta2 0.95 \
   	--warmup_steps $3  \
    	--stable_steps $4  \
   	--decay_schedule_name polynomial  \
     	--num_steps $5  \
     	--eval_interval 100 \
	--sharpness_interval 10 " > submit_job.sh
