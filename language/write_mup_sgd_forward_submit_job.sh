echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:59:00
#SBATCH --job-name=crit-$1-$2-$3
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
###SBATCH --gpus=h100:1

# Environment setup
source /home/dayal/miniconda3/bin/activate
conda activate torch

cd \$SLURM_SUBMIT_DIR

time srun python3 train_gpt_mup_sgd_forward.py  \
	--dataset_name fineweb  \
     	--num_layers 4  \
      	--num_heads 4  \
     	--init_var 1.0  \
     	--batch_size 16  \
     	--gradient_accumulation_steps $1  \
     	--lr_peak $2  \
   	--lr_min_factor inf  \
   	--weight_decay 0.0  \
   	--momentum $3  \
   	--warmup_steps 1000  \
    	--stable_steps 8000  \
   	--decay_schedule_name polynomial  \
     	--num_steps 10_000  \
     	--eval_interval 200  \
    	--estimate_critical_metrics True  \
     	--recompute_grads True " > submit_job.sh
