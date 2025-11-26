echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=32:59:00
#SBATCH --job-name=ft-$1-$2-$4-$6
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
###SBATCH --gpus=h100:1

# Environment setup
source /home/dayal/miniconda3/bin/activate
conda activate torch

cd \$SLURM_SUBMIT_DIR

time srun python3 finetune_gpt_adam_forward_ckpts.py  \
	--dataset_name alpaca  \
	--ckpt_load_path fineweb_v50304_gpt_var1.0_d12_h12_n768_c1024_AdamW_Tw$1_r1.0_Ts$2_polynomial_p1.0_T$3_ga128_lr$4_lrinf_wd0.1_bs16_b0.9_b0.95_eps1e-08_gc0.0.ckpt \
     	--batch_size 16  \
     	--gradient_accumulation_steps $5  \
     	--lr_peak $6  \
   	--lr_min_factor inf  \
   	--weight_decay $7  \
   	--beta1 0.9  \
	--beta2 0.95 \
   	--warmup_steps $8  \
    	--stable_steps $9  \
   	--decay_schedule_name polynomial  \
     	--num_steps ${10}  \
     	--eval_interval 100 \
	--ckpt_interval 200" > submit_job.sh
