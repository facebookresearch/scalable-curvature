# Scalable Curvature

[![arXiv](https://img.shields.io/badge/arXiv-2601.16979-b31b1b.svg)](https://arxiv.org/abs/2601.16979)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE.md)

Code for [A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs](https://arxiv.org/abs/2601.16979).

Dayal Singh Kalra, Jean-Christophe Gagnon-Audet, Andrey Gromov, Ishita Mediratta, Kelvin Niu, Alexander H Miller, Michael Shvartsman

## Overview

We introduce *critical sharpness*, a measure of loss landscape curvature that can be computed using forward passes alone. This makes it tractable for large models where Hessian-based methods are infeasible. The method requires approximately 5-6 forward passes given an update direction.

## Installation

```bash
conda create -n curvature python=3.10
conda activate curvature
pip install torch numpy pandas scipy wandb
pip install cupy-cuda12x  # adjust for your CUDA version -- needed for HVP computation
```

## Data

**CIFAR-10**: Downloaded automatically via torchvision.

**FineWeb**: Follow the [nanoGPT](https://github.com/karpathy/nanoGPT) data preparation instructions to tokenize FineWeb-Edu into `language/data/fineweb/`.

## Reproducing Results

### Figure 2: CIFAR-10 Sharpness Dynamics (SGD)

FCN with 4 layers, width 512, trained with SGD at constant learning rate 3e-02.

```bash
cd image

# Full batch (batch_size=50000)
python train_fcn_image_sgd_dir_sharp.py \
    --batch_size 50000 --lr_peak 3e-02 --num_steps 10000 \
    --warmup_steps 0 --stable_steps 10000 --loss_name xent

# Batch size 5000
python train_fcn_image_sgd_dir_sharp.py \
    --batch_size 5000 --lr_peak 3e-02 --num_steps 10000 \
    --warmup_steps 0 --stable_steps 10000 --loss_name xent

# Batch size 500
python train_fcn_image_sgd_dir_sharp.py \
    --batch_size 500 --lr_peak 3e-02 --num_steps 10000 \
    --warmup_steps 0 --stable_steps 10000 --loss_name xent
```

### Figure 3: GPT Pre-training Sharpness Dynamics (AdamW)

GPT with 12 layers, 768 embedding dim, trained on FineWeb with WSD schedule.

```bash
cd language

python train_gpt_adam_dir_sharp.py \
    --num_layers 12 --num_heads 12 --head_dim 64 \
    --lr_peak 3e-04 --num_steps 10000 \
    --warmup_steps 2000 --stable_steps 6000 \
    --batch_size 16 --gradient_accumulation_steps 64 \
    --weight_decay 0.0
```

## Configuration

Before running on your cluster, update the following:

- **SLURM settings**: Edit `MY_ACCT` and `MY_QOS` in `image/submit_job.sh` and `language/submit_job.sh` (and the corresponding `write_*_submit_job.sh` scripts) to match your cluster's account and QOS.
- **Checkpoint paths**: The `--ckpt_dir` argument defaults to `./checkpoints`. Modify as needed for your storage setup.

## Code Structure

```
scalable-curvature/
├── image/                          # CIFAR-10 experiments
│   ├── train_fcn_image_sgd_dir_sharp.py
│   ├── train_fcn_image_adamw_dir_sharp.py
│   └── utils/
│       ├── critical_lr_cache_utils.py   # Critical LR estimation
│       ├── sharpness_dir_utils.py       # Directional sharpness (HVP-based)
│       ├── sharpness_cupy_utils.py      # Hessian sharpness (LOBPCG)
│       └── optimizers.py                # Custom SGD/AdamW with virtual_step
└── language/                       # GPT experiments
    ├── train_gpt_adam_dir_sharp.py
    └── utils/
        ├── critical_learning_rate.py
        ├── gpt.py
        └── ...
```

## Citation

```bibtex
@article{kalra2026scalable,
  title={A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs},
  author={Kalra, Dayal Singh and Gagnon-Audet, Jean-Christophe and Gromov, Andrey and Mediratta, Ishita and Niu, Kelvin and Miller, Alexander H and Shvartsman, Michael},
  journal={arXiv preprint arXiv:2601.16979},
  year={2026}
}
```
