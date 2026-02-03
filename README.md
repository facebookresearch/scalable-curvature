# Scalable Curvature

[![arXiv](https://img.shields.io/badge/arXiv-2601.16979-b31b1b.svg)](https://arxiv.org/abs/2601.16979)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE.md)

Code for [A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs](https://arxiv.org/abs/2601.16979).

Dayal Singh Kalra, Jean-Christophe Gagnon-Audet, Andrey Gromov, Ishita Mediratta, Kelvin Niu, Alexander H Miller, Michael Shvartsman

## Overview

We analyze **critical sharpness** ($\lambda_c$), a computationally efficient measure of loss landscape curvature requiring fewer than 10 forward passes given the update direction. Critical sharpness is defined as $\lambda_c = 2/\eta_c$, where $\eta_c$ is the **critical learning rate** — the smallest learning rate that causes the training loss to increase in the next training step.

**Why critical sharpness?**

Despite its significance in analyzing training dynamics, direct measurement of Hessian sharpness (the largest eigenvalue of the loss Hessian) remains prohibitive for Large Language Models (LLMs) due to high computational cost. Critical sharpness provides a scalable alternative that:

- Requires only **5-6 forward passes** per measurement, given the update direction
- Captures well-documented Hessian sharpness phenomena, including **progressive sharpening** and **Edge of Stability**
- Works with modern distributed training (DDP/FSDP), as it relies solely on forward passes, avoiding the challenges of Hessian-based methods

This repository demonstrates how to measure critical sharpness using an adapted version of [nanoGPT](https://github.com/karpathy/nanoGPT).

## Installation

```bash
conda create -n curvature python=3.10
conda activate curvature

pip install torch numpy pandas scipy wandb tqdm transformers tiktoken datasets
```

## Data

**FineWeb**: Tokenize FineWeb-Edu into `data/fineweb/`. The `prepare.py` script follows the [nanoGPT](https://github.com/karpathy/nanoGPT) pipeline exactly.

```bash
cd data/fineweb
python prepare.py
```

## Quick Start

Run the demo training script (adapted from [nanoGPT](https://github.com/karpathy/nanoGPT)):

```bash
python demo_nanogpt.py
```

This will train a GPT model while logging critical sharpness metrics.

## Integration Guide

**Important:** This critical sharpness measurement function requires custom optimizers from `utils.optimizers` that implement a `_compute_update()` method for taking a virtual step for line search . Standard PyTorch optimizers (`torch.optim.AdamW`, `torch.optim.SGD`) are not supported.

### Why Custom Optimizers?

The measurement function needs to compute "virtual" parameter updates without actually modifying the model weights or optimizer state. We do this by implementing a `_compute_update()` method that yield the updates $\Delta \theta$ without updating the optimizer states or the model parameters.

### Step-by-Step Integration

#### 1. Import Required Components

```python
from utils.measure_critical_lr import measure_critical_lr
from utils.optimizers import AdamW  # torch.optim.AdamW not supported
```

#### 2. Initialize Custom Optimizer

```python
# Use custom optimizer instead of torch.optim
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.01
)
```

#### 3. Define Loss Closure

The measurement function requires a **loss closure** that takes a batch and returns a scalar loss. This closure should include:
- Forward pass of the model
- Loss computation
- Any distributed training synchronization (e.g., DDP/FSDP all-reduce operations)

```python
def loss_closure(batch):
    """Compute loss for a given batch.
    
    This closure is called multiple times during the search process.
    It should include all forward pass operations and distributed 
    training synchronization needed to compute the loss.
    """
    inputs, targets = batch
    
    # Forward pass
    logits = model(inputs)
    
    # Compute loss
    loss = loss_fn(logits, targets)

    # DDP/FSDP all reduce if required:
    dist.all_reduce(l_loss, op=dist.ReduceOp.AVG)
    
    return loss.item()  # Return scalar, not tensor
```

#### 4. Integrate into Training Loop

Place the measurement call **after** `loss.backward()` but **before** `optimizer.step()`, this utilizes the gradient computations from the training run.

```python
# Initialize lr_guess before training loop
current_lr_guess = 1e-3

for batch in dataloader:
    inputs, targets = batch
    
    # 1. Forward pass and backward pass
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    
    # 2. Measure critical sharpness (reuses gradients from backward pass above)
    lr_lower, lr_upper, n_iters = measure_critical_lr(
        model=model,
        optimizer=optimizer,
        batch=(inputs, targets),
        loss_closure=loss_closure,
        lr_guess=current_lr_guess,  # Use updated guess from previous iteration
        tol_power=4
    )
    
    # The critical sharpness is typically logged as the average
    critical_lr = (lr_lower + lr_upper) / 2.0
    critical_sharpness = 2.0 / critical_lr
    
    # Update lr_guess for next iteration (reduces exponential search steps)
    current_lr_guess = critical_lr

    # 3. Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

### Function Reference

```python
lr_lower, lr_upper, n_iters = measure_critical_lr(
    model,           # nn.Module
    optimizer,       # Custom optimizer (AdamW or SGD from utils.optimizers)
    batch,           # Batch data (e.g., tuple of (inputs, targets))
    loss_closure,    # Callable that takes batch and returns scalar loss
    lr_guess=1e-3,   # Initial guess for exponential search (update with previous critical_lr)
    tol_power=4,     # Search precision: tolerance = 1 / 2^tol_power
    logger=None      # Optional logger for debug output
)
```

**Returns:**
- `lr_lower` (float): Lower bound of critical learning rate
- `lr_upper` (float): Upper bound of critical learning rate  
- `n_iters` (int): Number of forward passes used in the search

**Critical sharpness** is computed as the average: `(lr_lower + lr_upper) / 2.0`

**Note:** For best efficiency, update `lr_guess` with the previous iteration's critical LR to minimize exponential search steps. This reduces the number of iterations to six --- two exponential search steps and four binary search steps

### How It Works

The measurement uses a two-phase search algorithm:

1. **Exponential Search**: Starting from `lr_guess`, exponentially increase or decrease the learning rate until we find a range $(\eta_{lower}, \eta_{upper})$ containing the critical learning rate.

2. **Binary Search**: Refine the range using a binary search until the relative gap is less than `1 / 2^tol_power` (default: 1/16 = 6.25%).

Each iteration involves:
- Computing virtual parameter updates at a candidate learning rate (without modifying the model or the optimizer states)
- Temporarily applying these updates
- Evaluating the loss
- Restoring the original parameters

## Code Structure

```text
scalable-curvature/
├── demo_nanogpt.py           # Training demo (adapted from nanoGPT by Andrej Karpathy)
├── data/
│   └── fineweb/              # Data preparation
│       ├── prepare.py
│       ├── train.bin
│       └── val.bin
└── utils/
    ├── measure_critical_lr.py   # Critical sharpness measurement (exponential + binary search)
    ├── optimizers.py            # Custom optimizers with virtual_step support (REQUIRED)
    ├── gpt_sp.py                # GPT model definition
    ├── schedules_utils.py       # Learning rate schedules
    ├── data_loader.py           # Data loading utilities
    ├── data_storage.py          # Logging utilities
    └── loss_functions.py        # Loss function definitions
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

## Acknowledgements

The GPT pre-training demo (`demo_nanogpt.py`) and data pipeline in this repository are adapted from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.