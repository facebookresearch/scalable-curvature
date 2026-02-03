# Scalable Curvature

[![arXiv](https://img.shields.io/badge/arXiv-2601.16979-b31b1b.svg)](https://arxiv.org/abs/2601.16979)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE.md)

Code for [A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs](https://arxiv.org/abs/2601.16979).

Dayal Singh Kalra, Jean-Christophe Gagnon-Audet, Andrey Gromov, Ishita Mediratta, Kelvin Niu, Alexander H Miller, Michael Shvartsman

## Overview

We analyze **critical sharpness** (λc), a computationally efficient measure of loss landscape curvature requiring fewer than 10 forward passes given the update direction. Critical sharpness is defined as λc = 2/ηc, where ηc is the **critical learning rate** — the smallest learning rate that causes the training loss to increase in the next training step.

**Why critical sharpness?**

Despite its significance in analyzing training dynamics, direct measurement of Hessian sharpness (the largest eigenvalue of the loss Hessian) remains prohibitive for Large Language Models (LLMs) due to high computational cost. Critical sharpness provides a scalable alternative that:

- Requires only **5-6 forward passes** per measurement (given the update direction)
- Uses **exponential search followed by binary search** to find the critical learning rate
- Captures well-documented Hessian sharpness phenomena, including **progressive sharpening** and **Edge of Stability**
- Works with modern distributed training (DDP/FSDP) — relies solely on forward passes, avoiding the challenges of Hessian-based methods

This repository demonstrates how to measure critical sharpness using an adapted version of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

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

This will train a GPT model while logging critical sharpness metrics to Weights & Biases.

## Integration Guide

**Important:** This measurement function requires custom optimizers from `utils.optimizers` that implement a `_compute_update()` method for virtual stepping. Standard PyTorch optimizers (`torch.optim.AdamW`, `torch.optim.SGD`) will not work.

### Why Custom Optimizers?

The measurement function needs to compute "virtual" parameter updates without actually modifying the model weights or optimizer state. This requires a `_compute_update()` method that our custom optimizers provide but standard PyTorch optimizers do not.

### Step-by-Step Integration

#### 1. Import Required Components

```python
from utils.measure_critical_lr import measure_critical_lr
from utils.optimizers import AdamW  # MUST use this, not torch.optim.AdamW
```

#### 2. Initialize Custom Optimizer

```python
# Use custom optimizer instead of torch.optim
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

#### 3. Define Loss Closure

The measurement function requires a **loss closure** that takes a batch and returns a scalar loss. This closure should include:
- Forward pass through the model
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
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
    
    # For DDP/FSDP: synchronization happens automatically in forward pass
    
    return loss.item()  # Return scalar, not tensor
```

#### 4. Integrate into Training Loop

Place the measurement call **after** `loss.backward()` but **before** `optimizer.step()`:

```python
# Initialize lr_guess before training loop
current_lr_guess = 1e-3

for batch in dataloader:
    inputs, targets = batch
    
    # 1. Forward pass and backward pass
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    
    # Optional: gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 2. Measure critical sharpness (uses gradients from backward pass)
    lr_lower, lr_upper, n_probes = measure_critical_lr(
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
    
    # 3. Log metrics
    wandb.log({
        "loss": loss.item(),
        "critical_lr": critical_lr,
        "critical_sharpness": critical_sharpness,
        "lr_lower": lr_lower,
        "lr_upper": lr_upper,
        "n_probes": n_probes
    })
    
    # 4. Optimizer step
    optimizer.step()
    optimizer.zero_grad()
```

### Function Reference

```python
lr_lower, lr_upper, n_probes = measure_critical_lr(
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
- `n_probes` (int): Number of forward passes used in the search

**Critical sharpness** is typically computed as the average: `(lr_lower + lr_upper) / 2.0`

**Note:** For best efficiency, update `lr_guess` with the previous iteration's critical LR to minimize exponential search steps.

### How It Works

The measurement uses a two-phase search algorithm:

1. **Exponential Search**: Starting from `lr_guess`, exponentially increase or decrease the learning rate until we bracket the critical point (where a single step would increase the loss).

2. **Binary Search**: Refine the bracket using binary search until the relative gap is less than `1 / 2^tol_power` (default: 1/16 = 6.25%).

Each probe involves:
- Computing virtual parameter updates at a candidate learning rate (without modifying the model)
- Temporarily applying these updates
- Evaluating the loss
- Restoring the original parameters

## Complete Working Example

Here's a minimal end-to-end example:

```python
import torch
import torch.nn.functional as F
from utils.measure_critical_lr import measure_critical_lr
from utils.optimizers import AdamW  # Custom optimizer

# Model and data
model = MyModel()
dataloader = MyDataLoader()

# MUST use custom optimizer
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Define loss closure
def loss_closure(batch):
    inputs, targets = batch
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return loss.item()

# Initialize lr_guess
current_lr_guess = 1e-3

# Training loop
for batch in dataloader:
    inputs, targets = batch
    
    # Forward & backward
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    
    # Measure critical sharpness
    lr_lower, lr_upper, n_probes = measure_critical_lr(
        model=model,
        optimizer=optimizer,
        batch=(inputs, targets),
        loss_closure=loss_closure,
        lr_guess=current_lr_guess
    )
    
    critical_lr = (lr_lower + lr_upper) / 2.0
    critical_sharpness = 2.0 / critical_lr
    
    # Update guess for next iteration
    current_lr_guess = critical_lr
    
    print(f"Loss: {loss.item():.4f}, Critical Sharpness: {critical_sharpness:.4f}")
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
```

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

## Distributed Training Notes

The measurement function is compatible with DDP and FSDP:

- **DDP**: Gradient synchronization happens during `loss.backward()`. The measurement uses these synchronized gradients.
- **FSDP**: Forward pass automatically handles parameter gathering. Make sure your `loss_closure` includes the forward pass so parameters are gathered correctly during probing.

## Troubleshooting

**Error: `AttributeError: 'AdamW' object has no attribute '_compute_update'`**
- You're using `torch.optim.AdamW` instead of `utils.optimizers.AdamW`
- Solution: Import and use the custom optimizer from `utils.optimizers`

**High number of probes (>20)**
- The `lr_guess` might be far from the true critical learning rate
- Solution: Adjust `lr_guess` to be closer to your actual training learning rate

**Loss closure returns tensor instead of scalar**
- The `loss_closure` should return `loss.item()`, not the tensor
- Solution: Add `.item()` to convert tensor to Python float

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