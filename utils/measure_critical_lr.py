# utils/measure_critical_lr.py
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Tuple, Dict, Callable, Any
from torch import Tensor

# Helper: Check gradients
def check_if_grads_None(model):
    for param in model.parameters():
        if param.requires_grad and param.grad is None:
            return True
    return False

# Context Manager: Temporarily swap parameters
@contextmanager
def atParams(model: torch.nn.Module, update_dict: dict):
    cache = {}
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                
                if name in update_dict:
                    cache[name] = param.clone()
                    param.add_(update_dict[name])
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in cache:
                    param.data.copy_(cache.pop(name))

# Helper: Compute virtual step (Requires custom optimizer)
@torch.no_grad()
def virtual_step(model: torch.nn.Module, optim: torch.optim.Optimizer, lr: float) -> Dict[str, Tensor]:
    updates = {}
    for group in optim.param_groups:
        for name, param in zip(group.get("param_names", [n for n, _ in model.named_parameters()]), group["params"]):
            if param.grad is None:
                continue
                
            # Requires custom AdamW/SGD from utils.optimizers
            if not hasattr(optim, '_compute_update'):
                raise AttributeError("Optimizer must have '_compute_update' method. Use custom AdamW/SGD from utils.optimizers.")
                
            state = optim.state[param]
            update = optim._compute_update(
                param, param.grad, state, group, lr_override=lr, virtual=True
            )
            updates[name] = update
    return updates

def measure_critical_lr(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,                             # <--- Tool handles the batch object
    loss_closure: Callable[[Any], float],   # <--- User defines how to compute loss on that batch
    lr_guess: float = 1e-3,
    tol_power: int = 4,
    logger = None
) -> Tuple[float, float, int]:
    
    # 1. Compute Initial Loss
    with torch.no_grad():
        loss_init = loss_closure(batch)

    if logger:
        logger.info(f"Initial loss: {loss_init:.4f}")

    # 2. Define the Probe Function
    @torch.no_grad()
    def update_and_compute_loss(lr: float):
        # Calculate what the weights WOULD be at this LR
        updates = virtual_step(model, optimizer, lr)
        
        # Temporarily apply them
        with atParams(model, updates):
            loss = loss_closure(batch)
        return loss

    # 3. Exponential Search
    lr = lr_guess
    iteration = 0
    loss_next = update_and_compute_loss(lr)
    iteration += 1
    
    direction = 1 if loss_next < loss_init else -1
    lr_lower, lr_upper = lr, lr
    
    MAX_ITERS = 100
    while iteration < MAX_ITERS:
        lr *= 2**direction
        loss_next = update_and_compute_loss(lr)
        iteration += 1
        
        if direction == 1 and loss_next > loss_init:
            lr_lower = lr / 2
            lr_upper = lr
            break
        if direction == -1 and loss_next < loss_init:
            lr_lower = lr
            lr_upper = lr * 2
            break

    # 4. Binary Search
    tol = 1 / (2**tol_power)
    while (1 - lr_lower / lr_upper) > tol and iteration < MAX_ITERS + 20:
        lr_mid = (lr_lower + lr_upper) / 2.0
        loss_mid = update_and_compute_loss(lr_mid)
        iteration += 1
        
        if loss_mid > loss_init:
            lr_upper = lr_mid
        else:
            lr_lower = lr_mid
            
    return lr_lower, lr_upper, iteration
