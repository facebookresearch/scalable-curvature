# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch import Tensor
from typing import Tuple, Dict, Any
import torch
from contextlib import contextmanager

from nano_instruct.utils.distributed_utils import (
    dist_mean,
)

def check_if_grads_None(model):
    """
    Checks if any parameter that requires grad has a None gradient.
    Logs parameter names and gradient norms.
    Returns True if any parameter has None grad, else False.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                return True
    return False

@contextmanager
def atUpdates(virtual_updates):
    """Context manager to temporarily apply virtual parameter updates.
    
    Arguments:
        virtual_updates: Dict from optimizer.virtual_step() mapping params to updates
        
    Usage:
        virtual_updates = optimizer.virtual_step(lr=0.01)
        with atParams(virtual_updates):
            loss = model(batch)  # Uses virtually updated parameters
        # Parameters restored automatically
    """
    try:
        # Apply virtual updates
        for param, update in virtual_updates.items():
            # gather here?
            param.data.add_(update)
            # scatter here?
        
        yield
        
    finally:
        # Remove virtual updates (restore original)
        for param, update in virtual_updates.items():
            # gather here?
            param.data.sub_(update)

@torch.no_grad()
def virtual_step(optim: torch.optim.Optimizer, lr: float) -> Dict[str, Tensor]:
    """Compute parameter updates without modifying parameters or states.
        
        Used for line search - computes what parameters would be after step.
        
        Arguments:
            lr: Learning rate to use. If None, uses group's learning rate
            
        Returns:
            Dict mapping parameters to their virtual updates
    """
    updates = {}
        
    for group in optim.param_groups:
        for param in group['params']:
            if param.grad is None:
                continue
            
            state = optim.state[param]
            update = optim._compute_update(param, param.grad, state, group, lr_override = lr, virtual = True)
            updates[param] = update  # updates already have the negative sign applied
                
    return updates


def compute_critical_learning_rate(
        world_mesh: Any,
        model: nn.Module, 
        loss_fn: nn.Module, 
        optim: torch.optim.Optimizer,
        batch: Tuple[torch.Tensor, torch.Tensor],
        lr_guess: float,
        tol_power: int = 4,
        recompute_grads: bool = True,
        logger: Any = None
    ) -> Tuple[Tuple[float, float], int]:
    """
    Computes the critical learning rate range for the given model and optimizer.
    Args:
        world_mesh: Distributed mesh for communication.
        model: The PyTorch model.
        loss_fn: Loss function that returns a dict with 'loss' key.
        optim: Optimizer instance (SGD or AdamW supported).
        batch: Input batch tuple.
        lr_guess: Initial guess for learning rate.
        tol_power: Controls binary search tolerance (default 4).
        recompute_grads: Whether to recompute gradients before starting.
        verbose: If True, logs detailed info.
    Returns:
        A tuple ((lr_lower, lr_upper), total_iterations) representing the critical LR range and number of iterations.
    """
    assert lr_guess > 0, f'Expected a non-zero positive number, got {lr_guess}'

    assert recompute_grads == True, f'recompute_grads = {recompute_grads} is not supported yet. Please set recompute_grads = True' 
    
    # setup optimizer function. 
    # NOTE: weight decay is not supported for torch SGD.
    # tracks the number of forward passes
    total_iters = 0

    # Compute loss and gradients    
    if recompute_grads or check_if_grads_None(model):
        # set gradients to None
        optim.zero_grad(set_to_none = True)
        
        # forward pass
        batch_results = loss_fn(batch)
        loss = batch_results['loss'] # per process loss
        # backward pass
        loss.backward()
        # compute the mean loss across all processes???
        loss_params = dist_mean(loss, mesh = world_mesh["world"]) 
    else:
        with torch.no_grad():
            # Forward pass plus loss computation
            batch_results = loss_fn(batch)
            loss = batch_results['loss'] # per process loss
            # compute the mean loss across all processes
            loss_params = dist_mean(loss, mesh = world_mesh["world"]) 
    
    total_iters += 1 # for the forward pass above
    if logger:
        logger.info(f'Initial loss: {loss_params:.4f} with lr: {lr_guess:.1e}')
    
    def update_and_compute_loss(lr: float):
        
        # Compute new parameters in place using optimizer step
        updates = virtual_step(optim = optim, lr = lr) # params next is a dict of {param: update}
    
        # Compute new loss with updated parameters
        with atUpdates(updates):
            batch_results = loss_fn(batch)
            loss = batch_results['loss'] # per process loss
            loss = dist_mean(loss, mesh = world_mesh["world"]) # compute the mean loss across all processes
            return loss # return the mean loss across all processes
        
    def exponential_search(loss_before: float, lr_guess: float, max_iters: int = 100,):
        
        lr = lr_guess
        iteration = 0

        # Compute initial loss after applying the initial learning rate
        loss_next = update_and_compute_loss(lr)
        iteration += 1

        # Determine initial direction:
        # 1. if loss decreases in the next step; we need to increase the learning rate 
        # 2. if the loss increases in the next step; we need to decrease the learning rate
        direction = 1 if loss_next < loss_before else -1

        while iteration < max_iters:
            # Adjust learning rate in the determined direction
            lr *= 2 ** direction
            # Update parameters with the new learning rate and compute loss
            loss_next = update_and_compute_loss(lr)
            if logger:
                logger.info(f'Exponential search iteration: {iteration} Loss before: {loss_before:0.1e} Loss next: {loss_next:0.1e} LR: {lr:0.1e}')

            iteration += 1 # tracks how many times we computed the loss

            # Check if the loss behavior has changed as desired in both cases
            if (direction == 1 and loss_next > loss_before):
                lr_lower = lr / 2
                lr_upper = lr
                return  (lr_lower, lr_upper), iteration
            
            if (direction == -1 and loss_next < loss_before):
                lr_lower = lr
                lr_upper = lr * 2     
                return (lr_lower, lr_upper), iteration

        # If we reach here, it means we didn't find a suitable range
        if logger:
            logger.warning(f'Exponential search did not converge within {max_iters} iterations. Returning the last computed range.')
        
        return (lr, lr), iteration

    def binary_search(loss_before: float, lr_lower: float, lr_upper: float, tol: float = 1/16):
        iterations = 0

        while (1 - lr_lower / lr_upper) > tol:
            # find the mid point of the learning rates
            lr_mid = (lr_lower + lr_upper) / 2.0
            # compute loss wrt at the current LR
            loss_mid = update_and_compute_loss(lr_mid)
            
            if logger:
                logger.info(f'Binary search iteration: {iterations} Loss before: {loss_before:0.1e} Loss mid: {loss_mid:0.1e} LR lower: {lr_lower:0.1e} LR upper: {lr_upper:0.1e} LR mid: {lr_mid:0.1e}')
            iterations += 1

            # Adjust the range based on the loss at the midpoint
            if loss_mid > loss_before:
                lr_upper = lr_mid
            else:
                lr_lower = lr_mid                
        return (lr_lower, lr_upper), iterations

    # Exponential search
    (lr_lower, lr_upper), num_iters = exponential_search(loss_params, lr_guess)
    
    total_iters += num_iters

    if lr_lower < lr_upper:
        # Binary search
        tol = 1 / (2 ** tol_power)
        (lr_lower, lr_upper), num_iters = binary_search(loss_params, lr_lower, lr_upper, tol)
        if logger:
            logger.info(f'Final critical LR range: {lr_lower:.1e} to {lr_upper:.1e} after {num_iters} iterations.')
        total_iters += num_iters

    # free up model gradients if recompute_grads is True
    if recompute_grads or check_if_grads_None(model):
        optim.zero_grad(set_to_none = True)
    
    return (lr_lower, lr_upper), total_iters

