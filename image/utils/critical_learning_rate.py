# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch import Tensor
from typing import Tuple, Dict
import torch


def check_if_grads_None(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(name, param.grad)
                return True
            else:
                print(name, param.shape, torch.norm(param.grad))
    return False

def sgd_fn(params: Dict[str, Tensor], optim: torch.optim.Optimizer, lr: float) -> Dict[str, Tensor]:
    """ Computes the next parameters using SGD w/ momentum"""
    params_next = {}
    param_to_name = {param: name for name, param in params}

    for group in optim.param_groups: 
        beta = group['momentum']
        tau = group['dampening']
        weight_decay = group['weight_decay']

        for param in group['params']:
            if param.grad is None:
                continue

            # get name and gradients
            name = param_to_name[param]
            grad = param.grad

            # Get momentum state
            param_state = optim.state[param]
            if 'momentum_buffer' in param_state:
                mu = param_state['momentum_buffer']
                mu_next = beta * mu + (1 - tau) * grad
                params_next[name] = param * (1 - lr * weight_decay) - lr * mu_next
            else:
                params_next[name] = param * (1 - lr * weight_decay) - lr * grad
                
    return params_next

def adam_fn(params: Dict[str, Tensor], optim: torch.optim.Optimizer, lr: float) -> Dict[str, Tensor]:
    """ Computes the next parameters using AdamW """
    params_next = {}
    param_to_name = {param: name for name, param in params}


    # optim.param_groups is a list of groups. Each group element is a dict
    for group in optim.param_groups: 
        # group is a dict with keys: [params, lr, weight_decay, betas, eps ...]
        beta1, beta2 = group['betas']
        eps = group['eps']
        weight_decay = group['weight_decay']

        for param in group['params']:
            # group[params] is a group of params
            if param.grad is None:
                continue

            # get name and gradients
            name = param_to_name[param]
            grad = param.grad

            # Get the optimizer state
            param_state = optim.state.get(param, {})
            step = param_state.get('step', 0) + 1

            mu = param_state.get('exp_avg', torch.zeros_like(grad.data))
            nu = param_state.get('exp_avg_sq', torch.zeros_like(grad.data))

            # update the first moment
            mu_next = beta1 * mu + (1-beta1) * grad
            mu_next = mu_next / (1-beta1**step) # bias correction

            nu_next = beta2 * nu + (1-beta2) * grad**2
            nu_next = nu_next / (1-beta2**step)
            
            params_next[name] = param * (1 - lr * weight_decay) - lr * mu_next / (nu_next.sqrt() + eps)
                
    return params_next



def compute_critical_learning_rate(
        model: nn.Module, 
        loss_fn: nn.Module, 
        optim: torch.optim.Optimizer,
        batch: Tuple[torch.Tensor, torch.Tensor],
        lr_guess: float,
        tol_power: int = 4,
        recompute_grads: bool = True,
        verbose: bool = False
    ):
    """
    Computes the critical learning rate using 

    Inputs:
        - model: torch model
        - loss_fn: torch loss function
        - batch: (x, y)
        - lr_guess: initial guess for the learning rate
        - delta: small constant for termination condition: L(theta_t+1) > L(theta_t)
    Returns:
        - lr_star: critical learning rate

    ## NOTE: Sanity check L(theta_t+1) = loss in the next step would fail for SGD
    """

    assert lr_guess > 0, f'Expected a non-zero positive number, got {lr_guess}'
    
    # setup optimizer function. NOTE: weight decay is not supported well.
    if isinstance(optim, torch.optim.SGD):
        optim_fn = sgd_fn
    elif isinstance(optim, torch.optim.AdamW):
        optim_fn = adam_fn
    else:
        raise ValueError("Unsupported optimizer. If using Adam, please switch to AdamW with zero weight decay.")

    total_iters = 0
    # extract the examples
    x, y = batch

    @torch.compile
    def compute_loss():
        logits = model(x)
        loss = loss_fn(logits, y)
        return loss

    # NOTE: I am not counting the forward and backward pass into the calculation because a smart implmentation can perform a optimizer step right here
    # Compute loss and gradients    
    if recompute_grads or check_if_grads_None(model):
        # print(f'Recomputing grads .....')
        # set gradients to None
        optim.zero_grad(set_to_none = True)
    
        loss = compute_loss()
        loss_params = loss.item()
        loss.backward()
    else:
        with torch.no_grad():
            loss_params = compute_loss().item()
    
    @torch.compile
    def update_and_compute_loss(lr: float):
        with torch.no_grad():
            # Compute new parameters without modifying the original ones
            params_next = optim_fn(params = model.named_parameters(), optim = optim, lr = lr)
            # Compute new loss with updated parameters
            logits = torch.func.functional_call(model, params_next, (x,))
            loss = loss_fn(logits, y)
        return loss
    
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
            if verbose:
                print(f'Exponential search: {loss_before:0.4f}, {loss_next:0.4f}, lr: {lr}')

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
            
        print('Failed to converge within the maximum number of iterations')
        return (lr, lr), iteration

    def binary_search(loss_before: float, lr_lower: float, lr_upper: float, tol: float = 1/16):
        iterations = 0

        while (1 - lr_lower / lr_upper) > tol:
            # find the mid point of the learning rates
            lr_mid = (lr_lower + lr_upper) / 2.0
            # compute loss wrt at the current LR
            loss_mid = update_and_compute_loss(lr_mid)
            if verbose:
                print(f'Binary search: {loss_before:0.4f}, {loss_mid:0.4f}, lr: {lr_mid}')
            iterations += 1

            # Adjust the range based on the loss at the midpoint
            if loss_mid > loss_before:
                lr_upper = lr_mid
            else:
                lr_lower = lr_mid                
        return (lr_lower, lr_upper), iterations

    # Exponential search
    (lr_lower, lr_upper), num_iters = exponential_search(loss_params, lr_guess)
    if verbose:
        print(f'Exponential Range found: ({lr_lower:0.4f}, {lr_upper:0.4f}) in {num_iters} iterations')

    total_iters += num_iters

    if lr_lower < lr_upper:
        # Binary search
        tol = 1 / (2 ** tol_power)
        (lr_lower, lr_upper), num_iters = binary_search(loss_params, lr_lower, lr_upper, tol)
        if verbose:
            print(f'Binary Range found: ({lr_lower:0.4f}, {lr_upper:0.4f}) in {num_iters} iterations')
        total_iters += num_iters
    else:
        print('Skipping binary search as exponential search did not converge...')

    # free up model gradients
    optim.zero_grad(set_to_none = True)
    
    return (lr_lower, lr_upper), total_iters

