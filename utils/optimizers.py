# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union
from torch.optim.optimizer import ParamsT

import torch

class SGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """
        SGD optimizer implementation.

        Arguments:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)
            momentum: Momentum factor (default: 0.0)
            weight_decay: Weight decay coefficient (default: 0.0)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr = lr, momentum = momentum, weight_decay = weight_decay)
        super().__init__(params, defaults)

    def _compute_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        lr_override: float = None,
        virtual: bool = False
    ) -> torch.Tensor:

        """Compute parameter update without applying it.

        Returns:
            update: The computed update tensor
            new_state: Updated state dict (only if not virtual)
        """
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()

        # State initialization
        if group['momentum'] != 0 and len(state) == 0:
            state['momentum_buffer'] = torch.zeros_like(param, memory_format = torch.preserve_format)
        
        
        if group['momentum'] != 0:
            mu = state['momentum_buffer']
            mu_next = mu * group['momentum'] + grad
            update = mu_next
            if not virtual:
                state['momentum_buffer'].copy_(mu_next)
        else:
            update = grad
        
        lr = (lr_override if lr_override is not None else group['lr'])        
        
        param_update = - lr * update

        if group['weight_decay'] != 0:
            param_update = param_update - lr * group['weight_decay'] * param

        return param_update
    

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                state = self.state[param]
                update = self._compute_update(param, param.grad, state, group)
                param.add_(update)

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        """
        AdamW optimizer implementation.
        
        Arguments:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients used for computing running averages of gradient 
                  and its square (default: (0.9, 0.999))
            eps: Term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay: Weight decay coefficient (default: 0.01)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
        super().__init__(params, defaults)

    def _compute_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        group: dict,
        lr_override: float = None,
        virtual: bool = False
    ) -> torch.Tensor:
        """Compute parameter update without applying it.
        
        Returns:
            update: The computed update tensor
            new_state: Updated state dict (only if not virtual)
        """
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param, memory_format = torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(param, memory_format = torch.preserve_format)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        # Compute what the new step would be
        new_step = state['step'] + 1
        bias_correction1 = 1 - beta1 ** new_step
        bias_correction2 = 1 - beta2 ** new_step

        # Compute new momentum states
        new_exp_avg = exp_avg * beta1 + grad * (1 - beta1)
        new_exp_avg_sq = exp_avg_sq * beta2 + (grad **2) * (1 - beta2)
        
        numer = new_exp_avg / bias_correction1
        denom = (new_exp_avg_sq / bias_correction2).sqrt() + group['eps']

        lr = (lr_override if lr_override is not None else group['lr']) 

        # Compute parameter update
        param_update = - lr * numer / denom 

        # Add weight decay
        if group['weight_decay'] != 0:
            param_update -= param * (lr * group['weight_decay'])

        # Update states only if not virtual
        if not virtual:
            state['step'] = new_step
            exp_avg.copy_(new_exp_avg)
            exp_avg_sq.copy_(new_exp_avg_sq)

        return param_update

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                update = self._compute_update(p, p.grad, state, group)
                p.add_(update)

        return loss
