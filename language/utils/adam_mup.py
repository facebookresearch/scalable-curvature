import math
from typing import Iterable, Tuple, Union, Optional
import json

import torch
import torch.nn as nn

class AdamW(torch.optim.Optimizer):
    """
    Generalized Adam optimizer with per 
    """
    def __init__(
            self,
            named_parameters,
            *,
            lr: Union[float, torch.Tensor],
            betas: Tuple[float, float],
            eps: float,
            weight_decay: float,
            embd_dim: int,
    ):

        optim_groups = []

        # create parameter groups
        for param_name, param in named_parameters:

            if not param.requires_grad:
                continue
                
            group_config = {} # dictionary for each parameter
            group_config['params'] = param
            
            # No weight decay for norm or bias params
            if param.dim() >= 2:
                group_config['weight_decay'] = weight_decay
            else:
                group_config['weight_decay'] = 0.0

            optim_groups.append(group_config)
        
        # default parameters for params
        defaults = dict(lr = lr, betas = betas, eps = eps) 
        super().__init__(optim_groups, defaults)
    
    @torch.no_grad()
    def step(self):

        """ Optimizer step """
                
        for group_config in self.param_groups:
            # group hparams
            beta1, beta2 = group_config['betas']
            lr, eps = group_config['lr'], group_config['eps']
            name = group_config['name']
            weight_decay = group_config['weight_decay']

            for param in group_config['params']:
                if param.grad is None:
                    continue
                # for every parameter, state stores the first and second moment
                state = self.state[param] # get the state corresponding to the param
            
                # Optimizer state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['mu'] = torch.zeros_like(param, memory_format = torch.preserve_format)
                    state['nu'] = torch.zeros_like(param, memory_format = torch.preserve_format)
                    
                # Optimizer state update
                # update first moment using p.grad
                state['mu'].mul_(beta1).add_(param.grad, alpha = 1-beta1)
                # update the second moment using squared gradients
                state['nu'].mul_(beta2).add_(param.grad.square(), alpha = 1-beta2)
                
                # Add weight decay
                if weight_decay > 0.0:
                    param.mul_(1 - lr*weight_decay)
                    
                # Compute update
                state['step'] += 1

                # bias correction
                mu_hat = state['mu'] / (1 - beta1 ** state['step']) 
                nu_hat = state['nu'] / (1 - beta2 ** state['step'])

                # Optimizer parameter update
                update = lr * (mu_hat / (nu_hat.sqrt() + eps))
                # Apply update
                param.add_(-update)
        return

