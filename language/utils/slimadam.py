import math
from typing import Iterable, Tuple, Union, Optional
import json

import torch
import torch.nn as nn
import torch.distributed as dist
# dist.init_process_group(backend='nccl')  # or 'gloo' for CPU
from torch.distributed._tensor import Replicate
device = 'cuda' if torch.cuda.is_available() else 'cpu'


import math
from typing import Iterable, Tuple, Union, Optional


class SlimAdamW(torch.optim.Optimizer):
    """
    Generalized SlimAdam which also takes into account KQV compressions
    * SlimAdam takes in a dictionary `axes` corresponding to each block
    """
    def __init__(
            self,
            named_parameters,
            *,
            lr: Union[float, torch.Tensor],
            betas: Tuple[float, float],
            eps: float,
            weight_decay: float,
            rules_json_path: Optional[str] = None,
            verbose = True,
    ):
        # named model parameters 
        self.named_parameters = named_parameters

        self.compression_rules = {} # if empty, get will assign None: group_config['compress_dims'] = self.compression_rules.get(param_name, None)
        if rules_json_path is not None:
            # load compression rules
            with open(rules_json_path, 'r') as fi:
                self.compression_rules = json.load(fi)

            # convert compression rules from list to tuple
            self.compression_rules = {k: tuple(v) if v is not None else None for k, v in self.compression_rules.items()}

            # Remove unwanted prefix if present (for nanoGPT compatibility)
            unwanted_prefix = '_orig_mod.'
            for key, value in list(self.compression_rules.items()):
                if key.startswith(unwanted_prefix):
                    self.compression_rules[key[len(unwanted_prefix):]] = self.compression_rules.pop(key)
        
        # for multiple gpus
        self.world_size = torch.cuda.device_count()
        self.verbose = verbose

        optim_groups = []

        total_params = 0
        total_slim_params = 0
    
        # create parameter groups
        for param_name, param in named_parameters:

            if not param.requires_grad:
                continue
                
            total_params += param.numel()

            group_config = {} # dictionary for each parameter
            group_config['name'] = param_name
            group_config['params'] = param

            # assign compression axes
            group_config['compress_dims'] = self.compression_rules.get(param_name, None)

            compressed_size = param.numel()
            compressed_shape = param.shape
            if group_config['compress_dims'] is not None:
                compressed_shape = list(param.shape)
                for dim in group_config['compress_dims']:
                    compressed_shape[dim] = 1
                compressed_size = math.prod(compressed_shape)
            
            total_slim_params += compressed_size
            
            if verbose:
                print(f'Found param block: {param_name} with shape: {param.shape}')
                print(f'Compressing along', group_config['compress_dims'])
                print(f'Resultant shape', tuple(compressed_shape))


            # No weight decay for normed params
            if param.dim() >= 2:
                group_config['weight_decay'] = weight_decay
            else:
                group_config['weight_decay'] = 0.0

            optim_groups.append(group_config)
        
        savings_pct = (1 - total_slim_params / total_params) * 100
        print("="*100)
        print(f"SlimAdam is saving {savings_pct:.2f}% of second moments")
        print("="*100)

        self.param_savings = {'total_params': total_params, 'total_slim_params': total_slim_params, 'savings_pct': savings_pct}
        # default parameters for params
        defaults = dict(lr = lr, betas = betas, eps = eps) 
        super().__init__(optim_groups, defaults)
    
    @torch.no_grad()
    def compress_grad_squared(self, grad: torch.Tensor, dims: Tuple[int, ...] = None) -> torch.Tensor:
        """ squares the gradient and the compresses along dim if provided; if dims = None, no compression is performed """
        if dims is None:
            return grad**2
        else:
            return torch.mean(grad * grad, dim = dims, keepdim = True)

    @torch.no_grad()
    def step(self):
        """ Optimizer step """
                
        for group_config in self.param_groups:
            # group hparams
            beta1, beta2 = group_config['betas']
            lr, eps = group_config['lr'], group_config['eps']
            name = group_config['name']
            weight_decay = group_config['weight_decay']

            # compression dims for the second moment
            compress_dims = group_config['compress_dims']
            
            for param in group_config['params']:
                if param.grad is None:
                    continue
                # for every parameter, state stores the first and second moment
                state = self.state[param] # get the state corresponding to the param
            
                # Optimizer state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['mu'] = torch.zeros_like(param, memory_format = torch.preserve_format)
                    state['nu'] = torch.zeros_like(self.compress_grad_squared(param, compress_dims), memory_format = torch.preserve_format)
                    if self.verbose:
                        print(name, compress_dims, state['nu'].shape)

                # Optimizer state update
                # update first moment using p.grad
                state['mu'].mul_(beta1).add_(param.grad, alpha = 1-beta1)
                # compute gradient squared                
                grad_squared = self.compress_grad_squared(param.grad, compress_dims)
                # update the second moment using squared gradients
                state['nu'].mul_(beta2).add_(grad_squared, alpha = 1-beta2)
                
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

