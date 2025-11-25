"""
Simple PyTorch HVP for multiple vectors (no vmap complications)
"""

import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import Tensor
from typing import Tuple, Dict

import cupy as cp                 
from cupyx.scipy.sparse.linalg import LinearOperator, lobpcg
from utils.optimizers import AdamW, SGD
from utils.critical_lr_cache_utils import virtual_step

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.compile
def compute_hvp(model: nn.Module, loss_fn: nn.Module, batch: Tuple, vec: Tensor):

    """ 
    Computes the Hessian-vector product (HVP) for a given model, loss function, and batch  
    If P is not None, HVP of P^{-1/2} H P^{-1/2} is computed
    """

    # extract the examples    
    x, y = batch
    params_dict = dict(model.named_parameters()) 
    

    vec_dict = {name:torch.empty_like(param) for name, param in params_dict.items()}
    vector_to_parameters(vec, vec_dict.values()) # convert the vector to parameters
    
    def compute_loss(params: Dict[str, Tensor]):
        """ Computes the loss for the given batch """
        with torch.amp.autocast(device_type = 'cuda', dtype = torch.float32):
            logits = functional_call(model, params, (x,)) 
            loss = loss_fn(logits, y)
            return loss
    
    # hvp computation
    with torch.amp.autocast(device_type = 'cuda', dtype = torch.float32):
        grad_fn = grad(compute_loss)
        _, hvp = jvp(grad_fn, (params_dict,), (vec_dict,)) # jvp computes the Jacobian-vector product
        flat_hvp = parameters_to_vector(hvp.values()) # flatten the hvp
        return flat_hvp.detach()


# def sgd_fn(params: Dict[str, Tensor], optim: torch.optim.Optimizer) -> Dict[str, Tensor]:
#     """ Computes the next parameters using SGD w/ momentum"""
#     update = {}
#     param_to_name = {param: name for name, param in params.items()}

#     for group in optim.param_groups: 
#         beta = group['momentum']
#         tau = group['dampening']
#         weight_decay = group['weight_decay']

#         for param in group['params']:
#             if param.grad is None:
#                 continue

#             # get name and gradients
#             name = param_to_name[param]
#             grad = param.grad

#             # Get momentum state
#             param_state = optim.state[param]
#             if 'momentum_buffer' in param_state:
#                 mu = param_state['momentum_buffer']
#                 mu_next = beta * mu + (1 - tau) * grad
#                 update[name] = - weight_decay * param - mu_next
#             else:
#                 update[name] =  - weight_decay * param - grad
                
#     return update

# def adam_fn(params: Dict[str, Tensor], optim: torch.optim.Optimizer) -> Dict[str, Tensor]:
#     """ Computes the next parameters using AdamW """
#     update = {}
#     param_to_name = {param: name for name, param in params.items()}

#     # optim.param_groups is a list of groups. Each group element is a dict
#     for group in optim.param_groups: 
#         # group is a dict with keys: [params, lr, weight_decay, betas, eps ...]
#         beta1, beta2 = group['betas']
#         eps = group['eps']
#         weight_decay = group['weight_decay']

#         for param in group['params']:
#             # group[params] is a group of params
#             if param.grad is None:
#                 continue

#             # get name and gradients
#             name = param_to_name[param]
#             grad = param.grad

#             # Get the optimizer state
#             param_state = optim.state.get(param, {})
#             step = param_state.get('step', 0) + 1

#             mu = param_state.get('exp_avg', torch.zeros_like(grad))
#             nu = param_state.get('exp_avg_sq', torch.zeros_like(grad))

#             # update the first moment
#             mu_next = beta1 * mu + (1-beta1) * grad
#             mu_next = mu_next / (1-beta1**step) # bias correction

#             nu_next = beta2 * nu + (1-beta2) * grad**2
#             nu_next = nu_next / (1-beta2**step)
            
#             update[name] = - weight_decay * param  - mu_next / (nu_next.sqrt() + eps)
                
#     return update

def get_dir_sharpness(model: nn.Module, loss_fn: nn.Module, optim: torch.optim.Optimizer, batch: Tuple):
    "Computes the sharpness in the parameter update direction "

    # NOTE: ideally, we will use the same optimizer as the one used for training by writing our own optimizers with optimal parameter updates

    # if isinstance(optim, torch.optim.SGD) or isinstance(optim, SGD):
    #     optim_fn = sgd_fn
    # elif isinstance(optim, torch.optim.AdamW) or isinstance(optim, AdamW):
    #     optim_fn = adam_fn
    # else:
    #     raise ValueError("Unsupported optimizer. If using Adam, please switch to AdamW with zero weight decay.")

    # extract the examples
    x, y = batch

    @torch.compile
    def compute_loss():
        with torch.amp.autocast(device_type = 'cuda', dtype = torch.float32):
            logits = model(x)
            loss = loss_fn(logits, y)
        return loss
    
    optim.zero_grad(set_to_none = True)
    
    loss = compute_loss()
    loss_params = loss.item()
    loss.backward()

    # compute the gradients
    grads_dict = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
    flat_grads = parameters_to_vector(grads_dict.values())  # flatten the gradients

    # compute the update direction
    update = virtual_step(model = model, optim = optim, lr = 1.0)
    flat_update = parameters_to_vector(update.values())  # flatten the update direction

    # compute the directional sharpness

    # compute hvp
    hvp = compute_hvp(model = model, loss_fn = loss_fn, batch = batch, vec = flat_update)

    # compute the sharpness
    dir_sharpness = torch.dot(flat_update, hvp) / torch.dot(flat_update, -flat_grads) 
    return dir_sharpness.detach().cpu()
    
    