"""
Simple PyTorch HVP for multiple vectors (no vmap complications)
"""

import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import Tensor
from typing import Tuple, Dict, Any

import utils.optimizers as optimizers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

@torch.no_grad()
def virtual_step(optim: torch.optim.Optimizer, lr: float = 1.0) -> Dict[str, Tensor]:
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
            # compute update without learning rate; thats why lr = 1.0
            update = optim._compute_update(param, param.grad, state, group, lr_override = lr, virtual = True)
            updates[param] = update  # updates already have the negative sign applied
                
    return updates


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
        with torch.amp.autocast(device_type = 'cuda', dtype = torch.bfloat16):
            logits = functional_call(model, params, (x,)) 
            loss = loss_fn(logits, y)
            return loss
    
    # hvp computation
    with torch.amp.autocast(device_type = 'cuda', dtype = torch.bfloat16):
        grad_fn = grad(compute_loss)
        _, hvp = jvp(grad_fn, (params_dict,), (vec_dict,)) # jvp computes the Jacobian-vector product
        flat_hvp = parameters_to_vector(hvp.values()) # flatten the hvp
        return flat_hvp.detach()


def get_dir_sharpness(ctx: Any, model: nn.Module, loss_fn: nn.Module, optim: torch.optim.Optimizer, batch: Tuple, recompute_grads: bool = True) -> Tensor:
    "Computes the sharpness in the parameter update direction "

    # NOTE: ideally, we will use the same optimizer as the one used for training by writing our own optimizers with optimal parameter updates

    # extract the examples
    x, y = batch

    @torch.compile
    def compute_loss():
        with ctx:
            logits = model(x)
            loss = loss_fn(logits, y)
        return loss

    if recompute_grads or check_if_grads_None(model):    
        # recompute the gradients
        optim.zero_grad(set_to_none = True)
    
        loss = compute_loss()
        loss_params = loss.item()
        loss.backward()

    # get the gradients
    grads_dict = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
    flat_grads = parameters_to_vector(grads_dict.values())  # flatten the gradients

    # compute the update direction
    update = virtual_step(optim, lr = 1.0)  # virtual step computes the update direction; note it has the negative sign applied
    # update is param: update; we want name: update
    update = {name: update[param] for name, param in model.named_parameters() if param.grad is not None}
    # flatten the update direction
    flat_update = parameters_to_vector(update.values())  # flatten the update direction

    # compute the directional sharpness

    # compute hvp
    hvp = compute_hvp(model, loss_fn, batch, flat_update)

    # compute the sharpness
    dir_sharpness = - torch.dot(flat_update, hvp) / torch.dot(flat_update, flat_grads)  # negative sign because the update is already negative

    return dir_sharpness.item()
    
    