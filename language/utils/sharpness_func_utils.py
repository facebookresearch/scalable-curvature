"""
Utility file for estimating the Hessian spectrum using Power Iteration"""

# imports
import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from torch import Tensor
from typing import Tuple, Any, Dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.compile
def compute_hvp(model: nn.Module, loss_fn: nn.Module, batch: Tuple, vec: Tensor, P: Tensor = None):

    # extract the examples    
    x, y = batch
    params_dict = dict(model.named_parameters()) 

    if P is not None:
        vec = vec / P.sqrt()
    
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
        if P is not None:
            flat_hvp = flat_hvp / P.sqrt()
        return flat_hvp.detach()

def power_iteration(model, loss_fn, batch, vec: Tensor = None, max_iters: int = 200, tol: float = 1e-06, P = None):
    """ Power iteration to compute the leading eigenvalue of the Hessian """

    nparams = len(parameters_to_vector((model.parameters())))
    if vec is None: # initialize a random vector
        vec = torch.randn(nparams, dtype = torch.float32, requires_grad = False).pin_memory().to(device, non_blocking = True) 
        
    lam_prev = 0.0
        
    for step in range(max_iters):
            
        # compute the Hessian vector product
        hvp = compute_hvp(model = model, loss_fn = loss_fn, batch = batch, vec = vec, P = P)
            
        # Rayleigh quotient
        lam_est = torch.dot(vec, hvp) / torch.dot(vec, vec) 
        # print(step, lam_est.item(), abs(1 - lam_est / lam_prev).item())

        # check for convergence
        if lam_est >= 0 and abs(1 - lam_est / lam_prev).item() < tol:
            break
            
        lam_prev = lam_est
        vec = hvp / torch.norm(hvp) # already normalized
            
    return lam_est, vec, step + 1

def get_sharpness_power_method(model: nn.Module, loss_fn: nn.Module, batch: Tuple, max_iters: int = 200, tol: float = 1e-06, vec: Tensor = None):
    " Computes sharpness of a model using power iterations "
    sharpness, vec, num_iters = power_iteration(model, loss_fn, batch, vec = vec, max_iters = max_iters, tol = tol, P = None)
    return sharpness, vec, num_iters


def get_adam_preconditioner(params, loss, optim):
    "Assumes the existence of gradients at param.grad "
    P_dict = {}

    param_to_name = {param: name for name, param in params}
    # optim.param_groups is a list of groups. Each group element is a dict
    for group in optim.param_groups: 
        # group is a dict with keys: [params, lr, weight_decay, betas, eps ...]
        beta1, beta2 = group['betas']
        eps = group['eps']
        weight_decay = group['weight_decay']
        lr_scale = group.get('lr_scale', 1.0)

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

            nu = param_state.get('exp_avg_sq', torch.zeros_like(grad.data))

            # compute the bias corrected nu
            nu_next = beta2 * nu + (1-beta2) * grad**2
            nu_next = nu_next / (1-beta2**step) 

            P_dict[name] = (1 - beta1**step) * (nu_next.sqrt() + eps) / lr_scale
    P = parameters_to_vector(P_dict.values())     
    return P


def get_pre_sharpness_power_method(model: nn.Module, loss_fn: nn.Module, optim: torch.optim.Optimizer, batch: Tuple, max_iters: int = 10_000, tol: float = 1e-06, vec: Tensor = None):

    x, y = batch

    @torch.compile
    def compute_loss():
        with torch.amp.autocast(device_type = 'cuda', dtype = torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits, y)
        return loss

    # Compute loss and gradients because Adam pre-conditioner requires gradients; Note grads can be utilized from critical LR computations
    print(f'Recomputing grads for Pre-conditioned sharpness computation....')
    # set gradients to None
    optim.zero_grad(set_to_none = True)
    
    loss = compute_loss()
    loss_params = loss.item()
    loss.backward()
    
    P = get_adam_preconditioner(model.named_parameters(), loss_fn, optim)
    assert len(P) == len(parameters_to_vector(model.parameters())), 'Pre-conditioner vector is smaller than the number of model params'
    optim.zero_grad(set_to_none = True) # free up the memory for sharpness computation

    sharpness, vec, num_iters = power_iteration(model, loss_fn, batch, vec = vec, max_iters = max_iters, tol = tol, P = P)
    return sharpness, vec, num_iters


def get_trace_hutchinson(model: nn.Module, loss_fn: nn.Module, batch: Tuple, max_iters: int = 10, tol: float = 1e-06, vec: Tensor = None, P = None):

    def hutchinson(vec: Tensor = None, max_iters: int = 100, tol: float = 1e-06):
        """ Hutchinson's trick to compute the trace of the Hessian """
        nparams = len(parameters_to_vector((model.parameters())))

        trace_prev = 0.0
        trace_cumsum = 0.0

        for step in range(max_iters):
            
            # compute the Hessian vector product
            vec = torch.randn(nparams, dtype = torch.float32, requires_grad = False).pin_memory().to(device, non_blocking = True) 
            hvp = compute_hvp(model = model, loss_fn = loss_fn, batch = batch, vec = vec, P = P)
            
            # Rayleigh quotient
            trace_cumsum += torch.dot(vec, hvp)

            trace_est = trace_cumsum / (step + 1)  # average over the number of iterations

            if abs(1 - trace_est / trace_prev).item() < tol:
                break
            # print(trace_est / trace_prev)
            # print(f"Step {step+1}, Trace Estimate: {trace_est.item()}, Relative Change: {abs(1 - trace_est / trace_prev).item()}")  
            
            trace_prev = trace_est
            
            
        return trace_est, step+1
        
    trace, num_iters = hutchinson(vec = vec, max_iters = max_iters, tol = tol)
    return trace, num_iters


# def get_hessian_batch(ctx: Any, model: nn.Module, loss_fn: nn.Module, batch: Tuple, topk = 1, tol = 1e-09, v0 = None, P = None):
#     """ 
#     Compute the leading Hessian eigenvalues using the entire dataset

#     Inputs:
#         - model: torch model
#         - loss_fn: torch loss function: should use sum loss and not mean loss as its helpful to do mini batch calculations
#         - batch: (x, y) example batch
#         - topk: number of eigenvalues requires
#         - v0: initial vector for the Lanczos method; if None, a random vector is used
#         - P: pre-conditioner for the Hessian
    
#     Output: 
#         - eigvals: top topk eigenvalues
#         - eigvecs: top topk eigenvectors
#     Remark: If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
#     """

#     nparams = len(parameters_to_vector((model.parameters())))

#     def matvec(vec: np.ndarray):
#         " Takes a numpy array, pushes it to GPU; computes hvp; and returns it to cpu again"
#         vec = torch.tensor(vec, dtype = torch.bfloat16).pin_memory().to(device, non_blocking = True) 
#         return compute_hvp(model = model, loss_fn = loss_fn, batch = batch, vec = vec, P = P).detach().cpu()
    
#     # create and operator for the eigsh function
#     operator = linalg.LinearOperator((nparams, nparams), matvec = matvec) # matvec is a callable function that returns H | vector >

#     # compute eigenvalues and eigenvector
#     eigvals, eigvecs = linalg.eigsh(operator, k = topk, tol = tol, return_eigenvectors = True, v0 = v0)
#     eigvals = eigvals

#     return eigvals, eigvecs