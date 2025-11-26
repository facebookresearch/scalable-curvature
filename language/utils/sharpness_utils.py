"""
Utility file for estimating the Hessian spectrum using Lanzcos. 
Code adopted from https://github.com/locuslab/edge-of-stability/
"""

# imports
import numpy as np

import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from torch import Tensor
from typing import Tuple, Any, Dict

import scipy.sparse.linalg as linalg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_hvp(ctx: Any, model: nn.Module, loss_fn: nn.Module, batch: Tuple, vec: Tensor, P: Tensor = None):
    """
    Computes Hessian-vector product (hvp = H v) for the entire dataset given a vector v
    hvp = H v

    Inputs:
        - model: torch model
        - loss_fn: torch loss function: should use sum loss and not mean loss as its helpful to do mini batch calculations
        - batch: (x, y) tuple of inputs and outputs 
        - vec: vec (v) for computing Hessian vector product; should be on cuda
        - batch_size: batch size for computing hvp
        TODO: Add pre-conditioner
        - P: pre-conditioner for the Hessian
    Outputs:
        - hvp = H v
        Remark: If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
    """
    vec = torch.tensor(vec, dtype = torch.float).pin_memory().to(device, non_blocking = True)
    # vec = torch.tensor(vec, dtype = torch.float).to(device)

    if P is not None:
        P = P.pin_memory().to(device, non_blocking = True)
        vec = vec / P.sqrt()

    x, y = batch
    # forward pass in bfloat16
    # NOTE: ctx is a context manager that allows us to use bfloat16 precision for the forward pass
    with ctx:
        logits = model(x)
        loss = loss_fn(logits, y)

    # compute the gradients
    grads = torch.autograd.grad(loss, inputs = model.parameters(), create_graph = True) # | grad > 
    # compute the dot profuct of the gradients and the vector
    vec = vec.view(-1)
    gv_prod = torch.dot(parameters_to_vector(grads), vec) # gradient vector product < grad | vector >
    
    del grads  # grads not needed; Free memory
    hvp = torch.autograd.grad(gv_prod, model.parameters(), retain_graph = True)

    del gv_prod  # gv_prod not needed; Free memory
    # flatten the hvp
    hvp = torch.cat([h.contiguous().view(-1) for h in hvp])

    if P is not None:
        hvp = hvp / P.sqrt()
    return hvp
    
def get_hessian_batch(ctx: Any, model: nn.Module, loss_fn: nn.Module, batch: Tuple, topk = 3, tol = 1e-09, P = None):
    """ 
    Compute the leading Hessian eigenvalues using the entire dataset

    Inputs:
        - model: torch model
        - loss_fn: torch loss function: should use sum loss and not mean loss as its helpful to do mini batch calculations
        - batch: (x, y) example batch
        - topk: number of eigenvalues requires
        - P: pre-conditioner for the Hessian
    
    Output: 
        - eigvals: top topk eigenvalues
        - eigvecs: top topk eigenvectors
    Remark: If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
    """

    nparams = len(parameters_to_vector((model.parameters())))

    def matvec(vec: np.ndarray):
        " Takes a numpy array, pushes it to GPU; computes hvp; and returns it to cpu again"
        return compute_hvp(ctx = ctx, model = model, loss_fn = loss_fn, batch = batch, vec = vec, P = P).detach().cpu()
    

    # create and operator for the eigsh function
    operator = linalg.LinearOperator((nparams, nparams), matvec = matvec) # matvec is a callable function that returns H | vector >

    # compute eigenvalues and eigenvector
    eigvals = linalg.eigsh(operator, k = topk, tol = 1e-09, return_eigenvectors = False)

    # reverse the order of eigvals and eigvecs
    eigvals = eigvals[::-1]

    return eigvals