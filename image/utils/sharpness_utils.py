# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility file for estimating the Hessian spectrum using Lanzcos. 
Code adopted from https://github.com/locuslab/edge-of-stability/
"""

# imports
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from torch import Tensor
from typing import Tuple

from scipy.sparse.linalg import LinearOperator, eigsh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_hvp(model: nn.Module, loss_fn: nn.Module, batch: Tuple, vector: Tensor, P: Tensor = None):
    """
    Computes Hessian-vector product (hvp = H v) for the entire dataset given a vector v
    hvp = H v

    Inputs:
        - model: torch model
        - loss_fn: torch loss function: should use sum loss and not mean loss as its helpful to do mini batch calculations
        - batch: (x, y) tuple of inputs and outputs 
        - vector: vector (v) for computing Hessian vector product; should be on cuda
        - batch_size: batch size for computing hvp
        TODO: Add pre-conditioner
        - P: pre-conditioner for the Hessian
    Outputs:
        - hvp = H v
        Remark: If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
    """

    vector = vector.to(device)

    if P is not None:
        vector = vector / P.cuda().sqrt()

    x, y = batch
    loss = loss_fn(model(x), y)

    # compute the gradients
    grads = torch.autograd.grad(loss, inputs = model.parameters(), create_graph = True) # | grad > 
    
    gv_prod = torch.dot(parameters_to_vector(grads), vector) # gradient vector product < grad | vector >
    
    hvp = torch.autograd.grad(gv_prod, model.parameters(), retain_graph = True)

    flat_hvp = torch.cat([h.view(-1) for h in hvp])
    
    return flat_hvp

    
def get_hessian_batch(model: nn.Module, loss_fn: nn.Module, batch: Tuple, topk = 3, P = None):
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

    def matvec(v: np.ndarray):
        " Takes a numpy array, pushes it to GPU; computes hvp; and returns it to cpu again"
        v = torch.tensor(v, dtype = torch.float).cuda()
        return compute_hvp(model, loss_fn, batch, v, P = P).detach().cpu()
    

    # create and operator for the eigsh function
    operator = LinearOperator((nparams, nparams), matvec = matvec) # matvec is a callable function that returns H | vector >

    # compute eigenvalues and eigenvector
    eigvals, eigvecs = eigsh(operator, k = topk)

    # reverse the order of eigvals and eigvecs
    eigvals = eigvals[::-1]
    eigvecs = np.flip(eigvecs, -1)

    return eigvals, eigvecs