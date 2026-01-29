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
from torch import Tensor
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import parameters_to_vector
from scipy.sparse.linalg import LinearOperator, eigsh

DEFAULT_BATCH_SIZE = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def iterate_dataset(dataset: Dataset, batch_size: int):
    """ Generic Dataloader """
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    for (x, y) in loader:
        yield x.to(device), y.to(device)

## HVP functions ###

def compute_hvp_dataset(model: nn.Module, loss_fn: nn.Module, dataset: Dataset, vector: Tensor, batch_size: int = DEFAULT_BATCH_SIZE, P: Tensor = None):
    """
    Computes Hessian-vector product (hvp = H v) for the entire dataset given a vector v
    hvp = H v

    Inputs:
        - model: torch model
        - loss_fn: torch loss function: should use sum loss and not mean loss as its helpful to do mini batch calculations
        - dataset: torch Dataset and not the DataLoader object
        - vector: vector (v) for computing Hessian vector product
        - batch_size: batch size for computing hvp
        - P: pre-conditioner for the Hessian
    Outputs:
        - hvp = H v
        Remark: If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
    """
    nparams = len(parameters_to_vector(model.parameters()))

    dataset_size = len(dataset)

    hvp = torch.zeros(nparams, dtype = torch.float, device = 'cuda')

    vector = vector.cuda()

    if P is not None:
        vector = vector / P.cuda().sqrt()

    for (x, y) in iterate_dataset(dataset, batch_size):

        loss = loss_fn(model(x), y) / dataset_size # compute the loss and normalize it with the dataset size; NOTE: loss_fn should compute the sum loss and not mean

        grads = torch.autograd.grad(loss, inputs = model.parameters(), create_graph = True) # | grad > 
        dot = parameters_to_vector(grads).mul(vector).sum() # < grad | vector >
        grads = [g.contiguous() for g in torch.autograd.grad(dot, model.parameters(), retain_graph = True)] # < H = d grad | vector >
        hvp += parameters_to_vector(grads) # flatten < H | vector > and add it for every batch

    if P is not None:
        hvp = hvp / P.cuda().sqrt()

    return hvp

def compute_hvp_batch(model: nn.Module, loss_fn: nn.Module, batch: Tuple, vector: Tensor, P: Tensor = None):
    """
    Computes Hessian-vector product (hvp = H v) for the entire dataset given a vector v
    hvp = H v

    Inputs:
        - model: torch model
        - loss_fn: torch loss function: should use sum loss and not mean loss as its helpful to do mini batch calculations
        - batch: (x, y) example batch
        - vector: vector (v) for computing Hessian vector product
        - P: pre-conditioner for the Hessian
    Outputs:
        - hvp = H v
        Remark: If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
    """
    nparams = len(parameters_to_vector(model.parameters()))

    hvp = torch.zeros(nparams, dtype = torch.float, device = 'cuda')

    vector = vector.cuda()

    if P is not None:
        vector = vector / P.cuda().sqrt()

    x, y = batch
    loss = loss_fn(model(x), y)

    grads = torch.autograd.grad(loss, inputs = model.parameters(), create_graph = True) # | grad > 
    dot = parameters_to_vector(grads).mul(vector).sum() # < grad | vector >
    grads = [g.contiguous() for g in torch.autograd.grad(dot, model.parameters(), retain_graph = True)] # < H = d grad | vector >
    hvp += parameters_to_vector(grads) # flatten < H | vector > and add it for every batch

    if P is not None:
        hvp = hvp / P.cuda().sqrt()

    return hvp


### solver algorithm ###

def solver(matrix_vector, dim: int, topk: int):
    """ Eigenvalue solver using Matrix-vector product """

    def mv(vec: np.ndarray):
        # pushes input vector to GPU and calls the matrix vector function
        gpu_vec = torch.tensor(vec, dtype = torch.float).cuda()
        return matrix_vector(gpu_vec)
    
    # create and operator for the eigsh function
    operator = LinearOperator((dim, dim), matvec = mv) # matvec is a callable function that returns H | vector >

    # compute eigenvalues and eigenvector
    evals, evecs = eigsh(operator, k = topk)

    # push the evals and arrays back to GPU; TODO: but why?
    evals = torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float()
    evecs = torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()
    return evals, evecs

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
        - evals: top topk eigenvalues
        - evecs: top topk eigenvectors
    Remark: If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
    """
    def hvp(v):
        return compute_hvp_batch(model, loss_fn, batch, v, P = P).detach().cpu()
    
    nparams = len(parameters_to_vector((model.parameters())))
    evals, evecs = solver(hvp, nparams, topk = topk)
    return evals, evecs


def get_hessian_dataset(model: nn.Module, loss_fn: nn.Module, dataset: Dataset, topk = 3, batch_size = 512, P = None):
    """ 
    Compute the leading Hessian eigenvalues using the entire dataset

    Inputs:
        - model: torch model
        - loss_fn: torch loss function: should use sum loss and not mean loss as its helpful to do mini batch calculations
        - dataset: torch Dataset and not the DataLoader object
        - niegs: number of eigenvalues requires
        - batch_size: batch size for computing hvp
        - P: pre-conditioner for the Hessian
    
    Output: 
        - evals: top topk eigenvalues
        - evecs: top topk eigenvectors
    Remark: If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
    """
    def hvp(v):
        return compute_hvp_dataset(model, loss_fn, dataset, v, batch_size = batch_size, P = P).detach().cpu()
    
    nparams = len(parameters_to_vector((model.parameters())))
    evals, evecs = solver(hvp, nparams, topk = topk)
    return evals, evecs

