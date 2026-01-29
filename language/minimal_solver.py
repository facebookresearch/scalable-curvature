# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from linear_operator.operators import LinearOperator
import scipy.sparse.linalg as linalg

def compute_hvp(vec: Tensor):
    "vec should be a vector of shape (n, k) and on the GPU"
    if not vec.is_cuda:
        vec = vec.cuda()
    n = vec.shape[0]
    H = torch.randn(n, n).cuda()
    hvp = H @ vec  # Simulating a Hessian-vector product
    return hvp

def get_hessian_torch(nparams = 1000, topk = 1, tol = 1e-09, niter = 100):
    
    def matvec(vec):
        " Takes a numpy array, pushes it to GPU; computes hvp; and returns it to cpu again"
        return compute_hvp(vec = vec)
    
    # create and operator for the eigsh function
    operator = LinearOperator(shape = (nparams, nparams), matmul = matvec) # matvec is a callable function that returns H | vector >

    # compute eigenvalues and eigenvector
    eigvals = torch.lobpcg(operator, k = topk, tol = tol, niter = niter)

    return eigvals

def get_hessian_scipy(nparams = 1000, topk = 1, tol = 1e-09, niter = 100):
    
    def matvec(vec):
        " Takes a numpy array, pushes it to GPU; computes hvp; and returns it to cpu again"
        vec = torch.tensor(vec, dtype = torch.float).cuda()
        return compute_hvp(vec = vec).detach().cpu()
    
    # create and operator for the eigsh function
    operator = linalg.LinearOperator((nparams, nparams), matvec = matvec) # matvec is a callable function that returns H | vector >

    # compute eigenvalues and eigenvector
    eigvals = linalg.eigsh(operator, k = topk, return_eigenvectors = False) # lanzcos solver

    return eigvals


eigvals = get_hessian_scipy(nparams = 1000)
print(eigvals)

eigvals = get_hessian_torch(nparams = 1000)
