# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch import Tensor

class MSELoss(nn.Module):
    " The OG MSE loss "
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x: Tensor, y: Tensor):
        return 0.5 * ((x - y) ** 2).mean()

