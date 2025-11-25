import torch.nn as nn
from torch import Tensor

class MSELoss(nn.Module):
    " The OG MSE loss "
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x: Tensor, y: Tensor):
        return 0.5 * ((x - y) ** 2).mean()

