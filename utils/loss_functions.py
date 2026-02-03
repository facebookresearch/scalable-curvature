import torch.nn as nn
from torch.nn import functional as F

class CrossEntropyLoss(nn.Module):
    " Flattens the logits and targets before loss fn for Language Modeling "
    def __init__(self,):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = -1)
