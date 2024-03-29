import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from typing import List


class CrossEntropyLossWrapper(CrossEntropyLoss):
    def __init__(self, weight: List = None):
        if weight is not None:
            super().__init__(weight=torch.tensor(weight))
        else:
            super().__init__()

    def forward(self, logits, target, **batch):
        return super().forward(logits, target)
