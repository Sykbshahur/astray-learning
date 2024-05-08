import torch.nn as nn
import torch
from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="labelSmoothLoss")
class labelSmoothLoss(AbstractLossClass):
    def __init__(self, smoothing = 0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = 2
        self.dim = -1

    def forward(self, inputs, targets):
        inputs = inputs.log_softmax(dim = self.dim)
        with torch.no_grad():
            #创建软标签
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * inputs, dim=self.dim))
