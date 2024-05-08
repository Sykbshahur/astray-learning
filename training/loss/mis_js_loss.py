import torch.nn as nn
from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC
import torch.nn.functional as F
import torch

@LOSSFUNC.register_module(module_name="misjs_loss")
class misJSDLoss(AbstractLossClass):
    def __init__(self):
        super(misJSDLoss, self).__init__()


    def forward(self, af, bf, sf, df):
        pos = F.pairwise_distance(df, af)
        neg = F.pairwise_distance(df, bf)
        res = F.pairwise_distance(sf, df)

        clamp = torch.clamp(res + pos - neg + 1, min = 0.0)
        loss = torch.mean(clamp)

        return loss