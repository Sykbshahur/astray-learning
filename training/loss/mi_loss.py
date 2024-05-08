import torch
from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="mi_loss")
class MIloss(AbstractLossClass):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label=None):
        pred_xy = pred[0]
        pred_x_y = pred[1]
        # print
        # loss = (torch.log(pred_xy).mean() +
        #         torch.log(1 - pred_x_y).mean())
        epsilon = 1e-7
        clamped_pred_xy = torch.clamp(pred_xy, epsilon, 1-epsilon)
        clamped_pred_x_y = torch.clamp(1 - pred_x_y, epsilon, 1-epsilon)

        loss = (torch.log(clamped_pred_xy).mean() +
                torch.log(clamped_pred_x_y).mean())

        # print(loss, 'mi,loss')
        return loss
