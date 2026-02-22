import torch
import torch.nn as nn


class YOLO_loss_FPN(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noObj=0.5):
        super().__init__()
        self.lamda_coord = lambda_coord
        self.lambda_noObj = lambda_noObj
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def single(self, preds, target):

        obj = target[..., 4] == 1
        noobj = target[..., 4] == 0

        loc = self.lamda_coord * self.mse(
            preds[..., 0:4][obj],
            target[..., 0:4][obj]
        )

        obj_loss = self.bce(
            preds[..., 4][obj],
            target[..., 4][obj]
        )

        noobj_loss = self.bce(
            preds[..., 4][noobj],
            target[..., 4][noobj]
        )

        cls = self.bce(
            preds[..., 5:][obj],
            target[..., 5:][obj]
        )

        return loc + obj_loss + self.lambda_noObj * noobj_loss + cls

    def forward(self, p13, p26, p52, t13, t26, t52):
        return (
            self.single(p13, t13) +
            self.single(p26, t26) +
            self.single(p52, t52)
        )