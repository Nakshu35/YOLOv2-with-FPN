import torch
import torch.nn as nn


class YOLO_loss_FPN(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def single_scale_loss(self, preds, target):

        device = preds.device
        batch_size = preds.size(0)

        obj_mask = target[..., 4] == 1
        noobj_mask = target[..., 4] == 0

        total_loss = torch.tensor(0.0, device=device)

        # ------------------ Localization ------------------
        if obj_mask.any():

            pred_box = preds[..., 0:4][obj_mask]
            target_box = target[..., 0:4][obj_mask]

            # Clamp predictions to prevent explosion
            pred_box = torch.clamp(pred_box, -10, 10)

            loc_loss = self.lambda_coord * self.mse(pred_box, target_box)
            total_loss += loc_loss

        # ------------------ Objectness ------------------
        if obj_mask.any():
            obj_loss = self.bce(
                preds[..., 4][obj_mask],
                target[..., 4][obj_mask]
            )
            total_loss += obj_loss

        # ------------------ No Object ------------------
        if noobj_mask.any():
            noobj_loss = self.bce(
                preds[..., 4][noobj_mask],
                target[..., 4][noobj_mask]
            )
            total_loss += self.lambda_noobj * noobj_loss

        # ------------------ Classification ------------------
        if obj_mask.any():
            pred_cls = preds[..., 5:][obj_mask]
            target_cls = target[..., 5:][obj_mask]

            cls_loss = self.bce(pred_cls, target_cls)
            total_loss += cls_loss

        # Normalize ONLY by batch (paper-like)
        total_loss = total_loss / batch_size

        return total_loss

    def forward(self, p13, p26, p52,
                t13, t26, t52):

        loss_13 = self.single_scale_loss(p13, t13)
        loss_26 = self.single_scale_loss(p26, t26)
        loss_52 = self.single_scale_loss(p52, t52)

        return loss_13 + loss_26 + loss_52