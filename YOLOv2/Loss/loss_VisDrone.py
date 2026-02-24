import torch
import torch.nn as nn
from Data.config_VisDrone import S, B, C

class YOLO_loss(nn.Module):
    def __init__(self, lambda_coord = 5, lambda_noObj = 0.5):
        super().__init__()
        self.lamda_coord = lambda_coord
        self.lambda_noObj = lambda_noObj
        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, preds, target):
        obj_mask = target[..., 4] == 1
        noObj_mask = target[..., 4] == 0

        total_loss = 0

        # Localization
        if obj_mask.sum() > 0:
            pred_boxes = preds[..., 0:4][obj_mask]
            target_boxes = target[..., 0:4][obj_mask]

            localization_loss = self.lamda_coord * self.mse(pred_boxes, target_boxes)
        else:
            localization_loss = torch.tensor(0.0, device=preds.device)

        # Confidence
        obj_loss = self.bce(preds[..., 4][obj_mask], target[..., 4][obj_mask]) \
            if obj_mask.sum() > 0 else torch.tensor(0.0, device=preds.device)

        noObj_loss = self.bce(preds[..., 4][noObj_mask], target[..., 4][noObj_mask]) \
            if noObj_mask.sum() > 0 else torch.tensor(0.0, device=preds.device)

        confidence_loss = obj_loss + self.lambda_noObj * noObj_loss

        # Classification
        if obj_mask.sum() > 0:
            pred_cls = preds[..., 5:5+self.C][obj_mask]
            target_cls = target[..., 5:5+self.C][obj_mask]

            classification_loss = self.bce(pred_cls, target_cls)
        else:
            classification_loss = torch.tensor(0.0, device=preds.device)

        total_loss = localization_loss + confidence_loss + classification_loss

        # normalize by batch size
        total_loss = total_loss / preds.size(0)

        return total_loss