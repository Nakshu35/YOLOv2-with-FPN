import torch.nn as nn
import torch
from Model.Darknet import Darknet19

class YOLOv2_Model(nn.Module):
    def __init__(self, num_classes, anchors):
        super().__init__()

        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.backbone = Darknet19()

        out_channels = self.num_anchors * (5 + num_classes)
        self.pred = nn.Conv2d(1024, out_channels, 1)

    def forward(self, x):

        features = self.backbone(x)
        x = self.pred(features)
        B, _, H, W = x.shape

        x = x.view(B,self.num_anchors,5 + self.num_classes,H,W)
        x = x.permute(0, 3, 4, 1, 2)

        return x
