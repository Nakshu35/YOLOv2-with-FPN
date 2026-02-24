import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Darknet_FPN import Darknet19_FPN


class YOLOv2_FPN_Model(nn.Module):
    def __init__(self, num_classes, anchors):
        super().__init__()

        self.num_classes = num_classes

        # assume 9 anchors
        self.anchors_52 = anchors[:3]
        self.anchors_26 = anchors[3:6]
        self.anchors_13 = anchors[6:]

        self.backbone = Darknet19_FPN()

        # Lateral reductions (VERY IMPORTANT)
        self.lat1 = nn.Conv2d(1024, 256, 1)  # for 26x26
        self.lat2 = nn.Conv2d(256, 128, 1)   # for 52x52

        out_13 = len(self.anchors_13) * (5 + num_classes)
        out_26 = len(self.anchors_26) * (5 + num_classes)
        out_52 = len(self.anchors_52) * (5 + num_classes)

        self.head_13 = nn.Conv2d(1024, out_13, 1)
        self.head_26 = nn.Conv2d(256, out_26, 1)
        self.head_52 = nn.Conv2d(128, out_52, 1)

    def reshape(self, x, num_anchors):
        B, _, H, W = x.shape
        x = x.view(B, num_anchors, 5 + self.num_classes, H, W)
        x = x.permute(0, 3, 4, 1, 2)
        return x

    def forward(self, x):

        c3, c4, c5 = self.backbone(x)

        # 13x13 head
        p13 = self.head_13(c5)

        # 26x26 head
        up1 = F.interpolate(self.lat1(c5), scale_factor=2, mode="nearest")
        p26_feat = c4 + up1
        p26 = self.head_26(p26_feat)

        # 52x52 head
        up2 = F.interpolate(self.lat2(p26_feat), scale_factor=2, mode="nearest")
        p52_feat = c3 + up2
        p52 = self.head_52(p52_feat)

        p13 = self.reshape(p13, len(self.anchors_13))
        p26 = self.reshape(p26, len(self.anchors_26))
        p52 = self.reshape(p52, len(self.anchors_52))

        return p13, p26, p52