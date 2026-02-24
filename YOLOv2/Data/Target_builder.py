import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class YOLO_TargetBuilder:
    def __init__(self, anchors, S=13, B=5, C=20, input_size=416):

        self.anchors = torch.tensor(anchors, dtype=torch.float32).to(device)
        self.S = S
        self.B = B
        self.C = C
        self.input_size = input_size

    def build_target(self, boxes, labels):
        target = torch.zeros((self.S, self.S, self.B, 5 + self.C), device=device)

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box

            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w / 2
            cy = ymin + h / 2

            cx /= self.input_size
            cy /= self.input_size
            w /= self.input_size
            h /= self.input_size

            grid_x = int(cx * self.S)
            grid_y = int(cy * self.S)
            # Safety clamp (rare boundary case)
            grid_x = min(grid_x, self.S - 1)
            grid_y = min(grid_y, self.S - 1)

            box_wh = torch.tensor([w, h], device=device)
            ious = []

            for anchor in self.anchors:

                min_w = torch.min(anchor[0], box_wh[0])
                min_h = torch.min(anchor[1], box_wh[1])

                intersection = min_w * min_h
                union = anchor[0]*anchor[1] + box_wh[0]*box_wh[1] - intersection

                iou = intersection / union
                ious.append(iou)

            best_anchor = torch.argmax(torch.tensor(ious))

            target[grid_y, grid_x, best_anchor, 0] = cx * self.S - grid_x
            target[grid_y, grid_x, best_anchor, 1] = cy * self.S - grid_y
            target[grid_y, grid_x, best_anchor, 2] = torch.log(w / self.anchors[best_anchor][0] + 1e-6)
            target[grid_y, grid_x, best_anchor, 3] = torch.log(h / self.anchors[best_anchor][1] + 1e-6)

            target[grid_y, grid_x, best_anchor, 4] = 1.0
            target[grid_y, grid_x, best_anchor, 5 + label] = 1.0

        return target