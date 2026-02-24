import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YOLO_TargetBuilder_FPN:
    def __init__(self, anchors, C=20, input_size=416):

        # assume 9 anchors total
        anchors = torch.tensor(anchors, dtype=torch.float32)

        self.anchors_52 = anchors[:3].to(device)   # small objects
        self.anchors_26 = anchors[3:6].to(device)  # medium
        self.anchors_13 = anchors[6:].to(device)   # large

        self.C = C
        self.input_size = input_size

    def build_empty(self, S, B):
        return torch.zeros((S, S, B, 5 + self.C), device=device)

    def assign_box(self, box, label, anchors, S, target):

        xmin, ymin, xmax, ymax = box

        w = xmax - xmin
        h = ymax - ymin
        cx = xmin + w / 2
        cy = ymin + h / 2

        cx /= self.input_size
        cy /= self.input_size
        w /= self.input_size
        h /= self.input_size

        grid_x = int(cx * S)
        grid_y = int(cy * S)

        grid_x = min(grid_x, S - 1)
        grid_y = min(grid_y, S - 1)

        box_wh = torch.tensor([w, h], device=device)

        ious = []
        for anchor in anchors:
            min_w = torch.min(anchor[0], box_wh[0])
            min_h = torch.min(anchor[1], box_wh[1])

            inter = min_w * min_h
            union = anchor[0]*anchor[1] + box_wh[0]*box_wh[1] - inter
            ious.append(inter / union)

        best_anchor = torch.argmax(torch.stack(ious))

        target[grid_y, grid_x, best_anchor, 0] = cx * S - grid_x
        target[grid_y, grid_x, best_anchor, 1] = cy * S - grid_y
        target[grid_y, grid_x, best_anchor, 2] = torch.log(
            w / anchors[best_anchor][0] + 1e-6
        )
        target[grid_y, grid_x, best_anchor, 3] = torch.log(
            h / anchors[best_anchor][1] + 1e-6
        )

        target[grid_y, grid_x, best_anchor, 4] = 1.0
        target[grid_y, grid_x, best_anchor, 5 + label] = 1.0

    def build_target(self, boxes, labels):

        t13 = self.build_empty(13, 3)
        t26 = self.build_empty(26, 3)
        t52 = self.build_empty(52, 3)

        for box, label in zip(boxes, labels):

            xmin, ymin, xmax, ymax = box
            area = (xmax - xmin) * (ymax - ymin)

            # simple scale routing
            if area > (96 ** 2):
                self.assign_box(box, label,
                                self.anchors_13, 13, t13)

            elif area > (32 ** 2):
                self.assign_box(box, label,
                                self.anchors_26, 26, t26)

            else:
                self.assign_box(box, label,
                                self.anchors_52, 52, t52)

        return t13, t26, t52