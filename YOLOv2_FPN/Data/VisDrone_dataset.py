import os
import torch
from torch.utils.data import Dataset
from PIL import Image

VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]


class VisDroneDataset(Dataset):
    def __init__(self, root, split="train", transform=None):

        self.root = root
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, f"VisDrone2019-DET-{split}", "images")
        self.ann_dir = os.path.join(root, f"VisDrone2019-DET-{split}", "annotations")

        self.images = sorted(os.listdir(self.image_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(VISDRONE_CLASSES)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)

        ann_path = os.path.join(
            self.ann_dir,
            img_name.replace(".jpg", ".txt")
        )

        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        with open(ann_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split(",")

            x = float(parts[0])
            y = float(parts[1])
            w = float(parts[2])
            h = float(parts[3])
            category = int(parts[5])

            # ignore category 0
            if category == 0 or category > 10:
                continue

            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(category - 1)  # shift index

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform:

            orig_w, orig_h = image.size
            image = self.transform(image)

            new_h, new_w = image.shape[1:]

            scale_x = new_w / orig_w
            scale_y = new_h / orig_h

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target