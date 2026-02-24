import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

class VOC_dataset(Dataset):
    def __init__(self, root, year="2007", image_set = "train", transform = None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform

        self.voc_root = os.path.join(root, f"VOC{year}")
        self.image_dir = os.path.join(self.voc_root, "JPEGImages")
        self.ann_dir = os.path.join(self.voc_root, "Annotations")

        split_file = os.path.join(self.voc_root, "ImageSets", "Main", f"{image_set}.txt")

        with open(split_file) as f:
            self.ids = [line.strip() for line in f.readlines()]

        self.class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        image_id = self.ids[index]

        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        annotation_path = os.path.join(self.ann_dir, image_id + ".xml")
        
        image = Image.open(image_path).convert("RGB")
        boxes, labels = self.parse_annotations(annotation_path)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id
        }

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


    def parse_annotations(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self.class_to_idx:
                continue

            label = self.class_to_idx[name]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels    