import torch

def collate_fn(batch):

    images = []
    targets = []

    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)

    images = torch.stack(images, dim=0)

    return images, targets