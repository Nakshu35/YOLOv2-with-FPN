import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from Data.voc_dataset import VOC_dataset
from Data.collate_fn import collate_fn
from Data.config import C, Inp_size, Epochs, Batch_size, Lr, Wd
from Data.anchor_boxes import extract_box_wh, kmeans

from Data.Target_builder import YOLO_TargetBuilder_FPN
from Loss.loss import YOLO_loss_FPN
from Model.YOLOv2_model_FPN import YOLOv2_FPN_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize((Inp_size, Inp_size)),
    T.ToTensor()
])

root_path = "D:\\DLCV_AI\\Project\\YOLOv2\\VOCdevkit\\VOC"

# Dataset
Voc_07 = VOC_dataset(root=root_path, year="2007",
                     image_set="trainval", transform=transform)

Voc_12 = VOC_dataset(root=root_path, year="2012",
                     image_set="trainval", transform=transform)

dataset = ConcatDataset([Voc_07, Voc_12])

Voc_07_val = VOC_dataset(root=root_path,
                         year="2007",
                         image_set="val",
                         transform=transform)

dataloader = DataLoader(dataset,
                        batch_size=Batch_size,
                        shuffle=True,
                        collate_fn=collate_fn)

val_loader = DataLoader(Voc_07_val,
                        batch_size=Batch_size,
                        shuffle=False,
                        collate_fn=collate_fn)

# Anchors
boxes_07 = extract_box_wh(Voc_07, Inp_size)
boxes_12 = extract_box_wh(Voc_12, Inp_size)
all_boxes = np.concatenate([boxes_07, boxes_12], axis=0)
anchors = kmeans(all_boxes, k=9)
anchors = torch.tensor(anchors, dtype=torch.float32).to(device)

# Model
target_builder = YOLO_TargetBuilder_FPN(
    anchors=anchors,
    C=C,
    input_size=Inp_size
)

model = YOLOv2_FPN_Model(C, anchors).to(device)
criterion = YOLO_loss_FPN()
optimizer = optim.SGD(model.parameters(), lr=Lr, weight_decay=Wd, momentum=0.9)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

train_losses = []
val_losses = []
best_val_loss = float("inf")

# Training
for epoch in range(Epochs):

    model.train()
    epoch_train_loss = 0.0

    loop = tqdm(dataloader,
                total=len(dataloader),
                desc=f"Epoch [{epoch+1}/{Epochs}]")

    for batch_idx, (images, targets) in enumerate(loop):

        images = images.to(device)

        t13_list, t26_list, t52_list = [], [], []

        for i in range(len(images)):
            t13, t26, t52 = target_builder.build_target(
                targets[i]["boxes"],
                targets[i]["labels"]
            )
            t13_list.append(t13)
            t26_list.append(t26)
            t52_list.append(t52)

        t13 = torch.stack(t13_list)
        t26 = torch.stack(t26_list)
        t52 = torch.stack(t52_list)

        p13, p26, p52 = model(images)

        loss = criterion(p13, p26, p52,
                         t13, t26, t52)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        epoch_train_loss += loss.item()

        loop.set_postfix(
            train_loss=loss.item(),
            avg_train_loss=epoch_train_loss/(batch_idx+1)
        )

    epoch_train_loss /= len(dataloader)
    train_losses.append(epoch_train_loss)

    # Validation

    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:

            images = images.to(device)

            t13_list, t26_list, t52_list = [], [], []

            for i in range(len(images)):
                t13, t26, t52 = target_builder.build_target(
                    targets[i]["boxes"],
                    targets[i]["labels"]
                )
                t13_list.append(t13)
                t26_list.append(t26)
                t52_list.append(t52)

            t13 = torch.stack(t13_list)
            t26 = torch.stack(t26_list)
            t52 = torch.stack(t52_list)

            p13, p26, p52 = model(images)

            loss = criterion(p13, p26, p52,
                             t13, t26, t52)

            epoch_val_loss += loss.item()

    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"\nEpoch [{epoch+1}/{Epochs}] "f"Train Loss: {epoch_train_loss:.4f} " f"Val Loss: {epoch_val_loss:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(),os.path.join(checkpoint_dir, "best_model.pth"))
        print("Best model saved.")

# Save last
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last_model.pth"))

# Loss Graph
plt.figure()
plt.plot(range(1, Epochs+1), train_losses, label="Train Loss")
plt.plot(range(1, Epochs+1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(checkpoint_dir, "loss_curve.png"))
plt.close()