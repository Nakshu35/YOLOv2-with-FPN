import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

from Data.voc_dataset import VOC_dataset
from Data.collate_fn import collate_fn
from Data.config import S, B, C, Inp_size, Epochs, Batch_size, Lr, Wd
from Data.Target_builder import YOLO_TargetBuilder
from Data.anchor_boxes import extract_box_wh, kmeans

from Loss.loss import YOLO_loss
from Model.YOLOv2_model import YOLOv2_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize((Inp_size, Inp_size)),
    T.ToTensor()
])

root_path = "/home/administrator/Desktop/Nakshatra/VOCdevkit"

Voc_07 = VOC_dataset(
    root = root_path,
    year = "2007",
    image_set="trainval",
    transform = transform
)

Voc_12 = VOC_dataset(
    root = root_path,
    year = "2012",
    image_set="trainval",
    transform = transform
)

dataset = ConcatDataset([Voc_07, Voc_12])

Voc_07_val = VOC_dataset(
    root=root_path,
    year="2007",
    image_set="val",
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=Batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    Voc_07_val,
    batch_size=Batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

#anchors
boxes_07 = np.array(extract_box_wh(Voc_07, Inp_size))
boxes_12 = np.array(extract_box_wh(Voc_12, Inp_size))
all_boxes = np.concatenate([boxes_07, boxes_12], axis=0)
anchors = kmeans(all_boxes, k=B)
anchors = torch.tensor(anchors, dtype=torch.float32).to(device)

target_builder = YOLO_TargetBuilder(
    anchors=anchors,
    S=S,
    B=B,
    C=C,
    input_size=Inp_size
)

model = YOLOv2_Model(
    num_classes=C,
    anchors=anchors
).to(device)

criterion = YOLO_loss()
optimizer = optim.SGD(model.parameters(), lr=Lr, weight_decay=Wd, momentum=0.9)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

train_losses = []
val_losses = []
best_val_loss = float("inf")


for epoch in range(Epochs):
    # Train
    model.train()
    epoch_train_loss = 0.0

    loop = tqdm(dataloader, total=len(dataloader),
                desc=f"Epoch [{epoch+1}/{Epochs}]")

    for batch_idx, (images, targets) in enumerate(loop):

        images = images.to(device)
        batch_targets = []

        for i in range(len(images)):
            yolo_target = target_builder.build_target(
                targets[i]["boxes"],
                targets[i]["labels"]
            )
            batch_targets.append(yolo_target)

        batch_targets = torch.stack(batch_targets).to(device)

        preds = model(images)
        loss = criterion(preds, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        loop.set_postfix(
            train_loss=loss.item(),
            avg_train_loss=epoch_train_loss / (batch_idx + 1)
        )

    epoch_train_loss /= len(dataloader)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:

            images = images.to(device)
            batch_targets = []

            for i in range(len(images)):
                yolo_target = target_builder.build_target(
                    targets[i]["boxes"],
                    targets[i]["labels"]
                )
                batch_targets.append(yolo_target)

            batch_targets = torch.stack(batch_targets).to(device)

            preds = model(images)
            loss = criterion(preds, batch_targets)

            epoch_val_loss += loss.item()

    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"\nEpoch [{epoch+1}/{Epochs}] "
          f"Train Loss: {epoch_train_loss:.4f} "
          f"Val Loss: {epoch_val_loss:.4f}")

    # Model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(),
                   os.path.join(checkpoint_dir, "best_model.pth"))
        print("Best model saved.")

# Save last model
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last_model.pth"))

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