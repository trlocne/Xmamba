from xmamba import Xmamba
from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial
import glob
import json
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import torch.nn.functional as F 
from torchinfo import summary
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.data import DataLoader, decollate_batch, Dataset
from monai.config import print_config
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    CropForegroundd,
    Resized,
    SpatialPadd,
    CenterSpatialCropd,
    RandAffined,
    RandGaussianNoised,
    RandAdjustContrastd
)
from tqdm import tqdm

from monai.utils import set_determinism

directory = os.environ.get("MONAI_DATA_DIRECTORY")

if directory is not None:

    os.makedirs(directory, exist_ok=True)

root_dir = tempfile.mkdtemp() if directory is None else directory

print(root_dir)

def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is NETC
    label 2 is SNFH
    label 3 is ET
    label 4 is RC
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 3), d[key] == 2))
            result.append(d[key] == 3)
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).float()
        return d
    
def get_loader(url, output_dir):
    training = {"training": []}
    items = sorted(glob.glob(f"{url}/*")) 
    fold = -1; 
    for i in range(len(items)):   
        if fold > 3:
            fold = -1            
        fold = fold + 1
        if i >= 150:   
            break   
        values = sorted(glob.glob(f"{items[i]}/*"))      
        training["training"].extend([{"fold": fold,"image": values[1:], "label": values[0]}])         
    with open('training_data.json', 'w') as file:      
        json.dump(training, file)
    train_files, validation_files = datafold_read(datalist=f"{output_dir}/training_data.json", basedir='', fold=1)
    
    # Training Transform
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Resized(keys=["image", "label"], spatial_size=[128, 128 , 128]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )

    # Validation Transform
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Resized(keys=["image", "label"], spatial_size=[128, 128 , 128]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )



    train_ds = Dataset(data=train_files, transform=train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_ds = Dataset(data=validation_files, transform=val_transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )



    return train_loader, val_loader, train_ds, val_ds

url = 'BraT24'
output_dir = '.'
train_loader, val_loader, train_ds, val_ds= get_loader(url, output_dir)

max_epochs = 120
val_interval = 1
VAL_AMP = True
roi = (128, 128, 128)

device = torch.device("cuda")
model = SegMamba(in_chans=4,
                 out_chans=4,
                 depths=[2,2,2,2],
                 feat_size=[48, 96, 192, 384]).to(device)


loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.amp.autocast(device_type='cuda'):
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.amp.GradScaler('cuda')
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

def load_checkpoint(filename, model, optimizer=None, lr_scheduler=None):
    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    print(f"Checkpoint loaded: Epoch {checkpoint['epoch']}, Best Accuracy: {checkpoint['best_acc']}")

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    
    return checkpoint["epoch"], checkpoint["best_acc"]

def save_checkpoint(optimizer, lr_scheduler, model, epoch, filename="model.pt", best_acc=0, dir_add="./"):
    state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    lr_scheduler_state_dict = lr_scheduler.state_dict()
    
    save_dict = {
        "epoch": epoch,
        "best_acc": best_acc,
        "state_dict": state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "lr_scheduler_state_dict": lr_scheduler_state_dict,
    }
    
    filename = os.path.join(dir_add, filename)
    torch.save(save_dict, filename)
    print(f"Checkpoint saved at {filename}")

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = [] 
metric_values_et = []
metric_values_rc = []

total_start = time.time()
for epoch in tqdm(range(max_epochs), desc="Training Epochs"):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                
                # Đo lường và tính toán các metrics cho từng lớp
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)


            # Tính tổng Dice score cho tất cả lớp (mean dice)
            metric = dice_metric.aggregate().item()
            metric_values.append(metric)

            # Các giá trị Dice cho từng lớp cụ thể
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item() 
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            metric_rc = metric_batch[3].item()
            metric_values_rc.append(metric_rc)

            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                save_checkpoint(
                    optimizer,
                    lr_scheduler,
                    model,
                    epoch,
                    best_acc=best_metric,
                )
    
                print("saved new best metric model")

            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f} \n"
                f"tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f} rc: {metric_rc:.4f} \n"
                f"Best mean dice: {best_metric:.4f} \n"
                f"at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

import matplotlib.pyplot as plt

# Đoạn vẽ đồ thị cho loss và Dice mean
plt.figure("train", (12, 6))

# Vẽ Epoch Average Loss
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(x, y, color="red")

# Vẽ Val Mean Dice
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.plot(x, y, color="green")

plt.show()

# Đoạn vẽ đồ thị cho các lớp (TC, WT, ET, RC)
plt.figure("train", (18, 6))

# Val Mean Dice TC (Tumor Core)
plt.subplot(1, 4, 1)
plt.title("Val Mean Dice TC")
x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
y = metric_values_tc
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.plot(x, y, color="brown")

# Val Mean Dice WT
plt.subplot(1, 4, 2)
plt.title("Val Mean Dice WT")
x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
y = metric_values_wt
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.plot(x, y, color="purple")

# Val Mean Dice ET
plt.subplot(1, 4, 3)
plt.title("Val Mean Dice ET")
x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
y = metric_values_et
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.plot(x, y, color="orange")

plt.subplot(1, 4, 4)
plt.title("Val Mean Dice RC")
x = [val_interval * (i + 1) for i in range(len(metric_values_rc))]
y = metric_values_rc
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.plot(x, y, color="blue")

plt.show()

import pandas as pd

df_loss = pd.DataFrame({
    "Epoch": range(1, len(epoch_loss_values) + 1),
    "Loss": epoch_loss_values
})
df_loss.to_csv("epoch_loss_values.csv", index=False)

df_metric = pd.DataFrame({
    "Epoch": [val_interval * (i + 1) for i in range(len(metric_values))],
    "Dice Score": metric_values
})
df_metric.to_csv("metric_values.csv", index=False)

classes = {
    "metric_values_tc": metric_values_tc,
    "metric_values_wt": metric_values_wt,
    "metric_values_et": metric_values_et,
    "metric_values_rc": metric_values_rc
}

for class_name, values in classes.items():
    df = pd.DataFrame({
        "Epoch": [val_interval * (i + 1) for i in range(len(values))],
        "Dice Score": values
    })
    df.to_csv(f"{class_name}.csv", index=False)
torch.save(torch.load(os.path.join("model.pt")), "model.pt")