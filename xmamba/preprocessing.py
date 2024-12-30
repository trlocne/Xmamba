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

from monai.utils import set_determinism
import torch
import glob
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


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

def get_loader(basedir = '/kaggle/input/brats2024-small-dataset/BraTS2024_small_dataset', data_list = 'training_data.json'):
    training = {"training": []}
    items = sorted(glob.glob(f"{basedir}/*")) 
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
    train_files, validation_files = datafold_read(datalist=f"{data_list}", basedir='', fold=1)
    
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

def test_get_loader(idir='GLI-02063-105'):
    # Các modal cần xử lý
    modalities = ["t1n", "t1c", "t2f", "t2w"]

    # Tạo danh sách test_files tự động
    test_files = [
        {
            "image": [
                f"/kaggle/input/BraTS2024_small_dataset/BraTS-{idir}/BraTS-{idir}-{modality}.nii"
                for modality in modalities
            ],
            "label": f"/kaggle/input/BraTS2024_small_dataset/BraTS-{idir}/BraTS-{idir}-seg.nii",
        }
    ]

    test_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    test_ds = Dataset(data=test_files, transform=test_transform)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return test_loader, test_ds