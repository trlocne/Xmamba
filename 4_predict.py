from xmamba.preprocessing import test_get_loader, get_loader
from xmamba.xmamba import XMamba
from xmamba.train_model import TrainXMamba
import torch
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt

idbase = "GLI-02063-105"
model = XMamba(in_chans=4,
                 out_chans=4,
                 depths=[2,2,2,2],
                 feat_size=[64, 128, 256, 512]).cuda()

test_loader, test_ds = test_get_loader(idir=idbase)
trainloader, val_loader, train_ds, val_ds = get_loader()
device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.no_grad():
    for batch_data in test_loader:
        image = batch_data["image"].to(device)
        prob = torch.sigmoid(TrainXMamba.inference(image))
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 3
        seg_out[seg[3] == 1] = 4

slice_num = 90

img_add = os.path.join(f"/kaggle/input/BraTS2024_small_dataset/BraTS-{idbase}/BraTS-{idbase}-t1c.nii")
label_add = os.path.join(f"/kaggle/input/BraTS2024_small_dataset/BraTS-{idbase}/BraTS-{idbase}-seg.nii")

img = nib.load(img_add).get_fdata()
label = nib.load(label_add).get_fdata()

plt.figure("image", (18, 6))
plt.subplot(1, 3, 1)
plt.title("image")

plt.imshow(img[:, :, slice_num], cmap="gray")
plt.subplot(1, 3, 2)
plt.title("label")

plt.imshow(label[:, :, slice_num])
plt.subplot(1, 3, 3)
plt.title("segmentation")

plt.imshow(seg_out[:, :, slice_num])
plt.show()