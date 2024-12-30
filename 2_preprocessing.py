from xmamba.preprocessing import get_loader
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

train_loader, val_loader, train_ds, val_ds= get_loader("/kaggle/input/brats2024-small-dataset/BraTS2024_small_dataset/")

img_add = os.path.join("/kaggle/input/brats2024-small-dataset/BraTS2024_small_dataset/BraTS-GLI-02063-105/BraTS-GLI-02063-105-t1c.nii")

label_add = os.path.join("/kaggle/input/brats2024-small-dataset/BraTS2024_small_dataset/BraTS-GLI-02063-105/BraTS-GLI-02063-105-seg.nii")

img = nib.load(img_add).get_fdata()

label = nib.load(label_add).get_fdata()

print(f"image shape: {img.shape}, label shape: {label.shape}")

plt.figure("image", (18, 6))

plt.subplot(1, 2, 1)

plt.title("image")

plt.imshow(img[:, :, 135], cmap="gray")

plt.subplot(1, 2, 2)

plt.title("label")

plt.imshow(label[:, :, 135])

plt.show()