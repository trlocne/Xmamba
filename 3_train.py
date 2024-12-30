from xmamba.preprocessing import get_loader
from xmamba.xmamba import XMamba
import os
from xmamba.train_model import TrainXMamba

train_loader, val_loader, train_ds, val_ds= get_loader("/kaggle/input/brats2024-small-dataset/BraTS2024_small_dataset/")
model = XMamba(in_chans=4,
                 out_chans=4,
                 depths=[2,2,2,2],
                 feat_size=[64, 128, 256, 512]).cuda()
train = TrainXMamba(max_epochs=400)
train.train(train_loader=train_loader, val_loader=val_loader)
train.plot()
