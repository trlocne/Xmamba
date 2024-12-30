import torch 
from xmamba.xmamba import XMamba
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

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import time
from tqdm import tqdm


class TrainXMamba():
    def __init__(self, 
                 max_epochs=400, 
                 val_interval=1, 
                 VAL_AMP=True, 
                 roi=(128, 128, 128), 
                 device = "cuda", 
                 in_chans=4, 
                 out_chans=4, 
                 depths=[2,2,2,2], 
                 feat_size=[64, 128, 256, 512]):
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.VAL_AMP = VAL_AMP
        self.roi = roi
        self.device = torch.device(device)
        self.model = XMamba(in_chans=in_chans, out_chans=out_chans, depths=depths, feat_size=feat_size).to(self.device)
        self.loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4, weight_decay=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)

        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.scaler = torch.amp.GradScaler('cuda')
        torch.backends.cudnn.benchmark = True
    
    @staticmethod
    def inference(self,input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=self.roi,
                sw_batch_size=1,
                predictor=self.model,
                overlap=0.5,
            )

        if self.VAL_AMP:
            with torch.amp.autocast(device_type='cuda'):
                return _compute(input)
        else:
            return _compute(input)
    
    @staticmethod
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

    def train(self, train_loader, val_loader):
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.best_metrics_epochs_and_time = [[], [], []]
        self.epoch_loss_values = []
        self.metric_values = []
        self.metric_values_tc = []
        self.metric_values_wt = []
        self.metric_values_et = []
        self.metric_values_rc = []

        self.total_start = time.time()
        for epoch in tqdm(range(0, self.max_epochs), desc="Training Epochs"):
            epoch_start = time.time()
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.max_epochs}")
            self.model.train()
            self.epoch_loss = 0
            self.step = 0
            for batch_data in train_loader:
                self.step += 1
                inputs, labels = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                )
                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12)
            self.lr_scheduler.step()
            self.epoch_loss /= self.step
            self.epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % self.val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(self.device),
                            val_data["label"].to(self.device),
                        )
                        val_outputs = self.device(val_inputs)
                        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
                        
                        # Đo lường và tính toán các metrics cho từng lớp
                        self.dice_metric(y_pred=val_outputs, y=val_labels)
                        self.dice_metric_batch(y_pred=val_outputs, y=val_labels)

                    # Tính tổng Dice score cho tất cả lớp (mean dice)
                    self.metric = self.dice_metric.aggregate().item()
                    self.metric_values.append(self.metric)

                    # Các giá trị Dice cho từng lớp cụ thể
                    self.metric_batch = self.dice_metric_batch.aggregate()
                    self.metric_tc = self.metric_batch[0].item() 
                    self.metric_values_tc.append(self.metric_tc)
                    self.metric_wt = self.metric_batch[1].item()
                    self.metric_values_wt.append(self.metric_wt)
                    self.metric_et = self.metric_batch[2].item()
                    self.metric_values_et.append(self.metric_et)
                    self.metric_rc = self.metric_batch[3].item()
                    self.metric_values_rc.append(self.metric_rc)

                    self.dice_metric.reset()
                    self.dice_metric_batch.reset()

                    if self.metric > best_metric:
                        best_metric = self.metric
                        best_metric_epoch = epoch + 1
                        self.best_metrics_epochs_and_time[0].append(best_metric)
                        self.best_metrics_epochs_and_time[1].append(best_metric_epoch)
                        self.best_metrics_epochs_and_time[2].append(time.time() - self.total_start)
                        self.save_checkpoint(
                            self.optimizer,
                            self.lr_scheduler,
                            self.model,
                            epoch,
                            best_acc=best_metric,
                        )
            
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {self.metric:.4f} \n"
                        f"tc: {self.metric_tc:.4f} wt: {self.metric_wt:.4f} et: {self.metric_et:.4f} rc: {self.metric_rc:.4f} \n"
                        f"Best mean dice: {best_metric:.4f} \n"
                        f"at epoch: {best_metric_epoch}"
                    )
            print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
        total_time = time.time() - self.total_start
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    
    def plot(self):
        # Đoạn vẽ đồ thị cho loss và Dice mean
        plt.figure("train", (12, 6))

        # Vẽ Epoch Average Loss
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(self.epoch_loss_values))]
        y = self.epoch_loss_values
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(x, y, color="red")

        # Vẽ Val Mean Dice
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [self.val_interval * (i + 1) for i in range(len(self.metric_values))]
        y = self.metric_values
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.plot(x, y, color="green")

        plt.show()

        # Đoạn vẽ đồ thị cho các lớp (TC, WT, ET, RC)
        plt.figure("train", (18, 6))

        # Val Mean Dice TC (Tumor Core)
        plt.subplot(1, 4, 1)
        plt.title("Val Mean Dice TC")
        x = [self.val_interval * (i + 1) for i in range(len(self.metric_values_tc))]
        y = self.metric_values_tc
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.plot(x, y, color="brown")

        # Val Mean Dice WT
        plt.subplot(1, 4, 2)
        plt.title("Val Mean Dice WT")
        x = [self.val_interval * (i + 1) for i in range(len(self.metric_values_wt))]
        y = self.metric_values_wt
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.plot(x, y, color="purple")

        # Val Mean Dice ET
        plt.subplot(1, 4, 3)
        plt.title("Val Mean Dice ET")
        x = [self.val_interval * (i + 1) for i in range(len(self.metric_values_et))]
        y = self.metric_values_et
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.plot(x, y, color="orange")

        plt.subplot(1, 4, 4)
        plt.title("Val Mean Dice RC")
        x = [self.val_interval * (i + 1) for i in range(len(self.metric_values_rc))]
        y = self.metric_values_rc
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.plot(x, y, color="blue")

        plt.show()