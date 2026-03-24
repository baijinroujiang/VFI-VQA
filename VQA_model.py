import numpy as np
import os,sys
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F

class block_r3d_18(torch.nn.Module):
    def __init__(self):
        super(block_r3d_18, self).__init__()
        resnet_pretrained_features = nn.Sequential(*list(models.video.r3d_18(pretrained=True).children())[:-2])
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        self.stage1.add_module(str(0), resnet_pretrained_features[0])
        self.stage2.add_module(str(1), resnet_pretrained_features[1])
        self.stage3.add_module(str(2), resnet_pretrained_features[2])
        self.stage4.add_module(str(3), resnet_pretrained_features[3])
        self.stage5.add_module(str(4), resnet_pretrained_features[4])
    def forward(self, x):
        h = self.stage1(x)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]  

class Res3d18_NR(torch.nn.Module):
    def __init__(self):
        super(Res3d18d_NR, self).__init__()

        self.blockf = block_r3d_18()
        self.chns = [64, 64, 128, 256, 512]
        self.quality = self.quality_regression(sum(self.chns), 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(middle_channels, out_channels),
            nn.Sigmoid()
        )
        return regression_block

    def dis_frame(self, feats_f1):
        dist2 = []
        c2 = 1e-6
        feats0 = [feats_f1[k][:, :, 0, :, :] for k in range(2)]
        feats1 = [feats_f1[k][:, :, 1, :, :] for k in range(2)]
        feats2 = [feats_f1[k][:, :, 2, :, :] for k in range(2)]
        for k in range(2):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            z_mean = feats2[k].mean([2, 3], keepdim=True)
            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            z_var = ((feats2[k] - z_mean) ** 2).mean([2, 3], keepdim=True)
            xz_cov = (feats0[k] * feats2[k]).mean([2, 3], keepdim=True) - x_mean * z_mean
            yz_cov = (feats1[k] * feats2[k]).mean([2, 3], keepdim=True) - y_mean * z_mean
            S2x = (2 * xz_cov + c2) / (x_var + z_var + c2)
            S2y = (2 * yz_cov + c2) / (y_var + z_var + c2)
            dist2.append((S2x * S2y).squeeze(3).squeeze(2))

        x_mean = feats_f1[2][:, :, 0, :, :].mean([2, 3], keepdim=True)
        y_mean = feats_f1[2][:, :, 1, :, :].mean([2, 3], keepdim=True)
        x_var = ((feats_f1[2][:, :, 0, :, :] - x_mean) ** 2).mean([2, 3], keepdim=True)
        y_var = ((feats_f1[2][:, :, 1, :, :] - y_mean) ** 2).mean([2, 3], keepdim=True)
        cov = (feats_f1[2][:, :, 0, :, :] * feats_f1[2][:, :, 1, :, :]).mean([2, 3], keepdim=True) - x_mean * y_mean
        S2 = (2 * cov + c2) / (x_var + y_var + c2)
        dist2.append(S2.squeeze(3).squeeze(2))
        for k in range(3, 5):
            dist2.append(feats_f1[k][:, :, 0, :, :].mean([2, 3]))
        dist2 = torch.cat(dist2, 1)
        return dist2

    def forward(self, ref, xyz, require_grad=False, batch_average=False):
        score = 0
        B, C, T, H, W = xyz.shape
        for i in range(int(T/3)):
            xyz_f = xyz[:, :, i*3: i*3+3]
            feats_f1 = self.blockf(xyz_f)
            dist2 = self.dis_frame(feats_f1)
            score += self.quality(dist2).squeeze()
        score = score/(T/3)
        return score
