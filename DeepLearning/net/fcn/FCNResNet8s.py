import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import numpy as np

class FCNResNet8s(nn.Module):
    def __init__(self, num_classes):
        super(FCNResNet8s, self).__init__()
        self.num_classes = num_classes
        self.pretrained = torchvision.models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=100,
                        bias=False)
        self.final_conv = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=self.num_classes, kernel_size=16, stride=8, bias=False)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, bias=False)
        self.deconv3 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, bias=False)


    def forward(self, x):
        img_size = x.size()[2:]

        x = self.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        c4 = self.final_conv(c4)

        def center_crop_tensor(t1, t2):
            (h_t1, w_t1, h_t2, w_t2) = (t1.size()[2], t1.size()[3], t2.size()[2], t2.size()[3])
            return t1[:,:,int((h_t1-h_t2)/2):int((h_t1-h_t2)/2)+h_t2, int((w_t1-w_t2)/2):int((w_t1-w_t2)/2)+w_t2]

        upsamled2x = self.deconv3(c4)
        sigma1 = c3 + center_crop_tensor(upsamled2x, c3)
        upsamled2x_sigmal1 = self.deconv2(sigma1)
        sigma2 = c2 + center_crop_tensor(upsamled2x_sigmal1, c2)
        upsampled8x = self.deconv1(sigma2)

        return center_crop_tensor(upsampled8x, torch.Tensor(1, 1, img_size[0], img_size[1]))