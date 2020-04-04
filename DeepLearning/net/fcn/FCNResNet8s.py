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
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=self.num_classes, kernel_size=16, stride=8, bias=False)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, bias=False)
        self.deconv3 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, bias=False)


    def forward(self, x):
        img_size = x.size()[2:]

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        def center_crop_tensor(t1, t2):
            (h_t1, w_t1, h_t2, w_t2) = (t1.size()[2], t1.size()[3], t2.size()[2], t2.size()[3])
            return t1[:,:,int((h_t1-h_t2)/2):int((h_t1-h_t2)/2)+h_t2, int((w_t1-w_t2)/2):int((w_t1-w_t2)/2)+w_t2]

        upsamled2x = self.deconv3(c4)
        sigma1 = upsamled2x + center_crop_tensor(c3, upsamled2x)
        upsamled2x_sigmal1 = self.deconv2(sigma1)
        sigma2 = upsamled2x_sigmal1 + center_crop_tensor(c2, upsamled2x_sigmal1)
        upsampled8x = self.deconv1(sigma2)
        
        return center_crop_tensor(upsampled8x, torch.Tensor(1, 1, img_size[0], img_size[1]))