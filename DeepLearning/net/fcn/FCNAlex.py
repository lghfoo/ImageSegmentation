import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy as np
import random
from PIL import Image

class LRN(nn.Module):
    def __init__(self, n=1, k=1.0, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(n, 1, 1),
                    stride=1,
                    padding=(int((n-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=n,
                    stride=1,
                    padding=int((n-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x

class AlexNetFCN(nn.Module):
    def __init__(self, num_classes):
        super(AlexNetFCN, self).__init__()
        self.num_classes = num_classes
        # 第一层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4,padding=100)
        self.relu1 = nn.ReLU()
        self.lrn1 = LRN(n=5, k=2.0, alpha=1e-4, beta=0.75)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 第二层
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, groups=2)
        self.relu2 = nn.ReLU()
        self.lrn2 = LRN(n=5, k=2.0, alpha=1e-4, beta=0.75)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 第三层
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        # 第四层
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=2)
        self.relu4 = nn.ReLU()
        # 第五层
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, groups=2)
        self.relu5 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc_conv1 = nn.Conv2d(256, 4096, kernel_size=6)
        self.relu6 = nn.ReLU()
        self.drop1 = nn.Dropout2d()

        self.fc_conv2 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU()
        self.drop2 = nn.Dropout2d()

        self.fc_conv3 = nn.Conv2d(4096, self.num_classes, kernel_size=1)

    def forward(self, x):
        img_size = x.size()[2:]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        
        x = self.fc_conv1(x)
        x = self.relu6(x)
        x = self.drop1(x)

        x = self.fc_conv2(x)
        x = self.relu7(x)
        x = self.drop2(x)

        x = self.fc_conv3(x)

        x = F.interpolate(x, size = img_size)
        return x




