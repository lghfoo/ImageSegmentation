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


        upsamled2x = F.interpolate(c4, c3.size()[1:], mode='nearest', align_corners=True)
        sigma1 = upsamled2x + c3
        upsamled2x_sigmal1 = F.interpolate(sigma1, c2.size()[1:], mode='nearest', align_corners=True)
        sigma2 = upsamled2x_sigmal1 + c2
        upsampled8x = F.interpolate(sigma2, torch.Tensor([self.num_classes, img_size[0], img_size[1]]), mode='nearest', align_corners=True)
        return upsampled8x