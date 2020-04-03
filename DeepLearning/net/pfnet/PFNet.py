import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import numpy as np

class PFNet(nn.Module):
    def __init__(self, num_classes):
        super(PFNet, self).__init__()
        self.num_classes = num_classes
        # 第一层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=100)    
        self.relu1 = nn.ReLU()                                                                                  
        # 第二层                                                                                    
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)             
        self.relu2 = nn.ReLU()                                              
        # 第三层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        # 第四层
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        # 第五层
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        # 第六层
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=2, padding=1)
        self.relu6 = nn.ReLU()
        # 第七层
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        # 第八层
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()
        # 第九层
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=2, padding=1)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.reconv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.reconv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.reconv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.reconv0 = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        img_size = x.size()[2:]
        # c1
        x = self.relu1(self.conv1(x))
        # c2
        x = self.relu2(self.conv2(x)) # 1/2
        # c3
        x1 = self.relu3(self.conv3(x))
        size1 = x1.size()[2:]

        # c4
        x = self.relu4(self.conv4(x1)) # 1/4
        # c5
        x = self.relu5(self.conv5(x))
        # c6
        x2 = self.relu6(self.conv6(x))
        size2 = x2.size()[2:]

        # c7
        x = self.relu7(self.conv7(x2)) # 1/8
        # c8
        x = self.relu8(self.conv8(x))
        # c9
        x3 = self.relu9(self.conv9(x))
        size3 = x3.size()[2:]

        x = self.conv10(x3)

        x = F.interpolate(x, size3, mode='bilinear', align_corners=True)
        x = x + x3
        # x = self.relu(self.reconv3(x))
        x = self.reconv3(x)

        x = F.interpolate(x, size2, mode='bilinear', align_corners=True)
        x = x + x2
        # x = self.relu(self.reconv2(x))
        x = self.reconv2(x)

        x = F.interpolate(x, size1, mode='bilinear', align_corners=True)
        x = x + x1
        # x = self.relu(self.reconv1(x))
        x = self.reconv1(x)

        x = F.interpolate(x, img_size, mode='bilinear', align_corners=True)
        x = self.reconv0(x)

        return x