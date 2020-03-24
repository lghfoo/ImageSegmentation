import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.num_classes = num_classes
        # 卷积层
        # 第一层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=100)    
        self.relu1 = nn.ReLU()                                                                                  
        # 第二层                                                                                    
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)             
        self.relu2 = nn.ReLU()                                              
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        # 第四层
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第五层
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.retlu5 = nn.ReLU()
        # 第六层
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        # 第七层
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第八层
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()
        # 第九层
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()
        # 第十层
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu10 = nn.ReLU()
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第十一层
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu11 = nn.ReLU()
        # 第十二层
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu12 = nn.ReLU()
        # 第十三层
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu13 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        #全连接层
        # 第一层
        self.fc_conv1 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        self.fc_relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        # 第二层
        self.fc_conv2 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        self.fc_relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        # 第三层
        self.fc_conv3 = nn.Conv2d(in_channels=4096, out_channels=self.num_classes, kernel_size=1)

    def forward(self, x):
        img_size = x.size()[2:]
        # c1
        x = self.relu1(self.conv1(x))
        # c2
        x = self.max_pool1(self.relu2(self.conv2(x))) # 1/2
        
        # c3
        x = self.relu3(self.conv3(x))
        # c4
        x = self.max_pool2(self.relu4(self.conv4(x))) # 1/4
        
        # c5
        x = self.relu5(self.conv5(x))
        # c6
        x = self.relu6(self.conv6(x))
        # c7
        x = self.max_pool3(self.relu7(self.conv7(x))) # 1/8
        size_8 = x.size()[2:]
        pool3_predict = x.clone()
        
        # c8
        x = self.relu8(self.conv8(x))
        # c9
        x = self.relu9(self.conv9(x))
        # c10
        x = self.max_pool4(self.relu10(self.conv10(x))) # 1/16
        size_16 = x.size()[2:]
        pool4_predict = x.clone()

        # c11
        x = self.relu11(self.conv11(x))
        # c12
        x = self.relu12(self.conv12(x))
        # c13
        x = self.max_pool5(self.relu13(self.conv13)) # 1/32

        # fc1
        x = self.dropout1(self.fc_relu1(self.fc_conv1(x)))
        # fc2
        x = self.dropout2(self.fc_relu2(self.fc_conv2(x)))
        # fc3
        x = self.fc_conv3(x)

        upsamled2x = F.interpolate(x, size = size_16)
        sigma1 = upsamled2x + pool4_predict
        upsamled2x_sigmal1 = F.interpolate(sigma1, size = size_8)
        sigma2 = upsamled2x_sigmal1 + pool3_predict
        upsamled8x = F.interpolate(sigma2, size = img_size)      
          
        return upsamled8x