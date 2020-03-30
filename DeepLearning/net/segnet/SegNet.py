import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import numpy as np

class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()
        self.num_classes = num_classes
        # 卷积层
        # 第一层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)    
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()                                                                                  
        # 第二层                                                                                    
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)             
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU()                                              
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 第三层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.relu3 = nn.ReLU()
        # 第四层
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.relu4 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 第五层
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.relu5 = nn.ReLU()
        # 第六层
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=256)
        self.relu6 = nn.ReLU()
        # 第七层
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=256)
        self.relu7 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 第八层
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=512)
        self.relu8 = nn.ReLU()
        # 第九层
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(num_features=512)
        self.relu9 = nn.ReLU()
        # 第十层
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(num_features=512)
        self.relu10 = nn.ReLU()
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 第十一层
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(num_features=512)
        self.relu11 = nn.ReLU()
        # 第十二层
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(num_features=512)
        self.relu12 = nn.ReLU()
        # 第十三层
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(num_features=512)
        self.relu13 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Decode-13
        self.upsample5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decode_conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.decode_bn13 = nn.BatchNorm2d(num_features=512)
        self.decode_relu13 = nn.ReLU()
        # Decode-12
        self.decode_conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.decode_bn12 = nn.BatchNorm2d(num_features=512)
        self.decode_relu12 = nn.ReLU()
        # Decode-11
        self.decode_conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.decode_bn11 = nn.BatchNorm2d(num_features=512)
        self.decode_relu11 = nn.ReLU()
        # Decode-10
        self.upsample4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decode_conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.decode_bn10 = nn.BatchNorm2d(num_features=512)
        self.decode_relu10 = nn.ReLU()
        # Decode-9
        self.decode_conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.decode_bn9 = nn.BatchNorm2d(num_features=512)
        self.decode_relu9 = nn.ReLU()
        # Decode-8
        self.decode_conv8 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.decode_bn8 = nn.BatchNorm2d(num_features=256)
        self.decode_relu8 = nn.ReLU()
        # Decode-7
        self.upsample3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decode_conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.decode_bn7 = nn.BatchNorm2d(num_features=256)
        self.decode_relu7 = nn.ReLU()
        # Decode-6
        self.decode_conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.decode_bn6 = nn.BatchNorm2d(num_features=256)
        self.decode_relu6 = nn.ReLU()
        # Decode-5
        self.decode_conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.decode_bn5 = nn.BatchNorm2d(num_features=128)
        self.decode_relu5 = nn.ReLU()
        # Decode-4
        self.upsample2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decode_conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.decode_bn4 = nn.BatchNorm2d(num_features=128)
        self.decode_relu4 = nn.ReLU()
        # Decode-3
        self.decode_conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.decode_bn3 = nn.BatchNorm2d(num_features=64)
        self.decode_relu3 = nn.ReLU()
        # Decode-2
        self.upsample1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decode_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.decode_bn2 = nn.BatchNorm2d(num_features=64)
        self.decode_relu2 = nn.ReLU()
        # Decode-1
        self.decode_conv1 = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()
    
    # def initialize_weights(self):
    #     nn.init.kaiming_normal_(self.conv1.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv2.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv3.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv4.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv5.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv6.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv7.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.max_pool3_predict.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv8.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv9.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv10.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.max_pool4_predict.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv11.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv12.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.conv13.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.fc_conv1.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.fc_conv2.weight.data, mode='fan_in', nonlinearity='relu')
    #     nn.init.kaiming_normal_(self.fc_conv3.weight.data, mode='fan_in', nonlinearity='relu')

    def initialize_weights(self):
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.conv1.weight.data = vgg16.features[0].weight.data.clone()
        self.conv2.weight.data = vgg16.features[2].weight.data.clone()
        self.conv3.weight.data = vgg16.features[5].weight.data.clone()
        self.conv4.weight.data = vgg16.features[7].weight.data.clone()
        self.conv5.weight.data = vgg16.features[10].weight.data.clone()
        self.conv6.weight.data = vgg16.features[12].weight.data.clone()
        self.conv7.weight.data = vgg16.features[14].weight.data.clone()
        self.conv8.weight.data = vgg16.features[17].weight.data.clone()
        self.conv9.weight.data = vgg16.features[19].weight.data.clone()
        self.conv10.weight.data = vgg16.features[21].weight.data.clone()
        self.conv11.weight.data = vgg16.features[24].weight.data.clone()
        self.conv12.weight.data = vgg16.features[26].weight.data.clone()
        self.conv13.weight.data = vgg16.features[28].weight.data.clone()

        self.conv1.bias.data = vgg16.features[0].bias.data.clone()
        self.conv2.bias.data = vgg16.features[2].bias.data.clone()
        self.conv3.bias.data = vgg16.features[5].bias.data.clone()
        self.conv4.bias.data = vgg16.features[7].bias.data.clone()
        self.conv5.bias.data = vgg16.features[10].bias.data.clone()
        self.conv6.bias.data = vgg16.features[12].bias.data.clone()
        self.conv7.bias.data = vgg16.features[14].bias.data.clone()
        self.conv8.bias.data = vgg16.features[17].bias.data.clone()
        self.conv9.bias.data = vgg16.features[19].bias.data.clone()
        self.conv10.bias.data = vgg16.features[21].bias.data.clone()
        self.conv11.bias.data = vgg16.features[24].bias.data.clone()
        self.conv12.bias.data = vgg16.features[26].bias.data.clone()
        self.conv13.bias.data = vgg16.features[28].bias.data.clone()


    def forward(self, x):
        # c1
        x = self.relu1(self.bn1(self.conv1(x)))
        # c2
        size1 = x.size()[2:]
        x, indices1 = self.max_pool1(self.relu2(self.bn2(self.conv2(x)))) # 1/2
        
        # c3
        x = self.relu3(self.bn3(self.conv3(x)))
        # c4
        size2 = x.size()[2:]
        x, indices2 = self.max_pool2(self.relu4(self.bn4(self.conv4(x)))) # 1/4
        
        # c5
        x = self.relu5(self.bn5(self.conv5(x)))
        # c6
        x = self.relu6(self.bn6(self.conv6(x)))
        # c7
        size3 = x.size()[2:]
        x, indices3 = self.max_pool3(self.relu7(self.bn7(self.conv7(x)))) # 1/8

        # c8
        x = self.relu8(self.bn8(self.conv8(x)))
        # c9
        x = self.relu9(self.bn9(self.conv9(x)))
        # c10
        size4 = x.size()[2:]
        x, indices4 = self.max_pool4(self.relu10(self.bn10(self.conv10(x)))) # 1/16

        # c11
        x = self.relu11(self.bn11(self.conv11(x)))
        # c12
        x = self.relu12(self.bn12(self.conv12(x)))
        # c13
        size5 = x.size()[2:]
        x, indices5 = self.max_pool5(self.relu13(self.bn13(self.conv13(x)))) # 1/32


        # decode-c13
        x = self.upsample5(x, indices5, output_size=size5)
        x = self.decode_relu13(self.decode_bn13(self.decode_conv13(x)))
        # decode-c12
        x = self.decode_relu12(self.decode_bn12(self.decode_conv12(x)))
        # decode-c11
        x = self.decode_relu11(self.decode_bn11(self.decode_conv11(x)))
        
        # decode-c10
        x = self.upsample4(x, indices4, output_size=size4)
        x = self.decode_relu10(self.decode_bn10(self.decode_conv10(x)))
        # decode-c9
        x = self.decode_relu9(self.decode_bn9(self.decode_conv9(x)))
        # decode-c8
        x = self.decode_relu8(self.decode_bn8(self.decode_conv8(x)))
        
        # decode-c7
        x = self.upsample3(x, indices3, output_size=size3)
        x = self.decode_relu7(self.decode_bn7(self.decode_conv7(x)))
        # decode-c6
        x = self.decode_relu6(self.decode_bn6(self.decode_conv6(x)))
        # decode-c5
        x = self.decode_relu5(self.decode_bn5(self.decode_conv5(x)))
        
        # decode-c4
        x = self.upsample2(x, indices2, output_size=size2)
        x = self.decode_relu4(self.decode_bn4(self.decode_conv4(x)))
        # decode-c3
        x = self.decode_relu3(self.decode_bn3(self.decode_conv3(x)))
        
        # decode-c2
        x = self.upsample1(x, indices1, output_size=size1)
        x = self.decode_relu2(self.decode_bn2(self.decode_conv2(x)))
        # decode-c1
        x = self.decode_relu1(self.decode_bn1(self.decode_conv1(x)))

        return x


