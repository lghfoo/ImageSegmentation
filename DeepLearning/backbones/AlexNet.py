import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

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


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.fc_num = 256 * 6 * 6
        self.img_size = (227, 227)
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4,padding=0),
            nn.ReLU(),
            LRN(n=5, k=2.0, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第二层
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(),
            LRN(n=5, k=2.0, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第三层
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            # 第四层
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(),
            # 第五层
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 全连接层
        self.fc_layers = nn.Sequential(
            # 第一层
            nn.Linear(self.fc_num, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # 第二层
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # 第三层
            nn.Linear(4096, self.num_classes),
        )
        #self.initialize()

    def initialize(self):
        # init conv1
        self.conv_layers[0].weight.data.normal_(mean=0, std=0.01)
        self.conv_layers[0].bias.data.fill_(0)
        # init conv2
        self.conv_layers[3].weight.data.normal_(mean=0, std=0.01)
        self.conv_layers[3].bias.data.fill_(1)
        # init conv3
        self.conv_layers[6].weight.data.normal_(mean=0, std=0.01)
        self.conv_layers[6].bias.data.fill_(0)
        # init conv4
        self.conv_layers[8].weight.data.normal_(mean=0, std=0.01)
        self.conv_layers[8].bias.data.fill_(1)
        # init conv5
        self.conv_layers[10].weight.data.normal_(mean=0, std=0.01)
        self.conv_layers[10].bias.data.fill_(1)
        # init fc1
        self.fc_layers[0].weight.data.normal_(mean=0, std=0.01)
        self.fc_layers[0].bias.data.fill_(1)
        # init fc2
        self.fc_layers[3].weight.data.normal_(mean=0, std=0.01)
        self.fc_layers[3].bias.data.fill_(1)
        # init fc3
        self.fc_layers[6].weight.data.normal_(mean=0, std=0.01)
        self.fc_layers[6].bias.data.fill_(0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_num)
        x = self.fc_layers(x)
        return x
        