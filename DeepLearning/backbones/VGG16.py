import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.img_size = (224, 224)
        self.fc_num = 7 * 7 * 512
        self.num_classes = num_classes
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第二层
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第三层
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第四层
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第五层
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第六层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第七层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第八层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            # 第一层
            nn.Linear(self.fc_num, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 第二层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 第三层
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_num)
        x = self.fc_layers(x)
        return x

class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.fc_num = 7 * 7 * 512 
        self.num_classes = num_classes
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),    
            nn.ReLU(),                                                                                  
            # 第二层                                                                                    
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),             
            nn.ReLU(),                                              
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第三层
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第四层
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第五层
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第六层
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第七层
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第八层
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第九层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第十层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第十一层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第十二层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第十三层
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            # 第一层
            nn.Linear(self.fc_num, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 第二层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 第三层
            nn.Linear(4096, self.num_classes),
        )
        # self.initialize()

    def initialize(self, NetVGG11):
        self.conv_layers[0].weight.data = NetVGG11.conv_layers[0].weight.data.clone()
        self.conv_layers[0].bias.data = NetVGG11.conv_layers[0].bias.data.clone()
        self.conv_layers[5].weight.data = NetVGG11.conv_layers[3].weight.data.clone()
        self.conv_layers[5].bias.data = NetVGG11.conv_layers[3].bias.data.clone()
        self.conv_layers[10].weight.data = NetVGG11.conv_layers[6].weight.data.clone()
        self.conv_layers[10].bias.data = NetVGG11.conv_layers[6].bias.data.clone()
        self.conv_layers[12].weight.data = NetVGG11.conv_layers[8].weight.data.clone()
        self.conv_layers[12].bias.data = NetVGG11.conv_layers[8].bias.data.clone()

        self.fc_layers[0].weight.data = NetVGG11.fc_layers[0].weight.data.clone()
        self.fc_layers[0].bias.data = NetVGG11.fc_layers[0].bias.data.clone()
        self.fc_layers[3].weight.data = NetVGG11.fc_layers[3].weight.data.clone()
        self.fc_layers[3].bias.data = NetVGG11.fc_layers[3].bias.data.clone()
        self.fc_layers[6].weight.data = NetVGG11.fc_layers[6].weight.data.clone()
        self.fc_layers[6].bias.data = NetVGG11.fc_layers[6].bias.data.clone()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_num)
        x = self.fc_layers(x)
        return x


