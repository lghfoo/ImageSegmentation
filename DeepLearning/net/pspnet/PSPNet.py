import torch
from torch import nn
import torchvision
import torch.nn.functional as F


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class PSPNet(nn.Module):
    def __init__(self, num_classes, bins=(1,2,3,6)):
        super(PSPNet, self).__init__()
        self.num_classes = num_classes
        self.bins = bins
        self.training = False
        self.pretrained = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=self.num_classes)
        for n, m in self.pretrained.backbone.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.pretrained.backbone.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        fea_dim = 2048
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, nn.BatchNorm2d)
        fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=nn.Dropout2d(p=0.1)),
                nn.Conv2d(256, self.num_classes, kernel_size=1)
            )

    def forward(self, x):
        x_size = x.size()[2:]
        x = self.pretrained.backbone(x)["out"]
        # backbone = self.pretrained.backbone
        # x = backbone.maxpool(backbone.relu(backbone.conv1(x)))
        # x = backbone.layer1(x)
        # x = backbone.layer2(x)
        # x = backbone.layer3(x)
        
        # x_tmp = self.layer3(x)
        # x = self.layer4(x_tmp)
        # if self.use_ppm:
        #     x = self.ppm(x)
        # x = self.cls(x)
        # x = F.interpolate(x, size=x_size, mode='bilinear', align_corners=True)

        # if self.training:
        #     aux = self.aux(x_tmp)
        #     # if self.zoom_factor != 1:
        #     aux = F.interpolate(aux, size=x_size, mode='bilinear', align_corners=True)
        #     # main_loss = self.criterion(x, y)
        #     # aux_loss = self.criterion(aux, y)
        #     # return x.max(1)[1], main_loss, aux_loss
        #     return x, aux
        # else:
        #     return x
        x = self.ppm(x)
        x = self.cls(x)
        x = F.interpolate(x, size=x_size, mode='bilinear', align_corners=True)
        return x

