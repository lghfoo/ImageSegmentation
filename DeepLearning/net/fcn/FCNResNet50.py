import torch
import torch.nn as nn
import torchvision

class FCNResNet50(nn.Module):
    def __init__(self, num_classes):
        super(FCNResNet50, self).__init__()
        self.num_classes = num_classes
        self.pretrained = torchvision.models.segmentation.fcn_resnet50(pretrained=True, num_classes=self.num_classes)
    
    def forward(self, x):
        self.pretrained(x)