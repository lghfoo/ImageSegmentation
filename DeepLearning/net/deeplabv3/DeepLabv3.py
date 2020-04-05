import torch
import torch.nn as nn
import torchvision
class DeepLabv3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabv3, self).__init__()
        self.num_classes = num_classes
        self.pretrained = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=self.num_classes)
    
    def forward(self, x):
        return self.pretrained(x)["out"]