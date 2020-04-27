import torch
import torch.nn as nn
import torchvision
class DeepLabv3(nn.Module):
    def __init__(self, num_classes,n_layers=50):
        super(DeepLabv3, self).__init__()
        self.num_classes = num_classes
        if n_layers == 50:
            self.pretrained = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=self.num_classes)
        elif n_layers == 101:
            self.pretrained = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=self.num_classes)
        else:
            print("layer num error")
    
    def forward(self, x):
        return self.pretrained(x)["out"]