import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models import resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead

def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True, dilation=[False, True, True]):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=dilation)

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model

def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model

def deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)

class CustomDeepLabv3(nn.Module):
    def __init__(self, num_classes,n_layers=50):
        super(CustomDeepLabv3, self).__init__()
        self.num_classes = num_classes
        if n_layers == 50:
            self.pretrained = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=self.num_classes, dilation=[False, False, True])
        elif n_layers == 101:
            self.pretrained = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=self.num_classes, dilation=[False, False, True])
        else:
            print("layer num error")
    
    def forward(self, x):
        return self.pretrained(x)["out"]