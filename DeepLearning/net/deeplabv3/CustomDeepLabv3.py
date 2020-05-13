import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3, ASPP
from torchvision.models.segmentation.fcn import FCN, FCNHead

model_urls = {
    'fcn_resnet50_coco': None,
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': None,
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}

class CustomDeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, rates=[12,24,36]):
        super(CustomDeepLabHead, self).__init__(
            ASPP(in_channels, rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        print('rates: {}'.format(rates))

def _segm_resnet(name, backbone_name, num_classes, aux, dilation, pretrained_backbone=True):
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
        'deeplabv3': (CustomDeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 2048
    print('dilation: {}'.format(dilation))
    if name == 'fcn':
        classifier = model_map[name][0](inplanes, num_classes)
    else:
        if (not dilation[0]) and dilation[1] and dilation[2]:
            classifier = model_map[name][0](inplanes, num_classes, [12, 24, 36])
        elif (not dilation[0]) and (not dilation[1]) and dilation[2]:
            classifier = model_map[name][0](inplanes, num_classes, [6, 12, 18])
        else:
            print('invalid dilation value')
            return None
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model

def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, dilation, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, dilation, **kwargs)
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
                       num_classes=21, aux_loss=None, dilation, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, dilation, **kwargs)

def deeplabv3_resnet101(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, dilation, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, aux_loss, dilation, **kwargs)

class CustomDeepLabv3(nn.Module):
    def __init__(self, num_classes,n_layers=50, output_stride=16):
        super(CustomDeepLabv3, self).__init__()
        assert(output_stride in [4, 8, 16, 32])
        self.num_classes = num_classes
        self.output_stride = output_stride
        stride_dict = {
            4: [True, True, True],
            8: [False, True, True],
            16: [False, False, True],
            32: [False, False, False]
        }
        print('output stride: {}, dilation: {}'.format(output_stride, stride_dict[self.output_stride]))
        if n_layers == 50:
            self.pretrained = deeplabv3_resnet50(pretrained=False, num_classes=self.num_classes, dilation=stride_dict[self.output_stride])
        elif n_layers == 101:
            self.pretrained = deeplabv3_resnet101(pretrained=False, num_classes=self.num_classes, dilation=stride_dict[self.output_stride])
        else:
            print("layer num error")
    
    def forward(self, x):
        return self.pretrained(x)["out"]