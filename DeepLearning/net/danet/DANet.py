###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import upsample
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter
import torchvision
from torchvision.models import resnet
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class BaseNet(nn.Module):
    def __init__(self, nclass, aux, se_loss, dilated=True, norm_layer=None,
                 base_size=576, crop_size=608, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='./pretrain_models',
                 multi_grid=False, multi_dilation=None):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        # self.pretrained = torchvision.models.resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
        # self.pretrained = torchvision.models.resnet101(pretrained=False)

    def base_forward(self, x):
        # x = self.pretrained.conv1(x)
        # x = self.pretrained.bn1(x)
        # x = self.pretrained.relu(x)
        # x = self.pretrained.maxpool(x)
        # c1 = self.pretrained.layer1(x)
        # c2 = self.pretrained.layer2(c1)
        # c3 = self.pretrained.layer3(c2)
        # c4 = self.pretrained.layer4(c3)
        # return c1, c2, c3, c4
        return self.pretrained.layer4(self.pretrained.layer3(self.pretrained.layer2(self.pretrained.layer1(self.pretrained.relu(self.pretrained.bn1(self.pretrained.conv1(x)))))))

class DANet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """
    def __init__(self, nclass, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet, self).__init__(nclass, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.num_classes = nclass
        # self.pretrained = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=self.num_classes)
        # self.head = DANetHead(2048, 1024, norm_layer)
        # self.classifier = torchvision.models.segmentation.fcn.FCNHead(2048, self.num_classes)

        backbone = resnet.__dict__['resnet50'](
            pretrained=False,
            replace_stride_with_dilation=[False, True, True])
        return_layers = {'layer4': 'out'}
        self.backbone = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.classifier = torchvision.models.segmentation.fcn.FCNHead(2048, self.num_classes)
        # base_model = torchvision.models.segmentation.fcn.FCN(backbone, classifier, None)
        
    
    # def forward(self, x):
    #     return self.pretrained(x)["out"]

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.backbone(x)

        result = torchvision.models.segmentation._utils.OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=imsize, mode='bilinear', align_corners=False)
        result["out"] = x

        return result["out"]
        # _, _, _, c4 = self.base_forward(x)
        # x = self.base_forward(x)
        # x = self.head(c4)
        # x = list(x)
        # x = self.classifier(x[0])
        # x = self.classifier(x)
        # x = F.interpolate(x, size=imsize, mode='bilinear', align_corners=False)
        # return x

        # x[0] = upsample(x[0], imsize, mode='bilinear', align_corners=True)
        # x[1] = upsample(x[1], imsize, mode='bilinear', align_corners=True)
        # x[2] = upsample(x[2], imsize, mode='bilinear', align_corners=True)

        # outputs = [x[0]]
        # outputs.append(x[1])
        # outputs.append(x[2])
        # return tuple(outputs)
        # return x[0]

        
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)