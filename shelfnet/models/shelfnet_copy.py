###########################################################################
# Created by: Florent Mahoudeau
# Copyright (c) 2019
###########################################################################

import math
import os.path

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import BatchNorm2d
import torch.nn.functional as F

__all__ = ['ShelfNet', 'BasicBlock', 'get_pose_net',
           'ResNet', 'resnet50', 'resnet101', 'Bottleneck']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

MODEL_ZOO_DIR = '/home/fanos/Documents/ModelZoo/PyTorch'


class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, dilated=False, norm_layer=None,
                 base_size=576, crop_size=608, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='~/Documents/Datasets/shelfnet/models'):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size

        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet50(pretrained=True, dilated=dilated,
                                       norm_layer=norm_layer, root=root)
        elif backbone == 'resnet101':
            self.pretrained = resnet101(pretrained=True, dilated=dilated,
                                        norm_layer=norm_layer, root=root)

        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4


class ShelfNet(BaseNet):
    def __init__(self, nclass, backbone, norm_layer=BatchNorm2d, dilated=False, **kwargs):
        super(ShelfNet, self).__init__(nclass, backbone, norm_layer=norm_layer, dilated=dilated, **kwargs)
        self.head = ShelfHead()
        self.final = nn.Conv2d(64, nclass, 1)

    def forward(self, x):
        features = self.base_forward(x)
        out = self.head(features)
        pred = self.final(out[-1])

        return pred


class BasicBlock(nn.Module):
    def __init__(self, planes):
        super(BasicBlock, self).__init__()

        self.planes = planes
        self.conv1 = nn.Conv2d(self.planes, self.planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=0.25)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = out + x

        return self.relu2(out)


class ShelfDecoder(nn.Module):
    def __init__(self, block=BasicBlock):
        super().__init__()

        # create basic blocks
        self.block_a = block(64)
        self.block_b = block(128)
        self.block_c = block(256)
        self.block_d = block(512)

        # create up-sampling layers
        self.up_conv_1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.up_conv_2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.up_conv_3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.block_d(x4)

        out2 = self.up_conv_1(out1) + x3
        out2 = self.block_c(out2)

        out3 = self.up_conv_2(out2) + x2
        out3 = self.block_b(out3)

        out4 = self.up_conv_3(out3) + x1
        out4 = self.block_a(out4)

        return [out1, out2, out3, out4]


class ShelfHead(nn.Module):
    def __init__(self):
        super(ShelfHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.decoder = ShelfDecoder()
        self.tail = ShelfTail()

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)

        out2 = self.conv2(x2)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)

        out3 = self.conv3(x3)
        out3 = self.bn3(out3)
        out3 = self.relu3(out3)

        out4 = self.conv4(x4)
        out4 = self.bn4(out4)
        out4 = self.relu4(out4)

        out = self.decoder([out1, out2, out3, out4])
        out = self.tail(out)  # Second encoder-decoder pass

        return out


class ShelfTail(nn.Module):
    def __init__(self, block=BasicBlock):
        super().__init__()

        # create basic blocks
        self.block_in = block(64)
        self.block_a = block(64)
        self.block_b = block(128)
        self.block_c = block(256)

        # create down-sampling layers
        self.down_conv_1 = nn.Conv2d(64, 128, stride=2, kernel_size=3, padding=1)
        self.down_conv_2 = nn.Conv2d(128, 256, stride=2, kernel_size=3, padding=1)
        self.down_conv_3 = nn.Conv2d(256, 512, stride=2, kernel_size=3, padding=1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.decoder = ShelfDecoder()

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.block_in(x4) + x4
        out1 = self.block_a(out1)

        out2 = self.down_conv_1(out1)
        out2 = self.relu1(out2) + x3
        out2 = self.block_b(out2)

        out3 = self.down_conv_2(out2)
        out3 = self.relu2(out3) + x2
        out3 = self.block_c(out3)

        out4 = self.down_conv_3(out3)
        out4 = self.relu3(out4)

        out = self.decoder([out1, out2, out3, out4])

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck"""
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes=1000, dilated=True, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('Loading Resnet50 weights')
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet34'],
        #                                         model_dir=os.path.join(MODEL_ZOO_DIR,
        #                                                                'official')))
        model.load_state_dict(torch.load(
            '/home/fanos/Documents/ModelZoo/PyTorch/hangzh/resnet50-853f2fb0.pth'))
    return model


def resnet101(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'],
                                                 model_dir=os.path.join(MODEL_ZOO_DIR,
                                                                        'official')), strict=False)
    return model


def get_pose_net(cfg, is_train, backbone='resnet50',
                 root='/home/fanos/Documents/ModelZoo/official', **kwargs):
    model = ShelfNet(cfg.MODEL.NUM_JOINTS, backbone=backbone, root=root, **kwargs)
    return model
