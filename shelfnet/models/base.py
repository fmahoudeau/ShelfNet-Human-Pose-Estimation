# ==============================================================================
# Copyright 2019 Florent Mahoudeau.
# Modified from https://github.com/juntang-zhuang/ShelfNet
# Licensed under the MIT License.
# ==============================================================================

import torch.nn as nn
import sys
sys.path.insert(0, '../../')

from shelfnet.backbones import resnet


__all__ = ['BaseNet']


class BaseNet(nn.Module):
    def __init__(self, n_joints, backbone, dilated=True, norm_layer=None,
                 base_size=576, crop_size=608, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='~/Documents/Datasets/shelfnet/models'):
        super(BaseNet, self).__init__()
        self.n_joints = n_joints
        self.backbone = backbone
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size

        # copying modules from pretrained models
        if backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer, root=root)
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer, root=root)
        elif backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer, root=root)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer, root=root)
        else:
            raise RuntimeError('Unknown backbone: {}'.format(backbone))

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
