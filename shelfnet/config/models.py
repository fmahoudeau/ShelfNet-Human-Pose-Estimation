# ==============================================================================
# Copyright 2019 Florent Mahoudeau. All Rights Reserved.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode


# pose shelfnet related params
POSE_SHELFNET = CfgNode()
POSE_SHELFNET.BACKBONE = 50
POSE_SHELFNET.EXTRA_BLOCK = False
POSE_SHELFNET.DECONV_WITH_BIAS = True
POSE_SHELFNET.FUSE_METHOD = 'PYRAMID'
POSE_SHELFNET.FINAL_CONV_KERNEL = 1
POSE_SHELFNET.PRETRAINED_LAYERS = ['*']


MODEL_EXTRAS = {
    'shelfnet': POSE_SHELFNET,
}
