# ==============================================================================
# Copyright 2019 Florent Mahoudeau.
# Licensed under the MIT License.
# ==============================================================================

import argparse
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import sys
sys.path.insert(0, '../')

from shelfnet.config import cfg
from shelfnet.config import update_config

from shelfnet import models


def parse_args():
    parser = argparse.ArgumentParser(description='Test speed keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def test_speed():
    args = parse_args()
    update_config(cfg, args)

    print('Compute device: ' + "cuda:%d" % cfg.GPUS[0] if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:%d" % cfg.GPUS[0] if torch.cuda.is_available() else "cpu")

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model_kwargs = {'is_train': False} \
        if cfg.MODEL.NAME == 'pose_hrnet' else {}
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, **model_kwargs)

    # count parameter number
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %d" % pytorch_total_params)

    model = model.to(device)
    model.eval()

    run_time = list()
    print("Input shape = " + str(cfg.MODEL.IMAGE_SIZE[::-1]))

    for i in range(0, 10000):
        input = torch.randn(1, 3, *cfg.MODEL.IMAGE_SIZE[::-1]).to(device)
        # ensure that context initialization and normal_() operations
        # finish before you start measuring time
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(input)

        torch.cuda.synchronize()  # wait for mm to finish
        end = time.perf_counter()
        run_time.append(end-start)

    run_time.pop(0)

    print('Mean running time is {:.5f}'.format(np.mean(run_time)))
    print('FPS = {:.1f}'.format(1 / np.mean(run_time)))


if __name__ == '__main__':
    test_speed()
