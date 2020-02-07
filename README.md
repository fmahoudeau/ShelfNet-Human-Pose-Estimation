# ShelfNet for Human Pose Estimation

Real-time highly accurate Human Pose Estimation using ShelfNet with PyTorch. 

This repository is the result of my curiosity to find out whether ShelfNet is an efficient CNN architecture for computer vision tasks other than semantic segmentation, and more specifically for the human pose estimation task. The answer is a clear yes, with 74.6 mAP and xx FPS on the MS COCO keypoints task which represents a 4x boost in FPS for an accuracy similar to the current state of the art. 

This repository includes:

* Source code of ShelfNet modified from the authors' [repository](https://github.com/juntang-zhuang/ShelfNet/tree/pascal)

* Code to prepare the MS COCO keypoints dataset

* Training and evaluation code for MS COCO keypoints modified from the HRNet authors' [repository](https://github.com/HRNet/HRNet-Human-Pose-Estimation)

* Pre-trained weights for ShelfNet50

If you use it in your projects, please consider citing this repository (bibtex below).

 
## ShelfNet Architecture Overview

The ShelfNet architecture was introduced by J. Zhuang, J. Yang, L. Gu and N. Dvornek through a paper available on [arXiv](https://arxiv.org/abs/1811.11254). The paper only evaluates the network on the semantic segmentation task. The authors' contribution is to have created a fast architecture with similar performance to the state of the art (PSPNet & EncNet) on PASCAL VOC and better performance on Cityscapes. Compared to other architectures, ShelfNet is more suitable for real-world applications with resource constraints.

![ShelfNet Architecture](assets/ShelfNet_Architecture.jpg)

As depicted above, ShelfNet uses a ResNet backbone combined with 2 encoder/decoder branches. The first encoder (in green) reduces channel complexity by a factor 4 for faster inference speed. The S-block is a residual block with shared-weights to significantly reduce the number of parameters. The network uses strided convolutions for down-sampling and transpose convolutions for up-sampling. The structure can be seen as an ensemble of [FCN](https://github.com/fmahoudeau/fcn) where the information flows through many different paths, resulting in increased accuracy.


## Results on Microsoft COCO KeyPoints

This section reports test results for ShelfNet50 on the famous [MS COCO KeyPoints](http://cocodataset.org/#keypoints-2019) dataset, and makes a comparison with the state of the art HRNet. All experiments use the same person detector with a mAP of 0.56. A single Titan RTX with 24GB RAM was used. The batch size for ShelfNet50 is 128 for an input size of 256x192 and 72 for 384x288.


| Architecture            | Input size  | Parameters  |    AP   |    AR   | Memory size  |   FPS   |
|-------------------------|-------------|-------------|---------|---------|--------------|---------|
| pose_hrnet_w32          | 256x192     | 28.5M       |  0.744  |  0.798  | xxx MB       | xxxx    |
| pose_hrnet_w32          | 384x288     | 28.5M       |  0.758  |  0.809  | xxx MB       | xxxx    |
| pose_hrnet_w48          | 256x192     | 63.6M       |  0.751  |  0.804  | xxx MB       | xxxx    |
| pose_hrnet_w48          | 384x288     | 63.6M       |  0.763  |  0.812  | xxx MB       | xxxx    |
| shelfnet_50             | 256x192     | xx.xM       |  0.725  |  0.782  | xxx MB       | xxxx    |
| shelfnet_50             | 384x288     | xx.xM       |  0.746  |  0.797  | xxx MB       | xxxx    |


## Training on Your Own

 

## Requirements

Python 3.7, Torch 1.3.1 or greater, requests, tqdm, yacs, json_tricks, and pycocotools.
Contrary to the ShelfNet repository, this repository is not based on torch-encoding.


## Citation

Use this bibtex to cite this repository:
```
@misc{fmahoudeau_shelfnet_human_pose_2020,
  title={ShelfNet for Human Pose Estimation},
  author={Florent Mahoudeau},
  year={2020},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/fmahoudeau/ShelfNet-Human-Pose-Estimation}},
}
