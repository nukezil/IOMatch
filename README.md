# IOMatch for Open-Set Semi-Supervised Learning

## Introduction

This is the official repository for our **ICCV 2023** paper:

> **IOMatch: Simplifying Open-Set Semi-Supervised Learning with Joint Inliers and Outliers Utilization**</br>
> Zekun Li, Lei Qi, Yinghuan Shi*, Yang Gao</br>

[[`Paper`](https://arxiv.org/abs/2308.13168)] [[`Poster`]](./pubs/Poster.pdf) [[`Slides`]](./pubs/Slides.pdf) [[`Models and Logs`](https://drive.google.com/drive/folders/1pLU6tqxMls55CBRvCgZmDBfHLXm7jGMv?usp=sharing)] [[`BibTeX`](#citation)]

## Preparation

### Required Packages

We suggest first creating a conda environment:

```sh
conda create --name iomatch python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Datasets

Please put the datasets in the ``./data`` folder (or create soft links) as follows:
```
IOMatch
├── config
    └── ...
├── data
    ├── cifar10
        └── cifar-10-batches-py
    └── cifar100
        └── cifar-100-python
    └── imagenet30
        └── filelist
        └── one_class_test
        └── one_class_train
    └── ood_data
├── semilearn
    └── ...
└── ...  
```

The data of ImageNet-30 can be downloaded in [one_class_train](https://drive.google.com/file/d/1B5c39Fc3haOPzlehzmpTLz6xLtGyKEy4/view) and [one_class_test](https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view).

The out-of-dataset testing data for extended open-set evaluation can be downloaded in [this link](https://drive.google.com/drive/folders/1IjDLYfpfsMVuzf_NmqQPoHDH0KAd94gn?usp=sharing).

## Usage

We implement [IOMatch](./semilearn/algorithms/iomatch/iomatch.py) using the codebase of [USB](https://github.com/microsoft/Semi-supervised-learning).

### Training

Here is an example to train IOMatch on CIFAR-100 with the seen/unseen split of "50/50" and 25 labels per seen class (*i.e.*, the task <u>CIFAR-50-1250</u> with 1250 labeled samples in total). 

```sh
# seed = 1
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_1250_1.yaml
```

Training IOMatch on other datasets with different OSSL settings can be specified by a config file:
```sh
# CIFAR10, seen/unseen split of 6/4, 25 labels per seen class (CIFAR-6-150), seed = 1  
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar10_150_1.yaml

# CIFAR100, seen/unseen split of 50/50, 4 labels per seen class (CIFAR-50-200), seed = 1  
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_200_1.yaml

# CIFAR100, seen/unseen split of 80/20, 4 labels per seen class (CIFAR-80-320), seed = 1    
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_cifar100_320_1.yaml

# ImageNet30, seen/unseen split of 20/10, 1% labeled data (ImageNet-20-p1), seed = 1  
CUDA_VISIBLE_DEVICES=0 python train.py --c config/openset_cv/iomatch/iomatch_in30_p1_1.yaml
```

### Evaluation

After training, the best checkpoints will be saved in ``./saved_models``. The closed-set performance has been reported in the training logs. For the open-set evaluation, please see [``evaluate.ipynb``](./evaluate.ipynb).

## Example Results

### Close-Set Classification Accuracy

#### CIFAR-10, seen/unseen split of 6/4, 4 labels per seen class (CIFAR-6-24)

| CIFAR-6-24 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch    | 90.70  | 75.15  | 78.90  | 81.58   | 6.63 |
| OpenMatch   | 42.05  | 48.18  | 40.67  | 43.63   | 3.26 |
| IOMatch     | 89.28  | 87.40  | 92.35  | 89.68   | 2.04 |

#### CIFAR-10, seen/unseen split of 6/4, 25 labels per seen class (CIFAR-6-150)

| CIFAR-6-150 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch    | 93.67  | 91.83  | 93.32  | 92.94   | 0.80 |
| OpenMatch   | 65.00  | 64.90  | 68.90  | 66.27   | 1.86 |
| IOMatch     | 94.05  | 93.88  | 93.67  | 93.87   | 0.16 |

#### CIFAR-100, seen/unseen split of 20/80, 4 labels per seen class (CIFAR-20-80)

| CIFAR-20-80 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch    | 45.80  | 46.00  | 47.00  | 46.27   | 0.64 |
| OpenMatch | 34.45 | 38.35 | 39.55 | 37.45 | 2.67 |
| IOMatch | 52.85 | 52.20 | 56.15 | 53.73 | 2.12 |

#### CIFAR-100, seen/unseen split of 20/80, 25 labels per seen class (CIFAR-20-500)

| CIFAR-20-500 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 66.00 | 66.05 | 67.30 | 66.45 | 0.74 |
| OpenMatch | 60.85 | 62.90 | 64.35 | 62.70 | 1.76 |
| IOMatch | 67.00 | 66.35 | 68.50 | 67.28 | 1.10 |

#### CIFAR-100, seen/unseen split of 50/50, 4 labels per seen class (CIFAR-50-200)

| CIFAR-50-200 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 48.80 | 43.94 | 54.04 | 48.93 | 5.05 |
| OpenMatch | 33.36 | 34.12 | 33.74 | 33.74 | 0.38 |
| IOMatch | 54.10 | 56.14 | 58.68 | 56.31 | 2.29 |

#### CIFAR-100, seen/unseen split of 50/50, 25 labels per seen class (CIFAR-50-1250)

| CIFAR-50-1250 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 67.82 | 68.92 | 69.58 | 68.77 | 0.89 |
| OpenMatch | 66.44 | 66.04 | 67.10 | 66.53 | 0.54 |
| IOMatch | 69.16 | 69.84 | 70.32 | 69.77 | 0.58 |

#### CIFAR-100, seen/unseen split of 80/20, 4 labels per seen class (CIFAR-80-320)

| CIFAR-80-320 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 44.45 | 42.36 | 42.36 | 43.06 | 1.21 |
| OpenMatch | 29.23 | 29.18 | 27.21 | 28.54 | 1.15 |
| IOMatch | 51.86 | 49.89 | 50.73 | 50.83 | 0.99 |

#### CIFAR-100, seen/unseen split of 80/20, 25 labels per seen class (CIFAR-80-2000)

| CIFAR-80-2000 | Seed=0 | Seed=1 | Seed=2 | Mean | Std. |
|:-------------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| FixMatch | 65.02 | 64.06 | 64.25 | 64.44 | 0.51 |
| OpenMatch | 62.11 | 61.09 | 60.50 | 61.23 | 0.81 |
| IOMatch | 65.31 | 64.28 | 64.65 | 64.75 | 0.52 |


## Acknowledgments

We sincerely thank the authors of [USB (NeurIPS'22)](https://github.com/microsoft/Semi-supervised-learning) for creating such an awesome SSL benchmark.

We sincerely thank the authors of the following projects for sharing the code of their great works:

- [UASD (AAAI'20)](https://github.com/yanbeic/ssl-class-mismatch)
- [DS3L (ICML'20)](https://github.com/guolz-ml/DS3L)
- [MTC (ECCV'20)](https://github.com/YU1ut/Multi-Task-Curriculum-Framework-for-Open-Set-SSL)
- [T2T (ICCV'21)](https://github.com/huangjk97/T2T)
- [OpenMatch (NeurIPS'21)](https://github.com/VisionLearningGroup/OP_Match)

## License

This project is licensed under the terms of the MIT License.
See the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@inproceedings{iomatch,
  title={IOMatch: Simplifying Open-Set Semi-Supervised Learning with Joint Inliers and Outliers Utilization},
  author={Li, Zekun and Qi, Lei and Shi, Yinghuan and Gao, Yang},
  booktitle={ICCV},
  year={2023}
}
```
