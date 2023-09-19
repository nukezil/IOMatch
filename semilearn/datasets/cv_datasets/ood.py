import os
import torchvision
import numpy as np

from torchvision import transforms
from .datasetbase import BasicDataset

mean, std = {}, {}
mean['svhn'] = [0.4380, 0.4440, 0.4730]
std['svhn'] = [0.1751, 0.1771, 0.1744]
img_size = 32


def svhn_as_ood(args, data_dir, len_per_dset=-1):
    crop_size = args.img_size

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['svhn'], std['svhn'])
    ])

    data_dir = os.path.join(data_dir, 'ood_data/svhn')
    dset = torchvision.datasets.SVHN(data_dir, split='test', download=False)
    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels  # data converted to [H, W, C] for PIL
    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]

    eval_dset = BasicDataset(args.algorithm, data, targets, 10, transform_val, False, None, False)

    return eval_dset


def lsun_as_ood(args, data_dir, len_per_dset=-1):
    crop_size = args.img_size

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor()
    ])

    data_dir = os.path.join(data_dir, 'ood_data')
    data = np.load(os.path.join(data_dir, 'LSUN_resize.npy'))
    targets = np.zeros(data.shape[0], dtype=int)

    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]

    eval_dset = BasicDataset(args.algorithm, data, targets, 1, transform_val, False, None, False)

    return eval_dset


def gaussian_as_ood(args, data_dir, len_per_dset=-1):
    crop_size = args.img_size

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor()
    ])

    data_dir = os.path.join(data_dir, 'ood_data')
    data = np.load(os.path.join(data_dir, 'Gaussian.npy'))
    targets = np.zeros(data.shape[0], dtype=int)

    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]

    eval_dset = BasicDataset(args.algorithm, data, targets, 1, transform_val, False, None, False)

    return eval_dset


def uniform_as_ood(args, data_dir, len_per_dset=-1):
    crop_size = args.img_size

    transform_val = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.Resize(crop_size),
        transforms.ToTensor()
    ])

    data_dir = os.path.join(data_dir, 'ood_data')
    data = np.load(os.path.join(data_dir, 'Uniform.npy'))
    targets = np.zeros(data.shape[0], dtype=int)

    total_len = data.shape[0]

    if len_per_dset > 0:
        idx = np.random.choice(total_len, len_per_dset, False)
        data, targets = data[idx], targets[idx]

    eval_dset = BasicDataset(args.algorithm, data, targets, 1, transform_val, False, None, False)

    return eval_dset


