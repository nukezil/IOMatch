import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data, split_ossl_data, reassign_target


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=False)
    data, targets = dset.data, dset.targets
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=False)
    test_data, test_targets = dset.data, dset.targets
    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset


def get_cifar_openset(args, alg, name, num_labels, num_classes, data_dir='./data', pure_unlabeled=False):
    name = name.split('_')[0]  # cifar10_openset -> cifar10
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=False)
    data, targets = dset.data, dset.targets

    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name], )
    ])

    if name == 'cifar10':
        seen_classes = set(range(2, 8))
        num_all_classes = 10
    elif name == 'cifar100':
        num_super_classes = num_classes // 5  # args.num_super_classes
        num_all_classes = 100
        super_classes = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        seen_classes = set(np.arange(num_all_classes)[super_classes < num_super_classes])
    else:
        raise NotImplementedError

    lb_data, lb_targets, ulb_data, ulb_targets = split_ossl_data(args, data, targets, num_labels, num_all_classes,
                                                                 seen_classes, None, True)

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    if pure_unlabeled:
        seen_indices = np.where(ulb_targets < num_classes)[0]
        ulb_data = ulb_data[seen_indices]
        ulb_targets = ulb_targets[seen_indices]

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_all_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=False)
    test_data, test_targets = dset.data, reassign_target(dset.targets, num_all_classes, seen_classes)
    seen_indices = np.where(test_targets < num_classes)[0]
    eval_dset = BasicDataset(alg, test_data[seen_indices], test_targets[seen_indices],
                             len(seen_classes), transform_val, False, None, False)
    test_full_dset = BasicDataset(alg, test_data, test_targets, num_all_classes, transform_val, False, None, False)
    return lb_dset, ulb_dset, eval_dset, test_full_dset
