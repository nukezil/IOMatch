import copy
import os

import numpy as np
from PIL import Image
from torchvision import transforms
import math

from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from .datasetbase import BasicDataset

mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imagenet30(args, alg, name, labeled_percent, num_classes, data_dir='./data'):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    data_dir = os.path.join(data_dir, name.lower())

    lb_dset = IN30Dataset(root=os.path.join(data_dir, "one_class_train"),
                          transform=transform_weak,
                          is_ulb=False,
                          alg=alg,
                          flist=os.path.join(data_dir, f'filelist/train_labeled_{labeled_percent}.txt'))

    ulb_dset = IN30Dataset(root=os.path.join(data_dir, "one_class_train"),
                           transform=transform_weak,
                           is_ulb=True,
                           alg=alg,
                           strong_transform=transform_strong,
                           flist=os.path.join(data_dir, f'filelist/train_unlabeled_full.txt'))

    test_dset = IN30Dataset(root=os.path.join(data_dir, "one_class_test"),
                            transform=transform_val,
                            is_ulb=False,
                            alg=alg,
                            flist=os.path.join(data_dir, f'filelist/test.txt'))

    test_data, test_targets = test_dset.data, test_dset.targets
    test_targets[test_targets >= num_classes] = num_classes
    seen_indices = np.where(test_targets < num_classes)[0]

    eval_dset = copy.deepcopy(test_dset)
    eval_dset.data, eval_dset.targets = eval_dset.data[seen_indices], eval_dset.targets[seen_indices]

    return lb_dset, ulb_dset, eval_dset, test_dset


def make_dataset(directory, class_to_idx):
    imgs = []
    targets = []
    directory = os.path.expanduser(directory)
    for target in os.listdir(directory):
        d = os.path.join(directory, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    imgs.append(path)
                    targets.append(class_to_idx[target])
    imgs = np.array(imgs)
    targets = np.array(targets)

    return imgs, targets


def make_dataset_from_list(flist):
    with open(flist) as f:
        lines = f.readlines()
        imgs = [line.split(' ')[0] for line in lines]
        targets = [int(line.split(' ')[1].strip()) for line in lines]
        imgs = np.array(imgs)
        targets = np.array(targets)
    return imgs, targets


def find_classes(directory):
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class IN30Dataset(BasicDataset):
    def __init__(self, root, transform, is_ulb, alg, strong_transform=None, flist=None):
        super(IN30Dataset, self).__init__(alg=alg, data=None, is_ulb=is_ulb,
                                          transform=transform, strong_transform=strong_transform)
        self.root = root

        classes, class_to_idx = find_classes(self.root)
        if flist is not None:
            imgs, targets = make_dataset_from_list(flist)
        else:
            imgs, targets = make_dataset(self.root, class_to_idx)

        if len(imgs) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.loader = pil_loader
        self.extensions = IMG_EXTENSIONS

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = imgs
        self.targets = targets

        self.strong_transform = strong_transform

    def __sample__(self, idx):
        path, target = self.data[idx], self.targets[idx]
        img = self.loader(path)
        return img, target
