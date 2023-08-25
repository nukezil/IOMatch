import os
import random
import numpy as np
import torch
from torch.utils.data import sampler, DataLoader
import torch.distributed as dist
from semilearn.datasets.samplers import DistributedSampler
from io import BytesIO

# TODO: better way
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


name2sampler = {'RandomSampler': DistributedSampler}


def split_ssl_data(args, data, targets, num_classes,
                   lb_num_labels, ulb_num_labels=None,
                   lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                   lb_index=None, ulb_index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes, 
                                                    lb_num_labels, ulb_num_labels,
                                                    lb_imbalance_ratio, ulb_imbalance_ratio)
    
    # manually set lb_idx and ulb_idx, do not use except for debug
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    
    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]


def sample_labeled_data(args, data, target, num_labels, num_classes, index=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    assert num_labels % num_classes == 0
    if index is not None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    dump_dir = os.path.join(base_dir, 'data', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, f'labels{args.num_labels}_seed{args.seed}_idx.npy')

    if os.path.exists(dump_path):
        lb_idx = np.load(dump_path)
        lb_data = data[lb_idx]
        lbs = target[lb_idx]
        return lb_data, lbs, lb_idx

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])

    np.save(dump_path, np.array(lb_idx))

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)


def sample_labeled_unlabeled_data(args, data, target, num_classes,
                                  lb_num_labels, ulb_num_labels=None,
                                  lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                                  load_exist=False):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    dump_dir = os.path.join(base_dir, 'data', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    lb_dump_path = os.path.join(dump_dir, f'lb_labels{args.num_labels}_seed{args.seed}_idx.npy')
    ulb_dump_path = os.path.join(dump_dir, f'ulb_labels{args.num_labels}_seed{args.seed}_idx.npy')

    if os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
        return lb_idx, ulb_idx
    
    # get samples per class
    # balanced setting, lb_num_labels is total number of labels for labeled data
    assert lb_num_labels % num_classes == 0, "lb_num_labels must be divideable by num_classes in balanced setting"
    lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes

    if ulb_num_labels is None:
        ulb_samples_per_class = [int(len(data) / num_classes) - lb_samples_per_class[c] for c in range(num_classes)]
    else:
        # ulb_num_labels must be dividable by num_classes in balanced setting
        assert ulb_num_labels % num_classes == 0
        ulb_samples_per_class = [int(ulb_num_labels / num_classes)] * num_classes

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c]+ulb_samples_per_class[c]])
    
    if isinstance(lb_idx, list):
        lb_idx = np.asarray(lb_idx)
    if isinstance(ulb_idx, list):
        ulb_idx = np.asarray(ulb_idx)

    np.save(lb_dump_path, lb_idx)
    np.save(ulb_dump_path, ulb_idx)
    
    return lb_idx, ulb_idx


def reassign_target(target, num_all_classes, seen_classes):
    target = np.array(target)

    all_classes = set(range(num_all_classes))
    unseen_classes = all_classes - seen_classes
    targets_new = np.ones_like(target) * (-1)  # assign new labels for OSSL, first k classes are seen-class

    for i, lbi in enumerate(seen_classes):
        all_lbi_indices = np.where(target == lbi)[0]
        targets_new[all_lbi_indices] = i
    for i, lbi in enumerate(unseen_classes):
        all_lbi_indices = np.where(target == lbi)[0]
        targets_new[all_lbi_indices] = len(seen_classes) + i

    return targets_new


def split_ossl_data(args, data, target, num_labels, num_all_classes, seen_classes, index=None, include_lb_to_ulb=True):
    # seen_classes should be a set
    data, target = np.array(data), np.array(target)
    target = reassign_target(target, num_all_classes, seen_classes)
    lb_data, lbs, lb_idx, = sample_labeled_data(args, data, target, num_labels, len(seen_classes), index)
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]


def get_collector(args, net):
    collect_fn = None
    return collect_fn


def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = random.randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]