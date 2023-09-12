import os
import math
import logging
import random
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from semilearn.datasets.utils import get_collector, name2sampler
from semilearn.nets.utils import param_groups_layer_decay, param_groups_weight_decay


def get_net_builder(net_name, from_name: bool):
    """
    built network according to network name
    return **class** of backbone network (not instance).

    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
    """
    if from_name:
        import torchvision.models as nets
        model_name_list = sorted(name for name in nets.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(nets.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return nets.__dict__[net_name]
    else:
        # TODO: fix bug here
        import semilearn.nets as nets
        builder = getattr(nets, net_name)
        return builder


def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def get_dataset(args, algorithm, dataset, num_labels, num_classes, data_dir='./data',
                include_lb_to_ulb=True, eval_open=False):
    """
    create dataset

    Args
        args: argparse arguments
        algorithm: algrorithm name, used for specific return items in __getitem__ of datasets
        dataset: dataset name 
        num_labels: number of labeled data in dataset
        num_classes: number of classes
        seed: random seed
        data_dir: data folder
    """
    from semilearn.datasets import get_cifar_openset, get_imagenet30
    from semilearn.datasets import svhn_as_ood, lsun_as_ood, gaussian_as_ood, uniform_as_ood
    get_ood_funcs = {'svhn': svhn_as_ood, 'lsun': lsun_as_ood, 'gaussian': gaussian_as_ood, 'uniform': uniform_as_ood}

    test_dset = None
    if dataset in ["cifar10_openset", "cifar100_openset"]:
        lb_dset, ulb_dset, eval_dset, eval_full_dset = get_cifar_openset(args, algorithm, dataset, num_labels,
                                                                         num_classes, data_dir=data_dir)
        if eval_open:
            test_dset = {'full': eval_full_dset}
    elif dataset == 'imagenet30':
        lb_dset, ulb_dset, eval_dset, eval_full_dset = get_imagenet30(args, algorithm, dataset, args.labeled_percent,
                                                                      num_classes, data_dir=data_dir)
        if eval_open:
            test_dset = {'full': eval_full_dset}
    else:
        raise NotImplementedError

    if dataset.split('_')[-1] == 'openset' and eval_open:
        num_existing_classes = test_dset['full'].num_classes
        ood_names = ['svhn', 'lsun', 'gaussian', 'uniform']
        ood_dsets = []
        for ood_name in ood_names:
            dset = get_ood_funcs[ood_name](args, data_dir=data_dir, len_per_dset=10000)
            dset.targets = dset.targets + num_existing_classes
            num_existing_classes += dset.num_classes
            ood_dsets.append(dset)
        test_dset['extended'] = ConcatDataset(ood_dsets)
        test_dset['ood_dsets'] = ood_dsets

    dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'test': test_dset}
    return dataset_dict


def get_data_loader(args,
                    dset,
                    batch_size=None,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    data_sampler='RandomSampler',
                    num_epochs=None,
                    num_iters=None,
                    generator=None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """

    assert batch_size is not None
    if num_epochs is None:
        num_epochs = args.epoch
    if num_iters is None:
        num_iters = args.num_train_iter
        
    collect_fn = get_collector(args, args.net)

    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collect_fn,
                          num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)

    if isinstance(data_sampler, str):
        data_sampler = name2sampler[data_sampler]

        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            num_replicas = 1
            rank = 0

        per_epoch_steps = num_iters // num_epochs
        num_samples = per_epoch_steps * batch_size * num_replicas

        return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collect_fn,
                          pin_memory=pin_memory, sampler=data_sampler(dset, num_replicas=num_replicas, rank=rank, num_samples=num_samples),
                          generator=generator, drop_last=drop_last)

    elif isinstance(data_sampler, torch.utils.data.Sampler):
        return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          collate_fn=collect_fn, pin_memory=pin_memory, sampler=data_sampler,
                          generator=generator, drop_last=drop_last)

    else:
        raise Exception(f"unknown data sampler {data_sampler}.")


def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, layer_decay=1.0, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    assert layer_decay <= 1.0

    no_decay = {}
    if hasattr(net, 'no_weight_decay') and bn_wd_skip:
        no_decay = net.no_weight_decay()
    
    if layer_decay != 1.0:
        per_param_args = param_groups_layer_decay(net, lr, weight_decay, no_weight_decay_list=no_decay, layer_decay=layer_decay)
    else:
        per_param_args = param_groups_weight_decay(net, weight_decay, no_weight_decay_list=no_decay)

    if optim_name == 'SGD':
        optimizer = torch.optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=nesterov)
    elif optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)

    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    from torch.optim.lr_scheduler import LambdaLR
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def get_port():
    """
    find a free port to used for distributed learning
    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(15000, 30000)
    if tt not in procarr:
        return tt
    else:
        return get_port()
