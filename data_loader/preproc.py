import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from data_loader.autoaugment import CIFAR10Policy,SVHNPolicy,ImageNetPolicy


def data_transforms(dataset, attack=False):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    elif dataset == 'mnist':
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [
        transforms.ToTensor()
    ]
    val_trans = [
        transforms.ToTensor()
    ]
    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(val_trans)

    return train_transform, valid_transform
