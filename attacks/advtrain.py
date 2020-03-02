import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


meanstd_dataset = {
    'cifar10': [[0.49139968, 0.48215827, 0.44653124],
                [0.24703233, 0.24348505, 0.26158768]],
    'mnist': [[0.13066051707548254],
                [0.30810780244715075]],
    'fashionmnist': [[0.28604063146254594],
                [0.35302426207299326]],
    'imagenet': [[0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]],
    'tinyimagenet': [[0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]],
    'tinyimagenet_64': [[0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]],
}


class Normalize(nn.Module):
    def __init__(self, dataset, input_channels, input_size):
        super(Normalize, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels

        self.mean, self.std = meanstd_dataset[dataset.lower()]
        self.mean = torch.Tensor(np.array(self.mean)[:, np.newaxis, np.newaxis]).cuda()
        self.mean = self.mean.expand(self.input_channels, self.input_size, self.input_size).cuda()
        self.std = torch.Tensor(np.array(self.std)[:, np.newaxis, np.newaxis]).cuda()
        self.std = self.std.expand(self.input_channels, self.input_size, self.input_size).cuda()
    
    def forward(self, input):
        device = input.device
        output = input.sub(self.mean.to(device)).div(self.std.to(device))
        return output
