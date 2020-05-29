from models.darts_cell_all import *
import utils.genotypes_all as gt
import numpy as np
from scipy.special import softmax
from models.base_module import MyModule, BaseSearchModule
import itertools
import random
import copy
import torch
from collections import namedtuple
import torch.nn.functional as F
from attacks.advtrain import Normalize


# training cnn
class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(MyModule):

    def __init__(self, C_in, C, num_classes, layers, auxiliary, pcl, genotype, drop_out=0, drop_path=0, input_size=None, dataset='', mannul=True):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self._pcl = pcl
        self._dropout = drop_out
        self.drop_path_prob = drop_path
        self.normalize = Normalize(dataset, C_in, input_size)

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        if 'tiny' not in dataset:
            self.stem = nn.Sequential(
                nn.Conv2d(C_in, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(C_in, C_curr // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(C_curr // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )

        C_prev, C_curr = C_curr, C
        self.cells = nn.ModuleList()
        denosing = False
        proxy_len = len(genotype) // 2
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            if mannul:
                # Denosing in 1/4, 2/4 and 3/4 layer.
                if i in [layers // 4, 2 * layers // 4, 3 * layers // 4]:
                    denosing = True
                else:
                    denosing = False

            if proxy_len == self._layers:
                k = i
            else:
                n_repeat = (self._layers - 2) // (proxy_len - 2)

                if i <= layers // 3 + 1:
                    k = (i + 0) // n_repeat
                elif i <= 2 * layers // 3 + 1:
                    k = (i + 1) // n_repeat
                else:
                    k = (i + 2) // n_repeat

            genotype_ = list(itertools.chain.from_iterable(genotype[2*k]))
            concat = genotype[2*k + 1]

            cell = Training_Cell(genotype_, concat, C_prev, C_curr, reduction, denosing)
            self.cells += [cell]
            C_prev = cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if self._dropout > 0:
            self.dropout = nn.Dropout(p=self._dropout)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        input = self.normalize(input)
        s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s1 = cell(s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        if self._dropout > 0:
            out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

