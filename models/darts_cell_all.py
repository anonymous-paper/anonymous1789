""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import darts_ops_1 as ops
import torch.nn.functional as F
import numpy as np
import itertools
from models.darts_ops_1 import DenoisingBlock


# training cell
class Training_Cell(nn.Module):

    def __init__(self, genotype, concat, C_prev, C, reduction, denosing=False):
        super(Training_Cell, self).__init__()
        self.reduction = reduction
        self.denosing = denosing
        self.preprocess1 = ops.ReLUConvBN(C_prev, C, 1, 1, 0)

        op_names, indices = zip(*genotype)
        self._compile(C, op_names, indices, concat, reduction)

        if self.denosing:
            self.denoseblock = DenoisingBlock(C*len(op_names), 1, filter_type='dotp') # gaus dotp

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 1 # 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 1 else 1
            op = ops.OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, x, drop_prob):
        s1 = self.preprocess1(x)

        states = [s1]
        for i in range(self._steps):
            h1 = states[self._indices[1 * i]]
            op1 = self._ops[1 * i]
            h1 = op1(h1)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, ops.Identity):
                    h1 = ops.drop_path_(h1, drop_prob)
            s = h1 
            states += [s]
        out = torch.cat([states[i] for i in self._concat], dim=1)
        if self.denosing:
            out = self.denoseblock(out)
        return out