""" Operations """
import torch
import torch.nn as nn
#import utils.genotypes_1 as gt
import numpy as np
import torch.nn.functional as F
from models.filter import GaborFilter, NonLocalFilter


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'sep_conv_m_3x3': lambda C, stride, affine: SepConv_M(C, C, 3, stride, 1, affine=affine),
    'sep_conv_m_5x5': lambda C, stride, affine: SepConv_M(C, C, 5, stride, 2, affine=affine),
    'conv_3x3': lambda C, stride, affine: Conv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
    'conv_5x5': lambda C, stride, affine: Conv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine),
    'gab_filt_3x3': lambda C, stride, affine: DenoisingBlock(C, stride, filter_type='gabor', affine=affine), # 3x3
    'dtp_blok_3x3': lambda C, stride, affine: DenoisingBlock(C, stride, filter_type='dotp', affine=affine), # 3x3
    # 'gas_blok_3x3': lambda C, stride, affine: DenoisingBlock(C, stride, filter_type='gaus', affine=affine) # 3x3
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
        # GFilter(C_in, C_in, 3, 1, 1),
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)
    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            # GFilter(C_in, C_in, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, groups=1, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),

            # nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation, groups=groups,
            #           bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class Conv(nn.Module):
    """ (ated) depthwise separable conv
    ReLU - (ated) depthwise separable - Pointwise - BN
    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, groups=1, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    Conv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            Conv(C_in, C_in, kernel_size, stride, padding, dilation=1, groups=C_in, affine=affine),
            Conv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv_M(nn.Module):
    """ Depthwise separable conv
    Conv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, groups=C_in, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class GFilter(nn.Module):
    """ Gabor Filter
    Conv(dilation=1) * 2
    """
    def __init__(self, C_in, kernel_size=3, stride=1, padding=1, affine=True):
        super().__init__()
        self.stride = stride
        # print(kernel_size)
        self.filter = nn.Sequential(
            GaborFilter(C_in, kernel_size, stride, padding),
            # GaborFilter(C_in, kernel_size, 1, padding),
            nn.Conv2d(C_in, C_in, 1, 1),
            nn.BatchNorm2d(C_in, affine=affine)
        )
        if stride != 1 or C_in != C_in:
            self.short = nn.Sequential(
                nn.Conv2d(C_in, C_in, 1, stride),
                nn.BatchNorm2d(C_in, affine=affine)
            )

    def forward(self, x):
        out = self.filter(x)
        out += x if self.stride == 1 else self.short(x)
        return out


filters = {
    'gaus': lambda C, affine: NonLocalFilter(C, affine, embed=True, softmax=True),
    # 'dotp': lambda C, affine: NonLocalFilter(C, affine, embed=True, softmax=True),
    'dotp': lambda C, affine: NonLocalFilter(C, affine, embed=False, softmax=False),
    'gabor': lambda C, affine: GaborFilter(C, 3, 1, 1),
    'mean': lambda C, affine: nn.AvgPool2d(3, 1, 1),
}

class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, stride, filter_type='dotp', affine=True):
        super().__init__()
        self.stride = stride
        self.filter = filters[filter_type.lower()](in_channels, affine)
        self.filterblock = nn.Sequential(
            # NonLocalFilter(in_channels, embed=True, softmax=True),
            self.filter,
            nn.BatchNorm2d(in_channels, affine=affine)
        )
        # print(self.filterblock)
        # print(self.filter)
        # print(self.filterblock[1])
        # print(self.filterblock[1].weight)
        if affine:
            nn.init.constant_(self.filterblock[1].weight, 0)
            nn.init.constant_(self.filterblock[1].bias, 0)

        # if stride != 1:
        #     self.short = nn.Sequential(
        #         nn.Conv2d(in_channels, in_channels, 1, stride),
        #         nn.BatchNorm2d(in_channels, affine=affine)
        #     )

    def forward(self, x):
        out = self.filterblock(x)
        out += x # if self.stride == 1 else self.short(x)
        out = out if self.stride == 1 else \
            F.interpolate(out, scale_factor=1/self.stride, mode="bilinear", align_corners=True)
        return out


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        # self.filter = GFilter(C_in, C_in, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        # x = self.filter(x)
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class SelectOp_PC(nn.Module):
    """ Mixed PC operation """

    def __init__(self, C, stride, M=4):
        super(SelectOp_PC, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2,2)
        self.M = M

        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C // self.M, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // self.M, affine=False))
            self._ops.append(op)


    def forward(self, x, weights):
        #channel proportion k=4  
        dim_2 = x.shape[1]
        xtemp = x[ : , :  dim_2 // self.M, :, :]
        xtemp2 = x[ : ,  dim_2 // self.M:, :, :]
        # temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        temp1 = self._ops[weights](xtemp)
        #reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1,xtemp2],dim=1)
        else:
            ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, self.M)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        #except channe shuffle, channel shift also works
        return ans


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class SelectOp(nn.Module):
    """ Mixed operation """

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        op = self._ops[weights]
        return op(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out