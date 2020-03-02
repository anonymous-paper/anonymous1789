import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def gabor_fn(kernel_size, channel_in, channel_out, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma    # [channel_out]
    sigma_y = sigma.float() / gamma     # element-wize division, [channel_out]

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1)
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float().cuda()
    x_0 = torch.arange(xmin, xmax+1)
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float().cuda()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    # [channel_out, channel_in, kernel, kernel]
    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb


class GaborFilter(nn.Module):
    def __init__(self, channel_in, kernel_size, stride=1, padding=1, dilation=1):
        super(GaborFilter, self).__init__()
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_in
        channel_out = channel_in
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.Lambda = nn.Parameter(torch.rand(channel_out), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.psi = nn.Parameter(torch.randn(channel_out) * 0.02, requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(channel_out) * 1.0, requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(channel_out) * 0.0, requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        theta = self.sigmoid(self.theta) * math.pi * 2.0
        gamma = 1.0 + (self.gamma * 0.5)
        sigma = 0.1 + (self.sigmoid(self.sigma) * 0.4)
        Lambda = 0.001 + (self.sigmoid(self.Lambda) * 0.999)
        psi = self.psi

        kernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, sigma, theta, Lambda, psi, gamma)
        # print(self.kernel_size)
        kernel = kernel.float()   # [channel_out, channel_in, kernel, kernel]
        # print(kernel.shape)

        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

        return out


class NonLocalFilter(nn.Module):
    def __init__(self, in_channels, affine=True, embed=True, softmax=True):
        super(NonLocalFilter, self).__init__()

        self.embed = embed
        self.softmax = softmax

        self.in_channels = in_channels

        self.inter_channels = in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        if self.embed:
            self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                                kernel_size=1, stride=1, padding=0)
            nn.init.normal_(self.theta.weight, std=0.01)

            self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)
            nn.init.normal_(self.phi.weight, std=0.01)
        
        # if not softmax:
        #     self.bn = nn.BatchNorm2d(self.in_channels, affine=affine) # affine


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size, channel_in, h, w = x.size()

        # print(x.shape)
        # if x.max() > 1e5:
            # print('0', x.max())
        # print('0', x.max())
        # print('0', x[0,0])

        if self.embed:
            theta_x = self.theta(x)
            phi_x = self.phi(x)
            g_x = x
        else:
            theta_x, phi_x, g_x = x, x, x

        # theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.view(batch_size, -1, h*w)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.view(batch_size, -1, h*w)

        g_x = g_x.view(batch_size, self.in_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.in_channels > h * w or self.softmax:
            f = torch.matmul(theta_x, phi_x) # hw x c, c x hw -> hw x hw
            if self.softmax:
                f = f / math.sqrt(self.inter_channels) # torch.sqrt(torch.tensor(self.inter_channels).float())
                f = F.softmax(f, dim=-1)
            # else:
                # print(f.min())
                # print(f.max())
                # print(f.shape)
                # f = f / math.sqrt(self.inter_channels) # torch.sqrt(torch.tensor(self.inter_channels).float())
                # f = F.softmax(f, dim=-1)
                # print(torch.min(f, -1, keepdim=True)[0])

                # m = f.min(-1, keepdim=True)[0]
                # f = (f - m) / (f.max(-1, keepdim=True)[0] - m)
                # f = f / torch.sum(f, dim=-1, keepdim=True)

                # print(f[1].sum(-1, keepdim=True).shape)
                # print(f.min())
                # print(f.max())
            y = torch.matmul(f, g_x) # hw x hw, hw x c -> hw x c
        # why?
        else:
            f = torch.matmul(phi_x, g_x) # c x hw,  hw x c -> c x c
            y = torch.matmul(theta_x, f) # hw x c, c x c -> hw x c

        # if not self.softmax:
        #     N = y.size(-1) # 错的！！！！！！
        #     y = y / N

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.in_channels, *x.size()[2:])

        if not self.softmax:
            # N = y.size(-1) # 对的！！！！！！
            N = channel_in * h * w
            y = y / N #  / (N*100)
            # y = self.bn(y / N) #  / (N*100)

            # if y.max() > 1e5:
            #     print('1', y.max())
            # print('1', y.max())
        # print('0', x.max())
        # print('1', y.max())
        # print('1', y[0,0])
        # print('1', y[0,0] - x[0,0])

        # print()

        return y
