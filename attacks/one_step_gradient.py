# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
from torch.autograd import Variable

from attacks.tools import clamp
from attacks.tools import normalize_by_pnorm

from .base import Attack
from .base import LabelMixin


class GradientSignAttack(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(GradientSignAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        x, y = self._verify_and_process_inputs(x, y)
        # print(y)
        xadv = x.requires_grad_()
        # xadv = Variable(x.detach(), requires_grad=True)
        # print(xadv)
        outputs = self.predict(xadv)

        if isinstance(outputs, tuple):
            logits = outputs[-2]
        else:
            logits = outputs

        loss = self.loss_fn(logits, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        # (loss / 2).backward()
        grad_sign = xadv.grad.detach().sign()

        xadv = xadv + self.eps * grad_sign

        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv


FGSM = GradientSignAttack


class GradientAttack(Attack, LabelMixin):
    """
    Perturbs the input with gradient (not gradient sign) of the loss wrt the
    input.

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3,
                 clip_min=0., clip_max=1., targeted=False):
        """
        Create an instance of the GradientAttack.
        """
        super(GradientAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad = normalize_by_pnorm(xadv.grad)
        xadv = xadv + self.eps * grad
        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv


FGM = GradientAttack
