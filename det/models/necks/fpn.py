# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from torch.nn import functional as F

from foundation.backends.torch.utils import weight_init
from ..registry import NeckStash
from ..utils import get_norm

__all__ = ['FPN']


class LastLevelMaxPool(nn.Module):
    """This module is used in Faster-RCNN to generate a downsampled P6 feature from P5."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """This module is used in RetinaNet to generate P6 and P7 from C5 feature."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


@NeckStash.register('FPN')
class FPN(nn.Module):
    """`Feature Pyramid Networks for Object Detection`_.

    Args:
        in_channels (tuple[int], optional): List of input channels, default is output
            channels of res2, res3, res4, res5, respectively.
            Default: (256, 512, 1024, 2048)
        out_channels (int, optional): Output channels of FPN. Default: 256
        top_block (str, optional): Extra convolutions used by Faster-RCNN, RetinaNet and
            so on, can be one of 'rcnn', 'retinanet'. Default: 'rcnn'
        norm (str, optional): Name of normalization.

    .._`Feature Pyramid Networks for Object Detection`:
        https://arxiv.org/abs/1612.03144
    """

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256,
                 top_block='rcnn', norm=None):
        super(FPN, self).__init__()

        self._in_levels = len(in_channels)

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        bias = norm is None
        for channels in in_channels:
            lateral_conv = nn.Conv2d(channels, out_channels, 1, bias=bias)
            output_conv = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1,
                                    bias=bias)
            if norm is not None:
                lateral_conv = nn.Sequential(lateral_conv, get_norm(norm, out_channels))
                output_conv = nn.Sequential(output_conv, get_norm(norm, out_channels))

            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        if top_block == 'rcnn':
            self.top_block = LastLevelMaxPool()
        elif top_block == 'retinanet':
            self.top_block = LastLevelP6P7(in_channels[-1], out_channels)
        else:
            raise ValueError('top_block should be either rcnn or retinanet')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_uniform_init(
                    m, a=1, mode='fan_in', nonlinearity='leaky_relu',
                )

    def forward(self, x):
        """

        Args:
            x (tuple[Tensor]): Tuple of tensor in high to low resolution order.

        Returns:
            tuple[Tensor]: Tuple of tensor in high to low resolution order.
        """
        assert len(x) == self._in_levels

        # Build laterals
        laterals = [lateral_conv(x[i])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        for i in range(self._in_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')

        # Build outputs
        # part 1: from original levels
        outs = [self.output_convs[i](laterals[i]) for i in range(self._in_levels)]
        # part 2: from extra levels
        if isinstance(self.top_block, LastLevelMaxPool):
            outs.extend(self.top_block(outs[-1]))
        elif isinstance(self.top_block, LastLevelP6P7):
            outs.extend(self.top_block(x[-1]))

        return tuple(outs)
