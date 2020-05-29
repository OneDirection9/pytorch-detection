from __future__ import absolute_import, division, print_function

from typing import Callable, Union

import torch
import torch.nn as nn
from foundation.nn import weight_init
from torch.nn import functional as F

from det import layers


class BasicBlock(layers.CNNBlockBase):
    """The basic residual block for ResNet-18 and ResNet-34.

    The block has two 3x3 conv layers and a projection shortcut if needed defined in
    `Deep Residual Learning for Image Recognition`_.

    .. _`Deep Residual Learning for Image Recognition`:
        https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        norm: Union[str, Callable] = 'BN',
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride of the first conv.
            norm: Normalization for all conv layers. See :func:`get_norm` for supported format.
        """
        super(BasicBlock, self).__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = layers.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=layers.get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = layers.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=layers.get_norm(norm, out_channels),
        )

        self.conv2 = layers.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=layers.get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.caffe2_msra_init(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(layers.CNNBlockBase):
    """The standard bottleneck block used by ResNet-50, 101 and 152.

    The block has 3 conv layers with kernels 1x1, 3x3, 1x1, and a projection shortcut if needed
    defined in `Deep Residual Learning for Image Recognition`_.

    .. _`Deep Residual Learning for Image Recognition`:
        https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bottleneck_channels: int,
        stride: int = 1,
        num_groups: int = 1,
        norm: Union[str, Callable] = 'BN',
        stride_in_1x1: bool = False,
        dilation: int = 1,
    ) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bottleneck_channels: Number of output channels for the 3x3 'bottleneck' conv layers.
            stride: Stride of the first conv.
            num_groups: Number of groups for the 3x3 conv layer.
            norm: Normalization for all conv layers. See :func:`get_norm` for supported format.
            stride_in_1x1: When stride>1, whether to put stride in the first 1x1 convolution or the
                bottleneck 3x3 convolution.
            dilation: The dilation rate of the 3x3 conv layer.
        """
        super(BottleneckBlock, self).__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = layers.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=layers.get_norm(norm, out_channels),
            )
            nn.init.uniform_(self.shortcut.weight)
        else:
            self.shortcut = None
