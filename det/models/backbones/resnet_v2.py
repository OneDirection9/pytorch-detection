from __future__ import absolute_import, division, print_function

from typing import Callable, Union

import torch
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

        if stride != 1 or in_channels != out_channels:
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

        if stride != 1 or in_channels != out_channels:
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

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = layers.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=layers.get_norm(norm, bottleneck_channels),
        )
        self.conv2 = layers.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=layers.get_norm(norm, bottleneck_channels),
        )
        self.conv3 = layers.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=layers.get_norm(norm, out_channels)
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.caffe2_msra_init(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO: this somehow hurts performance when training GN models from scratch.
        #   Add it as an option when we need to use this code to train a backbone.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = None

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(layers.CNNBlockBase):
    pass
