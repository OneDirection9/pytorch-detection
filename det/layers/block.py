from __future__ import absolute_import, division, print_function

from torch import nn

from .norm import FrozenBatchNorm2d

__all__ = ['CNNBlockBase']


class CNNBlockBase(nn.Module):
    """A CNN block is assumed to have input channels, output channels and a stride.

    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """
        Args:
            in_channels:
            out_channels:
            stride:
        """
        super(CNNBlockBase, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self) -> 'CNNBlockBase':
        """Makes this block not trainable.

        This method sets all parameters to `requires_grad=False`, and convert all BatchNorm layers
        to FrozenBatchNorm.

        Returns:
            The block itself.
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self
