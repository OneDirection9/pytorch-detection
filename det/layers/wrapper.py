from __future__ import absolute_import, division, print_function

import torch
from torch import nn

__all__ = ['Conv2d']


class Conv2d(nn.Conv2d):
    """A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features."""

    def __init__(self, *args, **kwargs) -> None:
        """Extra keyword arguments supported in addition to those in :class:`torch.nn.Conv2d`.

        Args:
            norm (nn.Module, optional): A normalization layer.
            activation (callable(Tensor) -> Tensor): A callable activation function.
        """
        norm = kwargs.pop('norm', None)
        activation = kwargs.pop('activation', None)

        super(Conv2d, self).__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), 'SyncBatchNorm does not support empty inputs!'

        x = super(Conv2d, self).forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
