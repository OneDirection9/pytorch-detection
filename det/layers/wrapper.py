from __future__ import absolute_import, division, print_function

import copy
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Any, Dict, Optional

import torch
from torch import nn

__all__ = ['Conv2d', 'ShapeSpec', 'BaseModule', 'nonzero_tuple']


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


class ShapeSpec(namedtuple('_ShapeSpec', ['channels', 'height', 'width', 'stride'])):
    """A simple structure that contains basic shape specification about a tensor.

    It is often used as the auxiliary inputs/outputs of models, to obtain the shape inference
    ability among PyTorch modules.

    Attributes:
        channels (int):
        height (int):
        width (int):
        stride (int):
    """

    def __new__(
        cls,
        *,
        channels: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> 'ShapeSpec':
        return super(ShapeSpec, cls).__new__(cls, channels, height, width, stride)

    def __copy__(self):
        return ShapeSpec(
            channels=self.channels, height=self.height, width=self.width, stride=self.stride
        )

    def __deepcopy__(self, memodict={}):
        return ShapeSpec(
            channels=copy.deepcopy(self.channels),
            height=copy.deepcopy(self.height),
            width=copy.deepcopy(self.width),
            stride=copy.deepcopy(self.stride),
        )


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Abstract base class for backbone, neck, and so on."""

    @abstractmethod
    def forward(self, *input: Any) -> Dict[str, torch.Tensor]:
        """Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: Mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    @abstractmethod
    def output_shape(self) -> Dict[str, ShapeSpec]:
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0


def nonzero_tuple(x: torch.Tensor) -> torch.Tensor:
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if x.dim() == 0:
        return x.unsqueeze(0).nonzero().unbind(1)
    return x.nonzero().unbind(1)
