from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from typing import Dict

from foundation.registry import Registry
from torch import nn

from ..shape_spec import ShapeSpec


class NeckRegistry(Registry):
    """Registry of necks."""
    pass


class Neck(nn.Module, metaclass=ABCMeta):
    """A wrapper around :class:`torch.nn.Module` to with :attr:`output_shape` and
    :attr:`size_divisibility`.
    """

    @property
    @abstractmethod
    def output_shape(self) -> Dict[str, ShapeSpec]:
        pass

    @property
    def size_divisibility(self) -> int:
        """
        Some necks require the input height and width to be divisible by a specific integer.
        This is typically true for encoder / decoder type networks with lateral connection
        (e.g., FPN) for which feature maps need to match dimension in the "bottom up" and "top down"
        paths. Set to 0 if no specific input size divisibility is required.
        """
        return 0
