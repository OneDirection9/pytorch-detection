from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from typing import Dict

from foundation.registry import Registry
from torch import nn

from ..shape_spec import ShapeSpec

__all__ = ['BackboneRegistry', 'Backbone']


class BackboneRegistry(Registry):
    """Registry of backbones."""
    pass


class Backbone(nn.Module, metaclass=ABCMeta):
    """A wrapper around :class:`torch.nn.Module` to :attr:`output_shape`."""

    @property
    @abstractmethod
    def output_shape(self) -> Dict[str, ShapeSpec]:
        pass
