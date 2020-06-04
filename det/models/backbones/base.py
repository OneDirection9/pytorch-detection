from __future__ import absolute_import, division, print_function

from abc import ABCMeta

from foundation.registry import Registry

from det import layers

__all__ = ['BackboneRegistry', 'Backbone']


class BackboneRegistry(Registry):
    """Registry of backbones."""
    pass


class Backbone(layers.Module, metaclass=ABCMeta):
    pass
