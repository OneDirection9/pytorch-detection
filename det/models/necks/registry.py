from __future__ import absolute_import, division, print_function

from typing import Any, Dict

from foundation.registry import Registry

from det.layers import BaseModule, ShapeSpec

__all__ = ['NeckRegistry', 'Neck', 'build_neck']


class NeckRegistry(Registry):
    """Registry of necks."""
    pass


# Alias of BaseModule
Neck = BaseModule


def build_neck(cfg: Dict[str, Any], input_shape: Dict[str, ShapeSpec]) -> Neck:
    """Builds a neck from config."""
    neck_name = cfg.pop('name')
    neck = NeckRegistry.get(neck_name)(input_shape, **cfg)
    assert isinstance(neck, Neck)
    return neck
