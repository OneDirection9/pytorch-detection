from __future__ import absolute_import, division, print_function

from typing import Any, Dict

from foundation.registry import Registry

from det.layers import BaseModule, ShapeSpec

__all__ = ['Neck', 'NeckRegistry', 'build_neck']

# Alias of BaseModule
Neck = BaseModule


class NeckRegistry(Registry):
    """Registry of necks."""
    pass


def build_neck(cfg: Dict[str, Any], input_shape: Dict[str, ShapeSpec]) -> Neck:
    """Builds a neck from config.

    Args:
        cfg:
        input_shape: Output shape of backbone.
    """
    neck_name = cfg.pop('name')
    neck = NeckRegistry.get(neck_name)(input_shape, **cfg)
    assert isinstance(neck, Neck)
    return neck
