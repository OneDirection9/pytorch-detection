from __future__ import absolute_import, division, print_function

from typing import Dict

from foundation.registry import Registry

from det.config import CfgNode
from det.layers import BaseModule, ShapeSpec

__all__ = ['Neck', 'NeckRegistry', 'build_neck']

# Alias of BaseModule
Neck = BaseModule


class NeckRegistry(Registry):
    """Registry of necks."""
    pass


def build_neck(cfg: CfgNode, input_shape: Dict[str, ShapeSpec]) -> Neck:
    """Builds a neck from `cfg.NECK.NAME`."""
    neck_name = cfg.NECK.NAME
    neck_cls = NeckRegistry.get(neck_name)
    if hasattr(neck_cls, 'from_config'):
        neck = neck_cls.from_config(cfg, input_shape)
    else:
        neck = neck_cls(cfg, input_shape)
    assert isinstance(neck, Neck)
    return neck
