from __future__ import absolute_import, division, print_function

from typing import Dict

from foundation.registry import Registry

from det.config import CfgNode
from det.layers import BaseModule, ShapeSpec

__all__ = ['Neck', 'NeckRegistry', 'build_neck']

# Alias of BaseModule
Neck = BaseModule


class NeckRegistry(Registry):
    """Registry of necks, which enhance the features extracted by backbones.

    The registered object must be a callable that accepts two arguments:

    1. cfg: A :class:`CfgNode`
    2. input_shape: The output shape of backbone mapping from name to shape specification

    It will be called with `obj.from_config(cfg, input_shape)` or `obj(cfg, input_shape)`.
    """
    pass


def build_neck(cfg: CfgNode, input_shape: Dict[str, ShapeSpec]) -> Neck:
    """Builds a neck from `cfg.MODEL.NECK.NAME`."""
    neck_name = cfg.MODEL.NECK.NAME
    neck_cls = NeckRegistry.get(neck_name)
    if hasattr(neck_cls, 'from_config'):
        neck = neck_cls.from_config(cfg, input_shape)
    else:
        neck = neck_cls(cfg, input_shape)
    assert isinstance(neck, Neck)
    return neck
