from __future__ import absolute_import, division, print_function

from typing import Optional

from foundation.registry import Registry

from det.config import CfgNode
from det.layers import BaseModule, ShapeSpec

__all__ = ['Backbone', 'BackboneRegistry', 'build_backbone']

# Alias of BaseModule
Backbone = BaseModule


class BackboneRegistry(Registry):
    """Registry of backbones, which extract feature maps from images.

    The registered object must be a callable that accepts two arguments:

    1. cfg: A :class:`CfgNode`
    2. input_shape: A :class:`ShapeSpec`, which contains the input shape specification.

    It will be called with `obj.from_config(cfg, input_shape)` or `obj(cfg, input_shape)`.
    """
    pass


def build_backbone(cfg: CfgNode, input_shape: Optional[ShapeSpec] = None) -> Backbone:
    """Builds a backbone from `cfg.MODEL.BACKBONE.NAME`."""
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone_cls = BackboneRegistry.get(backbone_name)
    if hasattr(backbone_cls, 'from_config'):
        backbone = backbone_cls.from_config(cfg, input_shape)
    else:
        backbone = backbone_cls(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone
