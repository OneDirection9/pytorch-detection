from __future__ import absolute_import, division, print_function

from typing import Any, Dict, Optional

from foundation.registry import Registry

from det.layers import BaseModule, ShapeSpec

__all__ = ['Backbone', 'BackboneRegistry', 'build_backbone']

# Alias of BaseModule
Backbone = BaseModule


class BackboneRegistry(Registry):
    """Registry of backbones."""
    pass


def build_backbone(cfg: Dict[str, Any], input_shape: Optional[ShapeSpec] = None) -> Backbone:
    """Builds a backbone from config."""
    if input_shape is None:
        # Default expected 3 channels input
        input_shape = ShapeSpec(channels=3)

    backbone_name = cfg.pop('name')
    backbone = BackboneRegistry.get(backbone_name)(input_shape, **cfg)
    assert isinstance(backbone, Backbone)
    return backbone
