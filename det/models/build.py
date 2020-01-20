from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from foundation.utils import build
from .registry import *

__all__ = ['build_model']

# arguments name to registry
name_to_registry = {
    'backbone': BackboneStash,
    'neck': NeckStash,
    'proposal_head': ProposalHeadStash,
    'anchor_generator': AnchorGeneratorStash,
    'roi_head': ROIHeadStash,
}


def build_model(model_cfg):
    def update(cfg):
        """Builds sub-modules from configs."""
        if isinstance(cfg, dict):
            updated_cfg = {}
            for k, v in cfg.items():
                if k in name_to_registry and isinstance(v, dict) and 'name' in v:
                    v = build(name_to_registry[k], update(v))
                updated_cfg[k] = v
            return updated_cfg
        else:
            return cfg

    return build(ArchStash, update(model_cfg))
