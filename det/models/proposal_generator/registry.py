from __future__ import absolute_import, division, print_function

from typing import Dict

from foundation.registry import Registry
from torch import nn

from det.config import CfgNode
from det.layers import ShapeSpec

__all__ = ['ProposalGeneratorRegistry', 'build_proposal_generator']


class ProposalGeneratorRegistry(Registry):
    """Registry of proposal generators which produces object proposals from feature maps.

    The registered object must be a callable that accepts two arguments:

    1. cfg: A :class:`CfgNode`
    2. input_shape: The output shape of backbone or neck mapping from name to shape specification

    It will be called with `obj.from_config(cfg, input_shape)` or `obj(cfg, input_shape)`.
    """
    pass


def build_proposal_generator(cfg: CfgNode, input_shape: Dict[str, ShapeSpec]) -> nn.Module:
    """Builds a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`."""
    proposal_generator_name = cfg.MODEL.PROPOSAL_GENERATOR.NAME
    proposal_generator_cls = ProposalGeneratorRegistry.get(proposal_generator_name)
    if hasattr(proposal_generator_cls, 'from_config'):
        proposal_generator = proposal_generator_cls.from_config(cfg, input_shape)
    else:
        proposal_generator = proposal_generator_cls(cfg, input_shape)
    return proposal_generator
