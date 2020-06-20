from __future__ import absolute_import, division, print_function

from typing import Any, Dict

from foundation.registry import Registry
from torch import nn

from det.layers import ShapeSpec

__all__ = ['ProposalGeneratorRegistry', 'build_proposal_generator']


class ProposalGeneratorRegistry(Registry):
    """Registry of proposal generators."""
    pass


def build_proposal_generator(cfg: Dict[str, Any], input_shape: Dict[str, ShapeSpec]) -> nn.Module:
    """Builds a proposal generator from config.

    Args:
        cfg:
        input_shape: Output shape of backbone or neck.
    """
    proposal_generator_name = cfg.pop('name')
    proposal_generator = ProposalGeneratorRegistry.get(proposal_generator_name)(input_shape, **cfg)
    return proposal_generator
