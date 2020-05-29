from __future__ import absolute_import, division, print_function

from foundation.registry import Registry

__all__ = [
    'ArchStash',
    'BackboneStash',
    'NeckStash',
    'ProposalHeadStash',
    'AnchorGeneratorStash',
    'ROIHeadStash',
]


class ArchStash(Registry):
    """Registry for top architectures."""
    pass


class BackboneStash(Registry):
    """Registry for backbones."""
    pass


class NeckStash(Registry):
    """Registry for necks."""
    pass


class ProposalHeadStash(Registry):
    """Registry for proposal heads."""
    pass


class AnchorGeneratorStash(Registry):
    """Registry for anchor generators."""
    pass


class ROIHeadStash(Registry):
    """Registry for ROI heads."""
    pass
