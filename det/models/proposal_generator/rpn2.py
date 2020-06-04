from __future__ import absolute_import, division, print_function

from torch import nn

from .base import ProposalGeneratorRegistry


@ProposalGeneratorRegistry.register('RPN')
class RPN(nn.Module):
    pass
