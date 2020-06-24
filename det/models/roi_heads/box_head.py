from __future__ import absolute_import, division, print_function

from typing import List

from foundation.registry import Registry
from torch import nn

from det.layers import ShapeSpec


class ROIBoxHeadRegistry(Registry):
    """Registry of box heads."""
    pass


class FastRCNNConvFCHead(nn.Module):
    """A head with several 3x3 conv layers (each followed by norm & relu) and then several fc
    layers (each followed by relu).
    """

    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        conv_dims: List[int],
        fc_dims: List[int],
        conv_norm: str = '',
    ) -> None:
        super(FastRCNNConvFCHead, self).__init__()
