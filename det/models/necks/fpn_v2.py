from __future__ import absolute_import, division, print_function

import math
from typing import Callable, Dict, List, Union

import torch
from foundation.nn import weight_init
from torch import nn
from torch.nn import functional as F

from det import layers
from det.layers import ShapeSpec


class LastLevelMaxPool(nn.Module):
    """This module is used in the original FPN to generate a downsampled P6 feature from P5."""

    def __init__(self) -> None:
        super(LastLevelMaxPool, self).__init__()

        self.num_levels = 1
        self.in_feature = 'p5'

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """This module is used in RetinaNet to generate extra layers, P6 and P7 from C5 features."""

    def __init__(self, in_channels: int, out_channels: int, in_feature: str = 'res5') -> None:
        super(LastLevelP6P7, self).__init__()

        self.num_levels = 2
        self.in_feature = in_feature

        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for layer in [self.p6, self.p7]:
            weight_init.caffe2_xavier_init(layer)

    def forward(self, c5: torch.Tensor) -> List[torch.Tensor]:
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class FPN(layers.Module):
    """`Feature Pyramid Networks for Object Detection`_.

    .._`Feature Pyramid Networks for Object Detection`:
        https://arxiv.org/abs/1612.03144
    """

    def __init__(
        self,
        backbone: layers.Module,
        in_features: List[str],
        out_channels: int = 256,
        norm: Union[str, Callable] = '',
        top_block: str = '',
        fuse_type: str = 'sum',
    ) -> None:
        super(FPN, self).__init__()

        if fuse_type not in ['avg', 'sum']:
            raise ValueError('fuse_type should be either avg or sum. Got {}'.format(fuse_type))

        self.in_features = in_features

        input_shapes = backbone.output_shape
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        self._size_divisibility = in_strides[-1]

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        use_bias = norm == ''
        for idx, channels in enumerate(in_channels):
            lateral_norm = layers.get_norm(norm, out_channels)
            output_norm = layers.get_norm(norm, out_channels)

            lateral_conv = layers.Conv2d(
                channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = layers.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm
            )

            weight_init.caffe2_xavier_init(lateral_conv)
            weight_init.caffe2_xavier_init(output_conv)

            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        if top_block == 'rcnn':
            self.top_block = LastLevelMaxPool()
        elif top_block == 'retinanet':
            self.top_block = LastLevelP6P7(in_channels[-1], out_channels)
        elif top_block == '':
            self.top_block = None
        else:
            raise ValueError("top_block can be one of '', 'rcnn', or 'retinanet'")

        if self.top_block is not None:
            pass

        self._output_shape = {}
        for stride in in_strides:
            name = 'p{}'.format(int(math.log2(stride)))
            self._output_shape[name] = layers.ShapeSpec(channels=out_channels, stride=stride)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = [x[f] for f in self.in_features]
        return x

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    @property
    def output_shape(self) -> Dict[str, ShapeSpec]:
        return self._output_shape


def _assert_strides_are_log2_contiguous(strides):
    """Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2"."""
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], 'Strides {} {} are not log2 contiguous'.format(
            stride, strides[i - 1]
        )
