from __future__ import absolute_import, division, print_function

import math
from typing import Callable, Dict, List, Optional, Union

import torch
from foundation.nn import weight_init
from torch import nn
from torch.nn import functional as F

from det import layers
from ..shape_spec import ShapeSpec
from .base import Neck, NeckRegistry

__all__ = [
    'LastLevelMaxPool',
    'LastLevelP6P7',
    'FPN',
    'build_rcnn_fpn_neck',
    'build_retinanet_fpn_neck',
]


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


class FPN(Neck):
    """`Feature Pyramid Networks for Object Detection`_.

    .._`Feature Pyramid Networks for Object Detection`:
        https://arxiv.org/abs/1612.03144
    """

    def __init__(
        self,
        in_channels: List[int],
        in_strides: List[int],
        in_features: List[str],
        out_channels: int = 256,
        norm: Union[str, Callable] = '',
        top_block: Optional[nn.Module] = None,
        fuse_type: str = 'sum',
    ) -> None:
        """
        Args:
            in_channels: List of input channels per scale.
            in_strides: List of input strides per scale. Each stride should be 2x times its
                preceding stride.
            in_features: Name of the input feature maps coming from the backbone to which FPN is
                attached. For example, if the backbone produces ["res2", "res3", "res4"], any
                *contiguous* sublist of these may be used; order must be from high to low
                resolution.
            out_channels: Number of channels in the output feature maps.
            norm: Normalization for lateral and output conv layers. See :func:`get_norm` for
                supported format.
            top_block: If provided, an extra operation will be performed on the last output of the
                bottom-up features or FPN output (smallest resolution), and the result will extend
                the result list. The top_block further downsamples the feature map. It must
                have an attribute "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_features", which is a string representing its input feature
                (e.g., p5 or res5).
            fuse_type (str): Types for fusing the top down features and the lateral ones. It can be
                "sum" (default), which sums up element-wise; or "avg", which takes the element-wise
                mean of the two.
        """
        super(FPN, self).__init__()

        if not (len(in_strides) == len(in_channels) == len(in_features)):
            raise ValueError('in_strides, in_channels, and in_features must have the same length')
        if fuse_type not in ['avg', 'sum']:
            raise ValueError('fuse_type should be either avg or sum. Got {}'.format(fuse_type))
        # Check that each stride is 2x times its preceding stride, i.e. "contiguous in log2"
        for i, stride in enumerate(in_strides[1:], 1):
            if stride != 2 * in_strides[i - 1]:
                raise ValueError(
                    'Strides {} {} are not log2 contiguous'.format(stride, in_strides[i - 1])
                )

        self._size_divisibility = in_strides[-1]
        self._in_features = in_features
        self._fuse_type = fuse_type

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        self.top_block = top_block

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

        self._output_shape = {}
        self._out_features = []
        for stride in in_strides:
            name = 'p{}'.format(int(math.log2(stride)))
            self._output_shape[name] = ShapeSpec(channels=out_channels, stride=stride)
            self._out_features.append(name)

        if self.top_block is not None:
            stage = int(math.log2(in_strides[-1])) + 1
            for s in range(stage, stage + self.top_block.num_levels):
                name = 'p{}'.format(s)
                self._output_shape[name] = ShapeSpec(channels=out_channels, stride=2 ** s)
                self._out_features.append(name)

    def forward(self, bottom_up_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = [bottom_up_features[f] for f in self._in_features]

        # Build laterals
        x = [lateral_conv(x[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        for i in range(len(self.lateral_convs) - 1, 0, -1):
            x[i - 1] += F.interpolate(x[i], scale_factor=2, mode='nearest')
            if self._fuse_type == 'avg':
                x[i - 1] /= 2

        # Build output
        # part1: from original levels
        outputs = [output_conv(x[i]) for i, output_conv in enumerate(self.output_convs)]
        # part2: from top_block
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = outputs[self._out_features.index(self.top_block.in_feature)]
            outputs.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(outputs)
        return dict(zip(self._out_features, outputs))

    @property
    def output_shape(self) -> Dict[str, ShapeSpec]:
        return self._output_shape

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility


@NeckRegistry.register('RCNN_FPN_Neck')
def build_rcnn_fpn_neck(
    input_shape: Dict[str, ShapeSpec],
    in_features: List[str] = ('res2', 'res3', 'res4', 'res5'),
    out_channels: int = 256,
    norm: Union[str, Callable] = '',
    fuse_type: str = 'sum',
) -> FPN:
    """
    Args:
        input_shape: Get from backbone module. Don't need specify explicitly.
        in_features:
        out_channels:
        norm:
        fuse_type:

    Returns:
        nn.Module: FPN neck with top_block is LastLevelMaxPool.
    """
    in_channels = [input_shape[f].channels for f in in_features]
    in_strides = [input_shape[f].stride for f in in_features]
    top_block = LastLevelMaxPool()

    return FPN(in_channels, in_strides, in_features, out_channels, norm, top_block, fuse_type)


@NeckRegistry.register('RetinaNet_FPN_Neck')
def build_retinanet_fpn_neck(
    input_shape: Dict[str, ShapeSpec],
    in_features: List[str] = ('res2', 'res3', 'res4', 'res5'),
    out_channels: int = 256,
    norm: Union[str, Callable] = '',
    fuse_type: str = 'sum',
) -> FPN:
    """
    Args:
        input_shape: Get from backbone module. Don't need specify explicitly.
        in_features:
        out_channels:
        norm:
        fuse_type:

    Returns:
        nn.Module: FPN neck with top_block is LastLevelP6P7.
    """
    in_channels = [input_shape[f].channels for f in in_features]
    in_strides = [input_shape[f].stride for f in in_features]
    top_block = LastLevelP6P7(input_shape['res5'], out_channels, 'res5')

    return FPN(in_channels, in_strides, in_features, out_channels, norm, top_block, fuse_type)
