from __future__ import absolute_import, division, print_function

import inspect
import math
from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from foundation.nn import weight_init
from torch import nn
from torch.nn import functional as F

from det import layers
from .registry import NeckRegistry

__all__ = [
    'LastLevelMaxPool',
    'LastLevelP6P7',
    'FPN',
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


@NeckRegistry.register('FPN')
class FPN(layers.BaseModule):
    """`Feature Pyramid Networks for Object Detection`_.

    .._`Feature Pyramid Networks for Object Detection`:
        https://arxiv.org/abs/1612.03144
    """

    def __init__(
        self,
        input_shape: Dict[str, layers.ShapeSpec],
        in_features: List[str] = ('res2', 'res3', 'res4', 'res5'),
        out_channels: int = 256,
        norm: Union[str, Callable] = '',
        fuse_type: str = 'sum',
        top_block: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            input_shape: Mapping from feature name (e.g. "res2") to feature map shape.
            in_features: Name of the input feature maps coming from the backbone to which FPN is
                attached. For example, if the backbone produces ["res2", "res3", "res4"], any
                *contiguous* sublist of these may be used; order must be from high to low
                resolution.
            out_channels: Number of channels in the output feature maps.
            norm: Normalization for lateral and output conv layers. See :func:`get_norm` for
                supported format.
            fuse_type (str): Types for fusing the top down features and the lateral ones. It can be
                "sum" (default), which sums up element-wise; or "avg", which takes the element-wise
                mean of the two.
            top_block: If provided, an extra operation will be performed on the last output of the
                bottom-up features or FPN output (smallest resolution), and the result will extend
                the result list. The top_block further downsamples the feature map. It must
                have an attribute "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_features", which is a string representing its input feature
                (e.g., p5 or res5). Usually use a wrapper that create top_block and pass it to FPN.
        """
        super(FPN, self).__init__()

        if fuse_type not in ['avg', 'sum']:
            raise ValueError('fuse_type should be either avg or sum. Got {}'.format(fuse_type))

        in_channels = [input_shape[f].channels for f in in_features]
        in_strides = [input_shape[f].stride for f in in_features]
        _assert_strides_are_log2_contiguous(in_strides)

        self._size_divisibility = in_strides[-1]
        self._in_features = in_features
        self._fuse_type = fuse_type

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        self.top_block = top_block
        self._out_features = []
        self._output_shape = {}
        self._lateral_names = []
        self._output_names = []

        use_bias = norm == ''
        for stride, channels in zip(in_strides, in_channels):
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

            stage = int(math.log2(stride))

            lateral_name = 'lateral{}'.format(stage)
            self.add_module(lateral_name, lateral_conv)
            self._lateral_names.append(lateral_name)

            output_name = 'p{}'.format(stage)
            self.add_module(output_name, output_conv)
            self._output_names.append(output_name)
            self._output_shape[output_name] = layers.ShapeSpec(channels=out_channels, stride=stride)

        if self.top_block is not None:
            stage = int(math.log2(in_strides[-1])) + 1
            for s in range(stage, stage + self.top_block.num_levels):
                name = 'p{}'.format(s)
                self._out_features.append(name)
                self._output_shape[name] = layers.ShapeSpec(channels=out_channels, stride=2 ** s)

    def forward(self, bottom_up_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = [bottom_up_features[f] for f in self._in_features]

        laterals = [lateral_conv(x_i) for x_i, lateral_conv in zip(x, self.lateral_convs)]

        # Build laterals
        laterals = [
            lateral_conv(bottom_up_features[feature])
            for feature, lateral_conv in zip(self._in_features, self.lateral_convs)
        ]

        # Build top-down path
        for i in range(len(self.lateral_convs) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            if self._fuse_type == 'avg':
                laterals[i - 1] /= 2

        # Build output
        # part1: from original levels
        outputs = [
            output_conv(lateral) for lateral, output_conv in zip(laterals, self.output_convs)
        ]
        # part2: from top_block
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = outputs[self._out_features.index(self.top_block.in_feature)]
            outputs.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(outputs)
        return dict(zip(self._out_features, outputs))

    @property
    def output_shape(self) -> Dict[str, layers.ShapeSpec]:
        return self._output_shape

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility


def _assert_strides_are_log2_contiguous(strides: List[int]) -> None:
    """Checks that each stride is 2x times its preceding stride, i.e. "contiguous in log2"."""
    for i, stride in enumerate(strides[1:], 1):
        if stride != 2 * strides[i - 1]:
            raise ValueError('Strides {} {} are not log2 contiguous'.format(stride, strides[i - 1]))


class TopBlock(layers.BaseModule, metaclass=ABCMeta):
    """Base class for the extra block in the FPN."""

    def __init__(
        self,
        in_stride: int,
        in_feature: str,
    ) -> None:
        """
        Args:
            in_stride: The stride of last level of FPN stem layer.
            in_feature: The feature name for input feature maps.
        """
        super(TopBlock, self).__init__()

        self._stage = int(math.log2(in_stride)) + 1
        self._in_feature = in_feature


# Presetting top_block
NeckRegistry.register_partial('RCNN_FPN_Neck', top_block=LastLevelMaxPool())(FPN)


@NeckRegistry.register('RetinaNet_FPN_Neck')
def build_retinanet_fpn_neck(input_shape: Dict[str, layers.ShapeSpec], **kwargs: Any) -> FPN:
    """Returns an instance of :class:`FPN` neck with top_block is LastLevelP6P7."""
    if 'top_block' in kwargs:
        raise ValueError('top_block will be set to LastLevelP6P7 automatically')

    sig = inspect.signature(FPN.__init__)
    out_channels = kwargs.get('out_channels', sig.parameters['out_channels'].default)

    top_block = LastLevelP6P7(input_shape['res5'].channels, out_channels, 'res5')
    return FPN(input_shape, **kwargs, top_block=top_block)
