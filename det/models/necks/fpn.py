# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import inspect
import math
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from foundation.nn import weight_init
from torch import nn
from torch.nn import functional as F

from det import layers
from .registry import NeckRegistry

__all__ = [
    'TopBlock',
    'LastLevelMaxPool',
    'LastLevelP6P7',
    'FPN',
    'rcnn_fpn_neck',
    'retinanet_fpn_neck',
]


class TopBlock(layers.BaseModule, metaclass=ABCMeta):
    """Base class for the extra block in the FPN."""

    @abstractmethod
    def forward(
        self,
        bottom_up_features: Dict[str, torch.Tensor],
        fpn_body_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bottom_up_features: The original bottom up feature maps.
            fpn_body_features: The FPN feature maps.

        Returns:
            Extra new feature maps.
        """
        pass


# TODO: Consider using in_stride as argument of constructor, so that can infer in_feature, and can
#  apply the TopBlock to FPN on the fly, don't need to fix the feature names.
class LastLevelMaxPool(TopBlock):
    """This module is used in the original FPN to generate a downsampled P6 from P5."""

    def __init__(self, in_channels: int) -> None:
        """
        Args:
            in_channels: Output channels of P5.
        """
        super(LastLevelMaxPool, self).__init__()

        self._in_feature = 'p5'

        self.p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        self._output_shape = {'p6': layers.ShapeSpec(channels=in_channels, stride=2)}
        assert set(self._output_shape.keys()).issubset([name for name, _ in self.named_children()])

    def forward(
        self,
        bottom_up_features: Dict[str, torch.Tensor],
        fpn_body_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {'p6': self.p6(fpn_body_features[self._in_feature])}

    @property
    def output_shape(self) -> Dict[str, layers.ShapeSpec]:
        return self._output_shape


class LastLevelP6P7(TopBlock):
    """This module is used in RetinaNet to generate extra layers, P6 and P7 from C5 features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels: Output channels of C5 features.
            out_channels: Output channels of P6 and P7 features.
        """
        super(LastLevelP6P7, self).__init__()

        self._in_feature = 'res5'

        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.caffe2_xavier_init(module)
        self._output_shape = {
            'p6': layers.ShapeSpec(channels=out_channels, stride=2),
            'p7': layers.ShapeSpec(channels=out_channels, stride=4),
        }
        assert set(self._output_shape.keys()).issubset([name for name, _ in self.named_children()])

    def forward(
        self,
        bottom_up_features: Dict[str, torch.Tensor],
        fpn_body_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        outputs = {}

        x = self.p6(bottom_up_features[self._in_feature])
        outputs['p6'] = x

        x = self.p7(x)
        outputs['p7'] = x

        return outputs

    @property
    def output_shape(self) -> Dict[str, layers.ShapeSpec]:
        return self._output_shape


@NeckRegistry.register('FPN')
class FPN(layers.BaseModule):
    """`Feature Pyramid Networks for Object Detection`_.

    .._`Feature Pyramid Networks for Object Detection`:
        https://arxiv.org/abs/1612.03144
    """

    def __init__(
        self,
        *,
        in_channels: List[int],
        in_strides: List[int],
        in_features: List[str],
        out_channels: int = 256,
        norm: Union[str, Callable] = '',
        fuse_type: str = 'sum',
        top_block: Optional[TopBlock] = None,
    ) -> None:
        """
        Args:
            in_channels: List of input channels of each feature map.
            in_strides: List of strides of each feature map.
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
                the result list. The top_block further downsamples the feature map. It is expected
                to take the bottom_up_features and fpn_body_features as input, and returns new
                feature maps. See :class:`TopBlock`. Usually use a wrapper to create top_block and
                pass it to FPN constructor.
        """
        super(FPN, self).__init__()

        if not (len(in_channels) == len(in_strides) == len(in_features)):
            raise ValueError('in_channels, in_strides, and in_features should have the same length')
        if fuse_type not in ['avg', 'sum']:
            raise ValueError('fuse_type should be either avg or sum. Got {}'.format(fuse_type))

        _assert_strides_are_log2_contiguous(in_strides)

        self._size_divisibility = in_strides[-1]
        self._in_features = in_features
        self._fuse_type = fuse_type

        self.top_block = top_block
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

        assert set(self._output_shape.keys()).issubset([name for name, _ in self.named_children()])

        if self.top_block is not None:
            current_stride = in_strides[-1]
            for k, v in self.top_block.output_shape.items():
                current_stride = current_stride * v.stride
                assert k not in self._output_shape, '{} already in the FPN output_shape'.format(k)
                self._output_shape[k] = layers.ShapeSpec(channels=v.channels, stride=current_stride)

    def forward(self, bottom_up_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Build laterals
        laterals = [
            getattr(self, name)(bottom_up_features[f])
            for name, f in zip(self._lateral_names, self._in_features)
        ]

        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            if self._fuse_type == 'avg':
                laterals[i - 1] /= 2

        # Build output
        # part1: from original levels
        outputs = {
            name: getattr(self, name)(lateral)
            for name, lateral in zip(self._output_names, laterals)
        }
        # part2: from top_block
        if self.top_block is not None:
            top_block_outputs = self.top_block(bottom_up_features, outputs)
            assert set(outputs.keys()).isdisjoint(set(top_block_outputs.keys()))
            outputs.update(top_block_outputs)

        return outputs

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


"""
Wrappers of FPN presetting top_block
"""


@NeckRegistry.register('RCNN_FPN_Neck')
def rcnn_fpn_neck(input_shape: Dict[str, layers.ShapeSpec], **kwargs: Any) -> FPN:
    """Returns an instance of :class:`FPN` neck with top_block is LastLevelMaxPool."""
    if 'top_block' in kwargs:
        raise ValueError('top_block will be set to LastLevelMaxPool automatically')
    if 'in_channels' in kwargs or 'in_strides' in kwargs:
        raise ValueError('in_channels and in_strides are inferred from backbone output shape')

    in_features = kwargs['in_features']
    in_channels = [input_shape[f].channels for f in in_features]
    in_strides = [input_shape[f].stride for f in in_features]

    sig = inspect.signature(FPN.__init__)
    out_channels = kwargs.get('out_channels', sig.parameters['out_channels'].default)
    top_block = LastLevelMaxPool(out_channels)

    kwargs.update({'in_channels': in_channels, 'in_strides': in_strides, 'top_block': top_block})

    return FPN(**kwargs)


@NeckRegistry.register('RetinaNet_FPN_Neck')
def retinanet_fpn_neck(input_shape: Dict[str, layers.ShapeSpec], **kwargs: Any) -> FPN:
    """Returns an instance of :class:`FPN` neck with top_block is LastLevelP6P7."""
    if 'top_block' in kwargs:
        raise ValueError('top_block will be set to LastLevelP6P7 automatically')
    if 'in_channels' in kwargs or 'in_strides' in kwargs:
        raise ValueError('in_channels and in_strides are inferred from backbone output shape')

    in_features = kwargs['in_features']
    in_channels = [input_shape[f].channels for f in in_features]
    in_strides = [input_shape[f].stride for f in in_features]

    sig = inspect.signature(FPN.__init__)
    out_channels = kwargs.get('out_channels', sig.parameters['out_channels'].default)
    top_block = LastLevelP6P7(input_shape['res5'].channels, out_channels)

    kwargs.update({'in_channels': in_channels, 'in_strides': in_strides, 'top_block': top_block})

    return FPN(**kwargs)
