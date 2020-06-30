# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from typing import List

import numpy as np
import torch
from foundation.nn import weight_init
from foundation.registry import Registry
from torch import nn
from torch.nn import functional as F

from det.config import CfgNode
from det.layers import Conv2d, ShapeSpec, get_norm

__all__ = ['RoIBoxHeadRegistry', 'build_box_head', 'FastRCNNConvFCHead']


class RoIBoxHeadRegistry(Registry):
    """Registry for box heads, which make box predictions from per-region features.

    The registered object must be a callable that accepts two arguments:

    1. cfg: A :class:`CfgNode`
    2. input_shape: A :class:`ShapeSpec`, which contains the input shape specification

    It will be called with `obj.from_config(cfg, input_shape)` or `obj(cfg, input_shape)`.
    """
    pass


def build_box_head(cfg: CfgNode, input_shape: ShapeSpec) -> nn.Module:
    """Builds a box head from `cfg.MODEL.ROI_BOX_HEAD.NAME`."""
    box_head_name = cfg.MODEL.ROI_BOX_HEAD.NAME
    box_head_cls = RoIBoxHeadRegistry.get(box_head_name)
    if hasattr(box_head_cls, 'from_config'):
        box_head = box_head_cls.from_config(cfg, input_shape)
    else:
        box_head = box_head_cls(cfg, input_shape)
    return box_head


@RoIBoxHeadRegistry.register('FastRCNNConvFCHead')
class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        conv_dims: List[int],
        fc_dims: List[int],
        conv_norm=''
    ) -> None:
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module('conv{}'.format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module('fc{}'.format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.caffe2_msra_init(layer)
        for layer in self.fcs:
            weight_init.caffe2_msra_init(layer)

    @classmethod
    def from_config(cls, cfg: CfgNode, input_shape: ShapeSpec) -> 'FastRCNNConvFCHead':
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return cls(
            input_shape=input_shape,
            conv_dims=[conv_dim] * num_conv,
            fc_dims=[fc_dim] * num_fc,
            conv_norm=cfg.MODEL.ROI_BOX_HEAD.NORM,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_shape(self) -> ShapeSpec:
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])
