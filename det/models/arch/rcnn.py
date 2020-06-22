# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from det.structures import ImageList
from ..backbones import Backbone, build_backbone
from ..necks import Neck, build_neck
from ..proposal_generator import build_proposal_generator
from .registry import ArchRegistry

__all__ = ['GeneralizedRCNN', 'generalized_rcnn_model']


class GeneralizedRCNN(nn.Module):
    """`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`_.

    .. _`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`:
        https://arxiv.org/abs/1506.01497
    """

    def __init__(
        self,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        neck: Optional[Neck] = None,
        pixel_mean: List[float] = (103.530, 116.280, 123.675),
        pixel_std: List[float] = (57.375, 57.120, 58.395),
    ) -> None:
        """
        Args:
            backbone (Module): Convolution neural network used to extra image features.
            neck:
            proposal_generator:
            roi_heads:
        """
        super(GeneralizedRCNN, self).__init__()

        if len(pixel_mean) != len(pixel_std):
            raise ValueError('Length of pixel_mean and pixel_std should be the same')

        self.backbone = backbone
        self.neck = neck
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        if self.neck is not None:
            self._size_divisibility = self.neck.size_divisibility
        else:
            self._size_divisibility = self.backbone.size_divisibility

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def extract_features(self, x):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, batched_inputs):
        images = [item['image'].to(self.device) for item in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self._size_divisibility)
        features = self.extract_features(images.tensor)

        if 'instances' in batched_inputs[0]:
            gt_instances = [x['instances'].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        self.proposal_generator(images, features, gt_instances)

        return features


@ArchRegistry.register('Generalized_RCNN_Model')
def generalized_rcnn_model(
    *,
    backbone: Dict[str, Any],
    proposal_generator: Dict[str, Any],
    roi_heads: Dict[str, Any],
    neck: Optional[Dict[str, Any]] = None,
    pixel_mean: List[float] = (103.530, 116.280, 123.675),
    pixel_std: List[float] = (57.375, 57.120, 58.395),
) -> GeneralizedRCNN:
    """
    Args:
        backbone: Config of backbone.
        proposal_generator: Config of proposal_generator.
        roi_heads: Config of roi_heads.
        neck: Config of neck.
        pixel_mean: See :class:`GeneralizedRCNN`.
        pixel_std: See :class:`GeneralizedRCNN`.
    """
    backbone = build_backbone(backbone)

    if neck is not None:
        neck = build_neck(neck, backbone.output_shape)

    output_shape = neck.output_shape if neck is not None else backbone.output_shape
    proposal_generator = build_proposal_generator(proposal_generator, output_shape)

    return GeneralizedRCNN(
        backbone=backbone,
        proposal_generator=proposal_generator,
        roi_heads=roi_heads,
        neck=neck,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
