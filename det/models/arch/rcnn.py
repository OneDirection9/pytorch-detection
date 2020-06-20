# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

from typing import List, Optional

import torch
from torch import nn

from det.structures import ImageList
from ..backbones import Backbone
from ..necks import Neck

__all__ = ['GeneralizedRCNN']


class GeneralizedRCNN(nn.Module):
    """`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`_.

    .. _`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`:
        https://arxiv.org/abs/1506.01497
    """

    def __init__(
        self,
        backbone: Backbone,
        neck: Optional[Neck] = None,
        proposal_generator=None,
        roi_head=None,
        pixel_mean: List[float] = (103.530, 116.280, 123.675),
        pixel_std: List[float] = (57.375, 57.120, 58.395),
    ) -> None:
        """
        Args:
            backbone (Module): Convolution neural network used to extra image features.
            neck:
            proposal_generator:
            roi_head:
        """
        super(GeneralizedRCNN, self).__init__()

        if len(pixel_mean) != len(pixel_std):
            raise ValueError('Length of pixel_mean and pixel_std should be the same')

        self.backbone = backbone
        self.neck = neck
        self.proposal_generator = proposal_generator
        self.roi_head = roi_head

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
