# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from foundation.backends.torch.utils import batch_tensors
from ..registry import ArchStash

__all__ = ['FasterRCNN']


@ArchStash.register('FasterRCNN')
class FasterRCNN(nn.Module):
    """`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`_.

    Args:
        backbone (Module): Convolution neural network used to extra image features.

    .. _`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`:
        https://arxiv.org/abs/1506.01497
    """

    def __init__(self, backbone, neck=None, proposal_head=None, roi_head=None,
                 device='cuda'):
        super(FasterRCNN, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.proposal_head = proposal_head
        self.roi_head = roi_head

        self._device = torch.device(device)

        self.to(device)

    def forward(self, items):
        images = self.preprocess_image(items)
        if 'bboxes' in items[0]:
            gt_bboxes = [item['bboxes'].to(self._device) for item in items]
        else:
            gt_bboxes = None

        # Extract image features
        features = self.backbone(images)
        if self.neck is not None:
            features = self.neck(features)

        # Generate proposals
        if self.proposal_head is not None:
            image_sizes = [item['ori_shape'].to(self._device) for item in items]
            proposals, proposal_losses = self.proposal_head(
                image_sizes, features, gt_bboxes
            )
        else:
            assert 'proposals' in items[0]
            proposals = [item['proposals'].to(self._device) for item in items]
            proposal_losses = {}

        image_ids = [item['id'] for item in items]
        gt_labels = [item['labels'].to(self._device) for item in items]
        _, detector_losses = self.roi_head(
            images, features, proposals, gt_bboxes, gt_labels
        )

        return features

    def preprocess_image(self, items):
        images = [item['image'].to(self._device) for item in items]
        images = batch_tensors(images)
        return images
