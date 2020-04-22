# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import torch
from torch import nn

from .. import utils as mo_utils
from ..registry import ROIHeadStash


@ROIHeadStash.register('ROIHead')
class ROIHead(nn.Module):
    """ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(
        self,
        proposal_append_gt=True,
        iou_threshold=0.5,
        batch_size_per_image=512,
        positive_sample_fraction=0.25,
        num_classes=80
    ):
        super(ROIHead, self).__init__()

        self._proposal_append_gt = proposal_append_gt
        self._mather = mo_utils.Matcher(iou_threshold, iou_threshold, False)
        self._sampler = mo_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_sample_fraction
        )
        self._num_classes = num_classes

    def sample_proposals(self, matched_idxs, matched_labels, gt_labels):
        """Based on the matching between K proposals and Ti ground truth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): A vector of length K, each is the best-matched
                gt index in [0, Ti) for each proposal.
            matched_labels (Tensor): A vector of length K, the matcher's label
                (one of POSITIVE and NEGATIVE) for each proposal.
            gt_labels (Tensor): A vector of length Ti.

        Returns:
            Tensor: A vector of indices of sampled proposals. Each is in [0, K).
            Tensor: A vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_labels.numel() > 0
        if has_gt:
            gt_labels = gt_labels[matched_idxs]
            gt_labels[matched_labels == mo_utils.NEGATIVE] = self._num_classes
        else:
            gt_labels = torch.zeros_like(matched_idxs) + self._num_classes

        sampled_fg_idxs, sampled_ng_idxs = self._sampler(matched_labels)
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_ng_idxs], dim=0)
        return sampled_idxs, gt_labels[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, gt_bboxes, gt_labels):
        """Prepares some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `gt_bboxes`, and assigns training
        labels to the proposals.
        It returns ``self._batch_size_per_image`` random samples from proposals and ground
        truth boxes, with a fraction of positives that is no larger than
        ``self._positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts the proposals
        # will be low quality due to random initialization. It's possible that none of
        # these initial proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head, cls head, mask
        # head). Adding the gt boxes to the set of proposals ensures that the second stage
        # components will have some positive examples from the start of training. For RPN,
        # this augmentation improves convergence and empirically improves box AP on COCO
        # by about 0.5 points (under one tested configuration).
        if self._proposal_append_gt:
            proposals = mo_utils.add_ground_truth_to_proposals(gt_bboxes, proposals)

        proposals_with_gt, num_fg_samples, num_bg_samples = [], [], []
        for proposals_i, gt_bboxes_i, gt_labels_i in zip(proposals, gt_bboxes, gt_labels):
            has_gt = len(gt_bboxes_i) > 0
            match_quality_matrix = mo_utils.box_iou(gt_bboxes_i, proposals_i[:, :4])
            matched_idxs, matched_labels = self._mather(match_quality_matrix)
            sampled_idxs, proposal_labels_i = self.sample_proposals(
                matched_idxs, matched_labels, gt_labels_i
            )
            proposals_i = proposals_i[sampled_idxs]

            if has_gt:
                sampled_gt_idxs = matched_idxs[sampled_idxs]
                proposal_targets = gt_bboxes_i[sampled_gt_idxs]
            else:
                proposal_targets = gt_bboxes_i.new_zeros((len(sampled_idxs)), 4)

            num_bg_samples.append((proposal_labels_i == self._num_classes).sum().item())
            num_fg_samples.append(proposal_labels_i.numel() - num_bg_samples[-1])
            proposals_with_gt.append(
                {
                    'proposals': proposals_i,
                    'gt_labels': proposal_labels_i,
                    'gt_bboxes': proposal_targets,
                }
            )

        # TODO: log the information
        # Log the number of fg/bg samples that are selected for training ROI heads

        return proposals_with_gt

    def forward(self, images, features, proposals, gt_bboxes, gt_labels):
        """

        Args:
            images (Tensor): Image source tensor with shape (N, C, H, W).
            features (list[Tensor]): List of L tensors, each with shape (N, Ci, Hi, Wi).
            proposals (list[Tensor]): List of N tensors, each with shape (K, B+1).
            gt_bboxes (list[Tensor]): List of N tensors, each with shape (Ti, B).
            gt_labels (list[Tensor]): List of N tensors, each with shape (Ti,).

        Returns:

        """
        self.label_and_sample_proposals(proposals, gt_bboxes, gt_labels)
        return images
