from __future__ import absolute_import, division, print_function

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from det.structures import Boxes, ImageList, Instances, pairwise_iou
from ..matcher import Matcher
from ..proposal_generator.utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels


class ROIHeads(nn.Module):
    """ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to
    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclass of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        batch_size_per_image: int,
        positive_fraction: float,
        proposal_matcher: Matcher,
        proposal_append_gt: bool = True,
    ) -> None:
        """
        Args:
            num_classes: Number of classes. Used to label background proposals.
            batch_size_per_image: Number of proposals to sample for training.
            positive_fraction: Fraction of positive (foreground) proposals to sampling for training.
            proposal_matcher: Matcher that matches proposals and ground truth.
            proposal_append_gt: Whether to include ground truth as proposals as well.
        """
        super(ROIHeads, self).__init__()

        self._num_classes = num_classes
        self._batch_size_per_image = batch_size_per_image
        self._positive_fraction = positive_fraction
        self._proposal_matcher = proposal_matcher
        self._proposal_append_gt = proposal_append_gt

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Based on the matching between N proposals and M ground-truth, sample the proposals and
        set their classification labels.

        Args:
            matched_idxs: A vector of length N, each is the best-matched gt index in [0, M) for each
                proposal.
            matched_labels: A vector of length N, the matcher's label for each proposal.
            gt_classes: A vector of length M.

        Returns:
            Tensor: A vector of indices of sampled proposals. Each is in [0, N).
            Tensor: A vector of the same length, the classification label for each sampled proposal.
                Each sample is labeled as either a category in [0, num_classes) or the background
                (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self._num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self._num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self._batch_size_per_image, self._positive_fraction, self._num_classes
        )
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:  # yapf: disable
        """Prepares some proposals to be used to train the ROI heads.

        It performs box matching between `proposals` and `targets`, and assigns training labels to
        the proposals.
        It returns `self._batch_size_per_image` random samples from proposals and ground-truth
        boxes, with a fraction of positive that is no longer than `self._positive_fraction`.

        Args:
            proposals: See :meth:`forward`.
            targets: See :meth:`forward`.

        Returns:
            List[Instances]: Length `N` list of `Instances`s containing the proposals sampled for
            training. Each `Instances` has the following fields:

            - proposal_boxes: The proposal boxes
            - gt_boxes: The ground-truth box that the proposal is assigned to (this is only
              meaningful if the proposal has a label > 0; if label = 0 then the ground-truth box is
              random).

            Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self._proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self._proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for trg_name, trg_value in targets_per_image.get_fields().items():
                    if trg_name.startswith('gt_') and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self._num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # TODO: log the number of fg/bg samples that are selected for training ROI heads

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images:
            features: Input data as a mapping from feature map name to tensor. Axis 0 represents the
                number of images `N` in the input data; axes 1-3 are channels, height, and width,
                which may vary between feature maps (e.g., if a feature pyramid is used).
            proposals: Length `N` list of `Instances`. The i-th `Instances` contains object
                proposals for the i-th input image, with fields "proposal_boxes" and
                "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations for the i-th input
                image.  Specify `targets` during training only. It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            List[Instances]: Length `N` list of `Instances` containing the detected instances.
                Returned during inference only; may be [] during training.
            Dict[str, Tensor]: Mapping from a named loss to a tensor storing the loss. Used during
                training only.
        """
        raise NotImplementedError
