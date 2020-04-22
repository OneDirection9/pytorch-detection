# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from foundation.backends.torch.utils import weight_init
from .. import utils as mo_utils
from ..registry import ProposalHeadStash

__all__ = ['RPN']
"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Ti: number of instances of the i-th image
    Hi, Wi, Ci: height, width and depth of the i-th feature map
    B: size of the box parameterization
    K: number of top proposals

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not
    object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_objectness_logits: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


class RPNHead(nn.Module):
    """RPN Head with classification and regression heads.

    Args:
        in_channels (int): Number of channels of the input feature.
        num_anchors (int): Number of anchors to be predicted.
        box_dim (int): The dimension of each anchor box.
    """

    def __init__(self, in_channels, num_anchors, box_dim):
        super(RPNHead, self).__init__()

        # 3x3 conv for the hidden presentation
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, 1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, 1, stride=1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.normal_init(m, std=0.01)

    def forward(self, features):
        """
        Args:
            features (tuple[Tensor]): Tuple of input features.

        Returns:
            tuple[tuple[Tensor]]: Tuple of tuple of tensor.
        """
        pred_objectness_logits, pred_anchor_deltas = [], []
        for x in features:
            t = F.relu(self.conv(x), inplace=True)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))

        return tuple(pred_objectness_logits), tuple(pred_anchor_deltas)


@ProposalHeadStash.register('RPN')
class RPN(nn.Module):
    """Region Proposal Network introduced by the Faster R-CNN paper.

    Args:
        anchor_generator (nn.Module): Anchor generator that must have num_anchors property
            that return a list of number of anchors per position.
        in_channels (int, optional): Feature input channels, for multilevel feature maps,
            each level must have the same channel. Default: 256
        low_threshold (float, optional): Same as :obj:`models/utils/matcher/Matcher`.
            Default: 0.3
        high_threshold (float, optional): Same as :obj:`models/utils/matcher/Matcher`.
            Default: 0.7
        boundary_threshold (int, optional): If >= 0, then anchors that extend beyond the
            image boundary by more than boundary_thresh are not used in training. Set to
            a very large number or < 0 to disable this behavior. Only needed in training.
            Default: 0
        bbox_reg_weights (tuple[float], optional): Weights on (dx, dy, dw, dh) for
            normalizing RPN anchor regression targets. Default: (1., 1., 1., 1.)
        min_size (int, optional): Threshold that proposal height and width both need to be
            greater than. Default: 0
        batch_size_per_image (int, optional): Total number RPN examples per image.
            Default: 256
        positive_fraction (float, optional): Target fraction of positive examples per RPN
            minibatch. Default: 0.25
        pre_nms_topk (dict, optional): Dictionary that has keys named 'training' and
            'testing'. The value is the number of top scoring RPN proposals to keep after
            applying NMS. When FPN is used, this is *per FPN level* (not total).
            Default: {'training': 2000, 'testing': 1000}
        post_nms_topk (dict, optional): Dictionary that has keys named 'training' and
            'testing'. The value is the number of top scoring RPN proposals to keep after
            applying NMS. When FPN is used, this limit is applied per level and then again
            to the union of proposals from all levels.
            Default: {'training': 1000, 'testing': 1000}
        nms_thresh (float, optional): NMS threshold used on RPN proposals. Default: 0.7
        smooth_l1_beta (float, optional): The transition point from L1 to L2 loss.
            Set to 0.0 to make the loss simply L1. Default: 0.0
        loss_weight (float, optional): The weight of RPN losses. Default: 1.0
    """

    def __init__(
        self,
        anchor_generator,
        in_channels=256,
        # Matcher
        low_threshold=0.3,
        high_threshold=0.7,
        boundary_threshold=-1,
        bbox_reg_weights=(1., 1., 1., 1.),
        min_size=0,
        # Sampler
        batch_size_per_image=256,
        positive_fraction=0.5,
        # NMS
        pre_nms_topk=None,
        post_nms_topk=None,
        nms_thresh=0.7,
        # Loss
        smooth_l1_beta=0.0,
        loss_weight=1.0
    ):
        super(RPN, self).__init__()

        assert isinstance(anchor_generator, nn.Module)
        num_anchors = anchor_generator.num_anchors
        assert len(set(num_anchors)) == 1, 'Each level must have same number of anchors'
        self._box_dim = anchor_generator.box_dim

        self.anchor_generator = anchor_generator
        self.rpn_head = RPNHead(in_channels, num_anchors[0], self._box_dim)

        self._matcher = mo_utils.Matcher(low_threshold, high_threshold, True)
        self._boundary_threshold = boundary_threshold
        self._bbox_reg_weights = bbox_reg_weights
        self._min_size = min_size
        self._batch_size_per_image = batch_size_per_image
        self._balanced_sampler = mo_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction,
        )

        # Default pre/post_nms_topk is for P2, P3, P4, P5, P6 per FPN level
        if pre_nms_topk is None:
            pre_nms_topk = {'training': 2000, 'testing': 1000}
        if post_nms_topk is None:
            post_nms_topk = {'training': 1000, 'testing': 1000}
        self._pre_nms_topk = pre_nms_topk
        self._post_nms_topk = post_nms_topk
        self._nms_thresh = nms_thresh

        self._smooth_l1_beta = smooth_l1_beta
        self._loss_weight = loss_weight

    @property
    def pre_nms_topk(self):
        if self.training:
            return self._pre_nms_topk['training']
        return self._pre_nms_topk['testing']

    @property
    def post_nms_topk(self):
        if self.training:
            return self._post_nms_topk['training']
        return self._post_nms_topk['testing']

    def get_ground_truth(self, image_sizes, anchors, gt_bboxes):
        """Gets ground truth logits and deltas for each anchor.

        Args:
            image_sizes (list[Tensor]): List of N tensors, each has 2 values of image
                original height and width respectively.
            anchors (list[list[Tensor]]): List of N lists of L tensors, each with shape
                (Hi*Wi*A, B)
            gt_bboxes (list[Tensor]): List of N tensors, each with shape (Ti, B).

        Returns:
            gt_objectness_logits, gt_anchor_deltas: List of L anchors, each with shape
                (N, Hi*Wi*A,) and (N, Hi*Wi*A, B) respectively.
        """

        def resample(label):
            """Resample label in place."""
            pos_idx, neg_idx = self._balanced_sampler(label)

            label.fill_(mo_utils.IGNORED)
            label.scatter_(0, pos_idx, mo_utils.POSITIVE)
            label.scatter_(0, neg_idx, mo_utils.NEGATIVE)
            return label

        gt_objectness_logits, gt_anchor_deltas = [], []

        num_anchors_per_map = [x.shape[0] for x in anchors[0]]
        # Cat anchors from all feature maps
        # List of N tensors, each with shape (num_anchors_per_image, B)
        anchors = [torch.cat(anchors_i, dim=0) for anchors_i in anchors]

        for image_size_i, anchors_i, gt_bboxes_i in zip(image_sizes, anchors, gt_bboxes):
            match_quality_matrix = mo_utils.box_iou(gt_bboxes_i, anchors_i)
            matched_idxs, gt_objectness_logits_i = self._matcher(match_quality_matrix)
            del match_quality_matrix

            if self._boundary_threshold >= 0:
                # Ignore anchors that go out of the boundaries of the image
                idxs_inside = mo_utils.box_idxs_inside_image(
                    anchors_i, image_size_i, self._boundary_threshold
                )
                gt_objectness_logits_i[~idxs_inside] = mo_utils.IGNORED

            if len(gt_bboxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as bg
                gt_anchor_deltas_i = torch.zeros_like(anchors_i)
            else:
                matched_gt_bboxes = gt_bboxes_i[matched_idxs]
                gt_anchor_deltas_i = mo_utils.box_deltas(
                    anchors_i,
                    matched_gt_bboxes,
                    self._bbox_reg_weights,
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        # Stack to: (N, num_anchors_per_image)
        gt_objectness_logits = torch.stack(
            # Sampling a batch of anchors to compute losses
            [resample(label) for label in gt_objectness_logits],
            dim=0
        )

        # Log the number of positive/negative anchors per-image that's used in training
        # TODO: log information when training
        # num_pos_anchors = (gt_objectness_logits == mo_utils.POSITIVE).sum().item()
        # num_neg_anchors = (gt_objectness_logits == mo_utils.NEGATIVE).sum().item()

        # Split to tuple of L tensors, each with shape (N, Hi*Wi*A)
        gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)

        # Stack to: (N, num_anchors_per_image, B)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
        # Split to tuple of L tensors, each with shape (N, Hi*Wi*A, B)
        gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)

        return gt_objectness_logits, gt_anchor_deltas

    def compute_losses(
        self, pred_objectness_logits, pred_anchor_deltas, gt_objectness_logits, gt_anchor_deltas
    ):
        """Computes RPN losses including objectness and localization losses.

        Args:
            pred_objectness_logits (list[Tensor]): List of L tensors, each with shape
                (N, Hi*Wi*A,).
            pred_anchor_deltas (list[Tensor]): List of L tensors, each with shape
                (N, Hi*Wi*A, B).
            gt_objectness_logits (list[Tensor]): List of L tensors, each with shape
                (N, Hi*Wi*A,).
            gt_anchor_deltas (list[Tensor]): List of L tensors, each with shape
                (N, Hi*Wi*A, B).

        Returns:
            losses (dict): normalized objectness and localization losses.
        """
        # Check shape across all feature maps
        for pred_objectness_logits_i, gt_objectness_logits_i in \
                zip(pred_objectness_logits, gt_objectness_logits):
            assert pred_objectness_logits_i.shape == gt_objectness_logits_i.shape
        for pred_anchor_deltas_i, gt_anchor_deltas_i in \
                zip(pred_anchor_deltas, gt_anchor_deltas):
            assert pred_anchor_deltas_i.shape == gt_anchor_deltas_i.shape

        num_images = gt_objectness_logits[0].shape[0]

        # Concat from all feature maps
        gt_objectness_logits = torch.cat(
            [
                # (N, Hi*Wi*A) -> (N*Hi*Wi*A,)
                x.flatten() for x in gt_objectness_logits
            ],
            dim=0,
        )
        gt_anchor_deltas = torch.cat(
            [
                # (N, Hi*Wi*A, B) -> (N*Hi*Wi*A, B)
                x.reshape(-1, x.shape[-1]) for x in gt_anchor_deltas
            ],
            dim=0,
        )

        # Concat from all feature maps
        pred_objectness_logits = torch.cat(
            [
                # (N, Hi*Wi*A) -> (N*Hi*Wi*A,)
                x.flatten() for x in pred_objectness_logits
            ],
            dim=0,
        )
        pred_anchor_deltas = torch.cat(
            [
                # (N, Hi*Wi*A, B) -> (N*Hi*Wi*A, B)
                x.reshape(-1, x.shape[-1]) for x in pred_anchor_deltas
            ],
            dim=0,
        )

        # Compute objectness loss
        valid_masks = gt_objectness_logits != mo_utils.IGNORED
        objectness_loss = F.binary_cross_entropy_with_logits(
            pred_objectness_logits[valid_masks],
            gt_objectness_logits[valid_masks].to(torch.float32),
            reduction='sum',
        )
        # Compute localization loss
        pos_masks = gt_objectness_logits == mo_utils.POSITIVE
        localization_loss = mo_utils.smooth_l1_loss(
            pred_anchor_deltas[pos_masks],
            gt_anchor_deltas[pos_masks],
            self._smooth_l1_beta,
            reduction='sum'
        )

        normalizer = 1.0 / (self._batch_size_per_image * num_images)
        loss_cls = objectness_loss * normalizer
        loss_loc = localization_loss * normalizer
        losses = {'loss_rpn_cls': loss_cls, 'loss_rpn_loc': loss_loc}

        return losses

    def predict_proposals(self, anchors, pred_anchor_deltas):
        """Transforms anchors into proposals by applying the predicted anchor deltas.

        Args:
            anchors (list[list[Tensor]]): List of N list of L tensors, each with shape
                (Hi*Wi*A, B).
            pred_anchor_deltas (list[Tensor]): List of L tensors, each with shape
                (N, Hi*Wi*A, B).

        Returns:
            proposals (list[Tensor]): List of L tensors, each with shape (N, Hi*Wi*A, B).
        """
        N, B = len(anchors), self._box_dim
        proposals = []
        # Transpose anchors from image-by-feature (N, L) to feature-by-image (L, N)
        anchors = list(zip(*anchors))
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            # (N, Hi*Wi*A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            anchors_i = torch.cat(anchors_i, dim=0)
            proposals_i = mo_utils.apply_deltas(
                pred_anchor_deltas_i, anchors_i, self._bbox_reg_weights
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))

        return proposals

    def find_top_proposals(self, proposals, pred_objectness_logits, image_sizes):
        """For each feature map, select the `pre_nms_topk` highest scoring proposals,
        apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
        highest scoring proposals among all the feature maps if `training` is True,
        otherwise, returns the highest `post_nms_topk` scoring proposals for each
        feature map.

        Args:
            proposals (list[Tensor]): List of L tensors, each has shape (N, Hi*Wi*A, B).
            pred_objectness_logits (list[Tensor]): List of L tensors, each has shape
                (N, Hi*Wi*A).
            image_sizes (list[Tensor]): List of N tensors, each has 2 values of image
                original height and width respectively.

        Returns:
            proposals (list[Tensor]): List of N tensors, each with shape (topk, B+1).
        """
        num_images = len(image_sizes)
        device = proposals[0].device

        # 1. Select top-k anchor for every level and every image
        topk_scores, topk_proposals, level_ids = [], [], []
        batch_idx = torch.arange(num_images, device=device)
        for level_id, proposals_i, logits_i in \
                zip(itertools.count(), proposals, pred_objectness_logits):
            Hi_Wi_A = logits_i.shape[1]
            num_proposals_i = min(self.pre_nms_topk, Hi_Wi_A)

            # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
            # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
            logits_i, idx = logits_i.sort(descending=True, dim=1)
            topk_scores_i = logits_i[batch_idx, :num_proposals_i]
            topk_idx = idx[batch_idx, :num_proposals_i]  # (N, K)
            topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # (N, K, B)

            topk_scores.append(topk_scores_i)
            topk_proposals.append(topk_proposals_i)
            level_ids.append(
                torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device)
            )

        # 2. Concat all levels together
        topk_scores = torch.cat(topk_scores, dim=1)  # (N, L*K)
        topk_proposals = torch.cat(topk_proposals, dim=1)  # (N, L*K, B)
        level_ids = torch.cat(level_ids, dim=0)  # (L*K,)

        # 3. For each image, run a per-level NMS, and choose topk results
        results = []
        for n, image_size in enumerate(image_sizes):
            boxes, scores = topk_proposals[n], topk_scores[n]  # (L*K, B), (L*K,)
            mo_utils.inplace_clip(boxes, image_size)

            # Filter empty boxes
            keep = mo_utils.remove_small_boxes(boxes, self._min_size)
            lvl = level_ids
            if keep.sum().item() != boxes.shape[0]:
                boxes, scores, lvl = boxes[keep], scores[keep], level_ids[keep]

            keep = box_ops.batched_nms(boxes, scores, lvl, self._nms_thresh)
            keep = keep[:self.post_nms_topk]

            results.append(torch.cat([boxes[keep], scores[keep].unsqueeze(1)], dim=1))  # (K, B+1)

        return results

    def forward(self, image_sizes, features, gt_bboxes=None):
        """
        Args:
            image_sizes (list[Tensor]): List of N tensors, each has 2 values of image
                original height and width respectively.
            features (tuple[Tensor]): Tuple of L tensors.
            gt_bboxes (list[Tensor]): List of N tensors, each with shape (Ti, B).

        Returns:
            proposals (list[Tensor]): List of N tensors, each with shape (K, B+1)
            losses (dict[str->Tensor]): RPN losses.
        """
        anchors = self.anchor_generator(features)
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        N, B = len(anchors), self._box_dim
        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            x.permute(0, 2, 3, 1).reshape(N, -1) for x in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) ->
            #           (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, B, x.shape[-2],
                   x.shape[-1]).permute(0, 3, 4, 1, 2).reshape(N, -1, B) for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_bboxes is not None
            gt_objectness_logits, gt_anchor_deltas = self.get_ground_truth(
                image_sizes,
                anchors,
                gt_bboxes,
            )
            losses = self.compute_losses(
                pred_objectness_logits, pred_anchor_deltas, gt_objectness_logits, gt_anchor_deltas
            )
            losses = {k: v * self._loss_weight for k, v in losses.items()}
        else:
            losses = {}

        with torch.no_grad():
            proposals = self.predict_proposals(anchors, pred_anchor_deltas)
            proposals = self.find_top_proposals(proposals, pred_objectness_logits, image_sizes)
            # For RPN-only models, the proposals are the final output and we return them
            # in high-to-low confidence order.
            # For end-to-end models, the RPN proposals are an intermediate state and this
            # sorting is actually not needed. But the cost is negligible.
            idxs = [torch.sort(p[:, 4], descending=True)[1] for p in proposals]
            proposals = [p[idx] for p, idx in zip(proposals, idxs)]

        return proposals, losses
