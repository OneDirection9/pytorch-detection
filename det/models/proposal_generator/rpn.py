# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

from typing import Dict, List, Optional, Tuple, Union

import torch
from foundation.nn import smooth_l1_loss, weight_init
from torch import nn
from torch.nn import functional as F

from det.layers import ShapeSpec, cat
from det.structures import Boxes, ImageList, Instances, RotatedBoxes, pairwise_iou
from ..anchor_generator import DefaultAnchorGenerator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..sampling import subsample_labels
from .registry import ProposalGeneratorRegistry
from .utils import find_top_rpn_proposals

__all__ = ['StandardRPNHead', 'RPN', 'standard_rcnn_rpn']
"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    B: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`), or 5d for rotated boxes.

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_labels: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


class StandardRPNHead(nn.Module):
    """Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.

    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts objectness
    logits for each anchor and a second 1x1 conv predicts bounding-box deltas specifying how to
    deform each anchor into an object proposal.
    """

    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4) -> None:
        """
       Args:
            in_channels: Number of input feature channels. When using multiple input features, they
                must have the same number of channels.
            num_anchors: Number of anchors to predict for *each spatial position* on the feature
                map. The total number of anchors for each feature map will be `num_anchors * H * W`.
            box_dim: Dimension of a box, which is also the number of box regression predictions to
                make for each anchor. An axis aligned box has box_dim=4, while a rotated box has
                box_dim=5.
        """
        super(StandardRPNHead, self).__init__()
        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            weight_init.normal_init(l, std=0.01, bias=0)

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:  # yapf: disable
        """
        Args:
            features: List of feature maps.

        Returns:
            pred_objectness_logits: A list of L elements. Element i is a tensor of shape
                (N, A, Hi, Wi) representing the predicted objectness logits for all anchors. A is
                the number of cell anchors.
            pred_anchor_deltas: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


class RPN(nn.Module):
    """Region Proposal Network, introduced by :paper:`Faster R-CNN`."""

    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[int, int],
        post_nms_topk: Tuple[int, int],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        loss_weight: float = 1.0,
        smooth_l1_beta: float = 0.0,
    ) -> None:
        """
        Args:
            in_features: List of names of input features to use.
            head: A module that predicts logits and regression deltas for each level from a list of
                pre-level features.
            anchor_generator: A module that creates anchors from a list of features. Usually an
                instance of :class:`AnchorGenerator`.
            anchor_matcher: Label the anchors by matching then with ground truth.
            box2box_transform: Defines the transform from anchors boxes to instance boxes.
            batch_size_per_image: Number of anchors per image to sample for training.
            positive_fraction: Fraction of foreground anchors to sample for training.
            pre_nms_topk: (train, test) that represents the number of top k proposals to select
                before NMS, in training and testing. When FPN is used, this is *per FPN level* (not
                total).
            post_nms_topk: (train, test) that represents the number of top k proposals to select
                after NMS, in training and testing. When FPN is used, this limit is applied per
                level and then again to the union of proposals from all levels
            nms_thresh: NMS threshold used to de-duplicate the predicted proposals.
            min_box_size: Remove proposal boxes with any side smaller than this threshold, in the
                unit of input image pixels.
            loss_weight: Weight to be multiplied to the loss.
            smooth_l1_beta: Beta parameter for the smooth L1 regression loss. Default to use L1
                loss.
        """
        super(RPN, self).__init__()

        self._in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self._anchor_matcher = anchor_matcher
        self._box2box_transform = box2box_transform
        self._batch_size_per_image = batch_size_per_image
        self._positive_fraction = positive_fraction
        # Map from self.training state to train/test setting
        self._pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self._post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self._nms_thresh = nms_thresh
        self._min_box_size = min_box_size
        self._loss_weight = loss_weight
        self._smooth_l1_beta = smooth_l1_beta

    def _subsample_labels(self, label: torch.Tensor) -> torch.Tensor:
        """Randomly sample a subset of positive and negative examples, and overwrite the label
        vector to the ignore value (-1) for all elements that are not included in the sample.

        Args:
            label: A vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self._batch_size_per_image, self._positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors: Anchors for each feature map.
            gt_instances: The ground-truth instances for each image.

        Returns:
            gt_labels: List of #image tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * Ai). Label
                values are in {-1, 0, 1}, with meanings: -1 = ignore, 0: negative class; 1: positive
                class.
            matched_gt_boxes: List of #image tensors. i-th element is a Rx4 tensor. The values are
                matched gt boxes for each anchor. Values are undefined for those anchors not labeled
                as 1.
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = self._anchor_matcher(match_quality_matrix)
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO: wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N, AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    def losses(
        self,
        anchors: List[Union[Boxes, RotatedBoxes]],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Returns the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors: Anchors for each feature map, each has shape (Hi*Wi*A, B), where B is box
                dimension (4 or 5).
            pred_objectness_logits: A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A) representing the predicted objectness logits for all anchors.
            gt_labels: Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas: A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, B) representing the predicted "deltas" used to transform anchors to
                proposals.
            gt_boxes: Output of :meth:`label_and_sample_anchors`.

        Returns:
            A dict mapping from loss name to loss value. Loss names are: `loss_rpn_cls` for
                objectness classification and `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*A))
        anchors = type(anchors[0]).cat(anchors).tensor  # (sum(Hi*Wi*A), B)
        gt_anchor_deltas = [self._box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*A), B)

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        # num_pos_anchors = pos_mask.sum().item()
        # num_neg_anchors = (gt_labels == 0).sum().item()

        localization_loss = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            self._smooth_l1_beta,
            reduction='sum',
        )
        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction='sum',
        )
        normalizer = self._batch_size_per_image * num_images
        return {
            'loss_rpn_loc': localization_loss / normalizer,
            'loss_rpn_cls': objectness_loss / normalizer,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images: Input images of length `N`.
            features: Input data as a mapping from feature map name to tensor. Axis 0 represents the
                number of images `N` in the input data; axes 1-3 are channels, height, and width,
                which may vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances: A length `N` list of `Instances`s. Each `Instances` stores ground-truth
                instances for the corresponding image.

        Returns:
            proposals: Contains fields "proposal_boxes", "objectness_logits"
            loss: See :meth:`losses` or None.
        """
        features = [features[f] for f in self._in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1) for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]  # yapf: disable

        if self.training:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
            losses = {k: v * self._loss_weight for k, v in losses.items()}
        else:
            losses = {}

        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        return proposals, losses

    @torch.no_grad()
    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> List[Instances]:
        """Decodes all the predicted box regression deltas to proposals. Find the top proposals by
        applying NMS and removing boxes that are too small.

        Returns:
            proposals: List of N Instances. The i-th instances stores post_nms_topk object proposals
                for image i, sorted by their objectness score in descending order.
        """
        # The proposals are treated as fixed for approximate joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses, so is approximate.
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return find_top_rpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self._nms_thresh,
            self._pre_nms_topk[self.training],
            self._post_nms_topk[self.training],
            self._min_box_size,
            self.training,
        )

    def _decode_proposals(
        self, anchors: List[Union[Boxes, RotatedBoxes]], pred_anchor_deltas: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Transforms anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals: A list of L tensors. Tensor i has shape (N, Hi*Wi*A, B).
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            # (N, Hi*Wi*A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self._box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals


"""
Wrappers of RPN
"""


@ProposalGeneratorRegistry.register('Standard_RCNN_RPN')
def standard_rcnn_rpn(
    input_shape: Dict[str, ShapeSpec],
    in_features: List[str] = ('res4',),
    # anchor generator
    sizes: Union[List[float], List[List[float]]] = ((32, 64, 128, 256, 512),),
    aspect_ratios: Union[List[float], List[List[float]]] = ((0.5, 1.0, 2.0),),
    offset: float = 0.0,
    # Matcher
    thresholds: List[float] = (0.3, 0.7),
    labels: List[int] = (0, -1, 1),
    # Box2BoxTransform
    bbox_reg_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    # RPN
    batch_size_per_image: int = 256,
    positive_fraction: float = 0.5,
    pre_nms_topk: Tuple[int, int] = (12000, 6000),
    post_nms_topk: Tuple[int, int] = (2000, 1000),
    nms_thresh: float = 0.7,
    min_box_size: float = 0.0,
    loss_weight: float = 1.0,
    smooth_l1_beta: float = 0.0,
) -> RPN:
    """
    Args:
        input_shape: Output shape of backbone or neck.
        in_features: See :class:`RPN`.
        sizes: See :class:`DefaultAnchorGenerator`.
        aspect_ratios: See :class:`DefaultAnchorGenerator`.
        offset: See :class:`DefaultAnchorGenerator`.
        thresholds: See :class:`Matcher`.
        labels: See :class:`Matcher`.
        bbox_reg_weights: See :class:`Box2BoxTransform`.
        batch_size_per_image: See :class:`RPN`.
        positive_fraction: See :class:`RPN`.
        pre_nms_topk: See :class:`RPN`.
        post_nms_topk: See :class:`RPN`.
        nms_thresh: See :class:`RPN`.
        min_box_size: See :class:`RPN`.
        loss_weight: See :class:`RPN`.
        smooth_l1_beta: See :class:`RPN`.
    """
    input_shape = [input_shape[f] for f in in_features]

    strides = [s.stride for s in input_shape]
    anchor_generator = DefaultAnchorGenerator(sizes, aspect_ratios, strides, offset)

    # Standard RPN is shared across levels:
    in_channels = [s.channels for s in input_shape]
    assert len(set(in_channels)) == 1, 'Each level must have the same channel!'
    in_channels = in_channels[0]

    num_anchors = anchor_generator.num_anchors
    assert len(
        set(num_anchors)
    ) == 1, 'Each level must have the same number of anchors per spatial position'
    num_anchors = num_anchors[0]

    box_dim = anchor_generator.box_dim
    head = StandardRPNHead(in_channels=in_channels, num_anchors=num_anchors, box_dim=box_dim)

    anchor_matcher = Matcher(list(thresholds), list(labels), allow_low_quality_matches=True)

    box2box_transform = Box2BoxTransform(bbox_reg_weights)

    return RPN(
        in_features=in_features,
        head=head,
        anchor_generator=anchor_generator,
        anchor_matcher=anchor_matcher,
        box2box_transform=box2box_transform,
        batch_size_per_image=batch_size_per_image,
        positive_fraction=positive_fraction,
        pre_nms_topk=pre_nms_topk,
        post_nms_topk=post_nms_topk,
        nms_thresh=nms_thresh,
        min_box_size=min_box_size,
        loss_weight=loss_weight,
        smooth_l1_beta=smooth_l1_beta,
    )
