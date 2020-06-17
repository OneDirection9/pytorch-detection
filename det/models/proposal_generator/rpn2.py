from __future__ import absolute_import, division, print_function

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from det.structures import ImageList, Instances
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from .registry import ProposalGeneratorRegistry

__all__ = ['RPN']
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
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self,
                features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features: List of feature maps

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


@ProposalGeneratorRegistry.register('RPN')
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
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
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
                before NMS, in training and testing.
            post_nms_topk: (train, test) that represents the number of top k proposals to select
                after NMS, in training and testing.
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

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images: Input images of length `N`
            features: Input data as a mapping from feature map name to tensor. Axis 0 represents the
                number of images `N` in the input data; axes 1-3 are channels, height, and width,
                which may vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances: A length `N` list of `Instances`s. Each `Instances` stores ground-truth
                instances for the corresponding image.

        Returns:
            proposals: Contains fields "proposal_boxes", "objectness_logits"
            loss:
        """
        features = [features[f] for f in self._in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.premute(0, 2, 3, 1).flatten(1) for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2],
                   x.shape[-1]).permute(0, 3, 4, 1, 2).flatten(1, -2) for x in pred_anchor_deltas
        ]
        print(anchors, pred_anchor_deltas, pred_objectness_logits)
