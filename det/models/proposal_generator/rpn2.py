from __future__ import absolute_import, division, print_function

from typing import List, Tuple

from torch import nn

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
        anchor_boundary_thresh: float = -1.0,
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
            anchor_boundary_thresh: Remove RPN anchors that go outside the image by BOUNDARY_THRESH
                pixels. Set to -1 or a large value, e.g. 100000, to disable pruning anchors.
            loss_weight: Weight to be multiplied to the loss.
            smooth_l1_beta: Beta parameter for the smooth L1 regression loss. Default to use L1
                loss.
        """
        super(RPN, self).__init__()
