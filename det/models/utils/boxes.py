# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import math

import torch

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


def box_area(boxes):
    """Calculates area of boxes. The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes (Tensor[N, 4]): N boxes in (xmin, ymin, xmax, ymax) order.

    Returns:
        Tensor[N]: Area.
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return w * h


def box_iou(boxes1, boxes2):
    """Calculates IoU between all N x M pairs of boxes.

    Args:
        boxes1 (Tensor[N, 4]): N boxes in (xmin, ymin, xmax, ymax) order.
        boxes2 (Tensor[M, 4]): M boxes in (xmin, ymin, xmax, ymax) order.

    Returns:
        Tensor[N, M]: IoU
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    wh = torch.min(boxes1[:, None, 2:],
                   boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    wh = wh.clamp_(min=0)  # [N, M, 2]
    inter = wh.prod(dim=2)  # [N, M]
    del wh

    # Handle empty bboxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def box_idxs_inside_image(boxes, image_size, boundary_threshold=0):
    """Finds the boxes that inside the image.

    Args:
        boxes (Tensor[N, 4]): N boxes in (xmin, ymin, xmax, ymax) order.
        image_size (tuple[2]): Tuple of image size in (h, w) format.
        boundary_threshold (int, optional): Boxes that extend beyond the reference box
            boundary by more than boundary_threshold are considered 'outside'. Default: 0

    Returns:
        Tensor[N]: A binary vector indicating whether each box is inside the image.
    """
    h, w = image_size
    idxs_inside = (
        (boxes[..., 0] >= - boundary_threshold)
        & (boxes[..., 1] >= -boundary_threshold)
        & (boxes[..., 2] < w + boundary_threshold)
        & (boxes[..., 3] < h + boundary_threshold)
    )  # yapf: disable
    return idxs_inside


def box_deltas(src_boxes, target_boxes, weights):
    """Calculates dx, dy, dw, dh.

    Args:
        src_boxes (Tensor[N, 4]): N boxes in (xmin, ymin, xmax, ymax) order.
        target_boxes (Tensor[N, 4]): N boxes in (xmin, ymin, xmax, ymax) order.
        weights (tuple[4]): Scaling factors that are applied to the (dx, dy, dw, dh)
            deltas. In Fast R-CNN, these were originally set such that the deltas have
            unit variance; now they are treated as hyperparameters of the system.

    Returns:
        Tensor[N, 4]: (dx, dy, dw, dh) for N source-target pairs.
    """
    src_widths = src_boxes[:, 2] - src_boxes[:, 0]
    src_heights = src_boxes[:, 3] - src_boxes[:, 1]
    src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
    src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

    target_widths = target_boxes[:, 2] - target_boxes[:, 0]
    target_heights = target_boxes[:, 3] - target_boxes[:, 1]
    target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
    target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

    wx, wy, ww, wh = weights
    dx = wx * (target_ctr_x - src_ctr_x) / src_widths
    dy = wy * (target_ctr_y - src_ctr_y) / src_heights
    dw = ww * torch.log(target_widths / src_widths)
    dh = wh * torch.log(target_heights / src_heights)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    assert (src_widths > 0).all().item(), 'src_bboxes are not valid'
    return deltas


def apply_deltas(deltas, boxes, weights, scale_clamp=_DEFAULT_SCALE_CLAMP):
    """Applies transformation `deltas` (dx, dy, dw, dh) to `bboxes`.

    Args:
        deltas (Tensor): Transformation deltas of shape (N, k*4), where k >= 1. deltas[i]
            represents k potentially different class-specific box transformations for the
            single box boxes[i].
        boxes (Tensor): Boxes to transform, of shape (N, 4)
        weights (tuple[4]): Scaling factors that are applied to the (dx, dy, dw, dh)
            deltas. In Fast R-CNN, these were originally set such that the deltas have
            unit variance; now they are treated as hyperparameters of the system.
        scale_clamp (float): When predicting deltas, the predicted box scaling factors
            (dw and dh) are clamped such that they are <= scale_clamp.

    Returns:
        pred_boxes (Tensor): The same shape as `boxes` in (x1, y1, x2, y2) order.
    """
    assert torch.isfinite(deltas).all().item(), 'Box regression deltas become infinite!'
    boxes = boxes.to(deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=scale_clamp)
    dh = torch.clamp(dh, max=scale_clamp)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2
    return pred_boxes


def inplace_clip(boxes, size):
    """Inplace Clip boxes so that they lie inside an image of size `size`.

    Args:
        boxes (Tensor[N, 4]): Boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): Size of the image
    """
    assert torch.isfinite(boxes).all(), 'Box tensor contains infinite!'
    h, w = size
    boxes[:, 0].clamp_(min=0, max=w)
    boxes[:, 1].clamp_(min=0, max=h)
    boxes[:, 2].clamp_(min=0, max=w)
    boxes[:, 3].clamp_(min=0, max=h)


def remove_small_boxes(boxes, min_size):
    """Removes boxes which contains at least one side no greater than min_size.

    Arguments:
        boxes (Tensor[N, 4]): Boxes in (x1, y1, x2, y2) format.
        min_size (int): Minimum size.

    Returns:
        keep (Tensor[K]): Indices of the boxes that have both sides larger than min_size.
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths > min_size) & (heights > min_size)
    return keep


def add_ground_truth_to_proposals(gt_bboxes, proposals):
    """

    Args:
        gt_bboxes (list[Tensor]): List of N tensors, each with shape (Ti, B).
        proposals (list[Tensor]): List of N tensors, each with shape (K, B+1).
    """
    assert len(gt_bboxes) == len(proposals)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_bboxes_i, proposals_i)
        for gt_bboxes_i, proposals_i in zip(gt_bboxes, proposals)
    ]


def add_ground_truth_to_proposals_single_image(gt_bboxes, proposals):
    """Augments `proposals` with ground-truth boxes from `gt_bboxes`.

    Args:
        gt_bboxes (Tensor): Boxes with shape (Ti, B).
        proposals (Tensor): Proposals with shape (K, B+1), where the last element in
            second dimension is objectness logits.

    Returns:
        new_proposals (Tensor): Proposals with shape (K+Ti, B+1).
    """
    device = proposals.device
    # Concatenating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) \approx 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))

    gt_logits = gt_logit_value * torch.ones((len(gt_bboxes), 1), device=device)
    gt_proposals = torch.cat([gt_bboxes, gt_logits], dim=1)  # (Ti, B+1)

    new_proposals = torch.cat([proposals, gt_proposals], dim=0)  # (K+Ti, B+1)
    return new_proposals
