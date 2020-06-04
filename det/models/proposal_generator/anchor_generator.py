# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import copy
import math

import torch
from torch import nn

from ..registry import AnchorGeneratorStash

__all__ = ['StandardAnchorGenerator']


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


@AnchorGeneratorStash.register('StandardAnchorGenerator')
class StandardAnchorGenerator(nn.Module):
    """For a set of feature maps, computes s set of anchors.

    Args:
        strides (tuple[int], optional): The strides for multilevel feature maps. Default
            is the stride for P2, P3, P4, P5, P6 in Faster R-CNN with FPN.
            Default: (4, 8, 16, 32, 64,)
        sizes (tuple[tuple[int]], optional): Anchor sizes given in absolute pixel w.r.t.
            the scaled network input. The SIZES[i] specifies the tuple of sizes to use
            for FEATURES[i]; len(SIZES) == len(FEATURES) must be true, or len(SIZES) == 1
            is true and SIZES[0] is used for all FEATURES.
            Default: ((32,), (64,), (128,), (256,), (512,),)
        aspect_ratios (tuple[tuple[int]], optional): Anchor aspect ratios.
            ASPECT_RATIOS[i] specifies the tuple of aspect ratios to use for FEATURES[i];
            len(ASPECT_RATIOS) == len(FEATURES) must be true, or len(ASPECT_RATIOS) == 1
            is true and ASPECT_RATIOS[0] is used for all FEATURES.
            Default: ((0.5, 1.0, 2.0,),)
        offset (float, optional): Relative offset between the center of the first anchor
            and the top-left corner of the image. Fraction of feature map stride (e.g.,
            0.5 means half stride). Allowed values are floats in [0, 1). Recommended value
            is 0.5, although it is not expected to affect model accuracy. Default: 0.0
    """

    def __init__(
        self,
        strides=(4, 8, 16, 32, 64,),
        sizes=((32,), (64,), (128,), (256,), (512,),),
        aspect_ratios=((0.5, 1.0, 2.0,),),
        offset=0.0
    ):  # yapf: disable
        super(StandardAnchorGenerator, self).__init__()

        assert 0. <= offset < 1, offset

        # If one size (or aspect ratio) is specified and there are multiple feature maps,
        # then we "broadcast" anchors of that single size (or aspect ratio) over all
        # feature maps.
        num_features = len(strides)
        if len(sizes) == 1:
            sizes *= num_features
        if len(aspect_ratios) == 1:
            aspect_ratios *= num_features
        assert len(sizes) == num_features
        assert len(aspect_ratios) == num_features

        self._strides = strides
        self._offset = offset

        # Calculate base anchors for each level feature map
        # Convert to buffer so that anchors are copied to the same device as the model
        self._base_anchors = BufferList(
            [self.generate_base_anchors(s, ar) for s, ar in zip(sizes, aspect_ratios)]
        )

    @property
    def num_anchors(self):
        """
        Returns:
            list[int]: Number of anchors at each feature map.
        """
        return [len(a) for a in self._base_anchors]

    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 4

    @classmethod
    def generate_base_anchors(cls, sizes, aspect_ratios):
        """Generates base anchors.

        Args:
            sizes (tuple[int]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after undergoing necessary
                scaling). The absolute size is given as the side length of a box.
            aspect_ratios (tuple[float]]): Aspect ratios of the boxes computed as box
                width / height.

        Returns:
            Tensor[N * M, 4]: Tensor storing anchor boxes in XYXY format where N is length
            of sizes and M is length of aspect_ratios.
        """
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # w / h = ar -> w = h * ar
                # h * w = area
                # h * h * ar = area -> h = sqrt(area / ar)
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])

        return torch.tensor(anchors)

    def _meshgrid(self, grid_size, stride, device):
        grid_h, grid_w = grid_size
        shifts_x = torch.arange(
            stride * self._offset, grid_w * stride, step=stride, dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            stride * self._offset,
            grid_h * stride,
            step=stride,
            dtype=torch.float32,
            device=device,
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        return shift_x, shift_y

    def grid_anchors(self, grid_sizes):
        """
        Args:
            grid_sizes (list[list[2]]: List of feature map sizes in (h, w) format.

        Returns:
            anchors (list[list[Tensor]]): List of L tensors each with shape (Hi*Wi*A, B)
        """
        anchors = []
        for grid_size, stride, base_anchors_i in \
                zip(grid_sizes, self._strides, self._base_anchors):
            shift_x, shift_y = self._meshgrid(grid_size, stride, base_anchors_i.device)
            # shifts has shape (h * w, 4), where the first feat_w elements correspond to
            # the first row of shifts
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # Add A anchors (1, A, 4) to K shifts (K, 1, 4) to get shifted anchors
            # (K, A, 4), reshape to (K * A, 4). First A rows correspond to A anchors of
            # (0, 0) in feature map, then (0, 1), (0, 2), ...
            anchors.append((shifts.view(-1, 1, 4) + base_anchors_i.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def forward(self, features):
        """
        Args:
            features (tuple[Tensor]): Tuple of L tensors, each with shape (N, Ci, Hi, Wi).

        Returns:
            list[list[Tensor]]: List of #image elements. Each is a list of #feature level
            Tensor.
        """
        num_images = features[0].shape[0]
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_map = self.grid_anchors(grid_sizes)

        anchors = [copy.deepcopy(anchors_over_all_feature_map) for _ in range(num_images)]
        return anchors
