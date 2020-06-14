from __future__ import absolute_import, division, print_function

import math
from typing import Iterator, List, Optional, Tuple, Union

import torch
from torch import nn

from det.structures import Boxes, RotatedBoxes

_T = Union[List[float], List[List[float]]]


class BufferList(nn.Module):
    """The same as nn.ParameterList, but for buffers."""

    def __init__(self, buffers: Optional[List[torch.Tensor]] = None) -> None:
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers: List[torch.Tensor]) -> 'BufferList':
        offset = len(self)
        for i, buffer in buffers:
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self) -> int:
        return len(self._buffers)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self._buffers.values())


def _broadcast_params(params: _T, num_features: int, name: str) -> List[List[float]]:
    """Broadcasts anchors of that single size (or aspect ratio) over all feature maps.

    If params is List[float], or List[List[float]] with len(params) == 1, repeat it num_features
    time.

    Returns:
        params for each feature.
    """
    if not isinstance(params, (list, tuple)):
        raise TypeError('{} in anchor generator has to be a list!. Got {}'.format(name, params))
    if len(params) == 0:
        raise ValueError('{} in anchor generator cannot be empty!'.format(params))

    if not isinstance(params[0], (list, tuple)):  # List[float]
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    if len(params) != num_features:
        raise ValueError(
            'Got {} of length {} in anchor generator, but the number of input features is {}!'
            .format(name, len(params), num_features)
        )
    return params


def _create_grid_offsets(
    size: Tuple[int, int],
    stride: int,
    offset: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height, grid_width = size
    shifts_x = torch.arange(
        offset * stride, grid_width * stride, step=stride, dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        offset * stride, grid_height * stride, step=stride, dtype=torch.float32, device=device
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


class DefaultAnchorGenerator(nn.Module):
    """Computes anchors in the standard ways described in
    `Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`_.

    .. _`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`:
        https://arxiv.org/abs/1506.01497
    """
    # The dimension of each anchor box.
    box_dim = 4

    def __init__(
        self, sizes: _T, aspect_ratios: _T, strides: List[int], offset: float = 0.5
    ) -> None:
        super(DefaultAnchorGenerator, self).__init__()

        if not (0.0 <= offset <= 1.0):
            raise ValueError('offset should be between 0.0 and 1.0. Got {}'.format(offset))

        self._strides = strides
        self._offset = offset

        num_features = len(strides)
        sizes = _broadcast_params(sizes, num_features, 'sizes')
        aspect_ratios = _broadcast_params(aspect_ratios, num_features, 'aspect_ratios')
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

    @property
    def num_cell_anchors(self) -> List[int]:
        """Alias of `num_anchors`."""
        return self.num_anchors

    @property
    def num_anchors(self) -> List[int]:
        """Returns the number of anchors at every pixel location on each feature map.

        For example, if at every pixel we use anchors of 3 aspect ratios and 5 sizes, the number of
        anchors is 15. In standard RPN models, `num_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _calculate_anchors(
        self, sizes: List[List[float]], aspect_ratios: List[List[float]]
    ) -> BufferList:
        """Calculates base anchors which are centered at (0, 0) on each feature map."""
        cell_anchors = [
            self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    def generate_cell_anchors(
        self,
        sizes: List[float] = (32, 64, 128, 256, 512),
        aspect_ratios: List[float] = (0.5, 1, 2)
    ) -> torch.Tensor:
        """Generates anchors which are centered at (0, 0).

        Generate a tensor storing canonical anchor boxes, which are all anchor boxes of different
        sizes and aspect_ratios centered at (0, 0). We can later build the set of anchors for a full
        feature map by shifting and tiling these tensors (see :meth:`_grid_anchors`).

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes in XYXY
            format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size ** 2
            for aspect_ratio in aspect_ratios:
                # h / w = ar -> h = w * ar
                # h * w = area
                # w * w * ar = area -> w = sqrt(area / ar)
                w = math.sqrt(area / aspect_ratio)
                h = w * aspect_ratio
                x1, y1, x2, y2 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x1, y1, x2, y2])
        return torch.tensor(anchors)

    def _grid_anchors(self, grid_sizes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        """
        Returns:
            list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4.
        """
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self._strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self._offset, base_anchors.device)
            # shifts has shape (h * w, 4), where the first w elements correspond to the first row of
            # shifts
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # Add A anchors (1, A, 4) to K shifts (K, 1, 4) to get shifted anchors
            # (K, A, 4), reshape to (K * A, 4). First A rows correspond to A anchors of
            # (0, 0) in feature map, then (0, 1), (0, 2), ...
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def forward(self, features: List[torch.Tensor]) -> List[Boxes]:
        """
        Args:
            features: List of backbone feature maps on which to generate anchors.

        Returns:
            A list of Boxes containing all the anchors for each feature map (i.e. the cell anchors
            repeated over all locations in the feature map). The number of anchors of each feature
            map is Hi x Wi x num_cell_anchors, where Hi, Wi are resolution of the feature map
            divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_feature_maps]


class RotatedAnchorGenerator(nn.Module):
    """Computes rotated anchors used by Rotated RPN (RRPN), described in
    "Arbitrary-Oriented Scene Text Detection via Rotation Proposals".
    """
    # The dimension of each anchor box.
    box_dim = 5

    def __init__(
        self,
        sizes: _T,
        aspect_ratios: _T,
        angles: _T,
        strides: List[int],
        offset: float = 0.5,
    ) -> None:
        super(RotatedAnchorGenerator, self).__init__()

        if not (0.0 <= offset <= 1.0):
            raise ValueError('offset should be between 0.0 and 1.0. Got {}'.format(offset))

        self._strides = strides
        self._offset = offset

        num_features = len(strides)
        sizes = _broadcast_params(sizes, num_features, 'sizes')
        aspect_ratios = _broadcast_params(aspect_ratios, num_features, 'aspect_ratios')
        angles = _broadcast_params(angles, num_features, 'angles')
        self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios, angles)

    @property
    def num_cell_anchors(self) -> List[int]:
        """Alias of `num_anchors`."""
        return self.num_anchors

    @property
    def num_anchors(self) -> List[int]:
        """Returns the number of anchors at every pixel location on each feature map.

        For example, if at every pixel we use anchors of 3 aspect ratios, 2 sizes and 5 angles, the
        number of anchors is 30. In standard RRPN models, `num_anchors` on every feature map is the
        same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def _calculate_anchors(
        self, sizes: List[List[float]], aspect_ratios: List[List[float]], angles: List[List[float]]
    ) -> BufferList:
        cell_anchors = [
            self.generate_cell_anchors(size, aspect_ratio, angle).float()
            for size, aspect_ratio, angle in zip(sizes, aspect_ratios, angles)
        ]
        return BufferList(cell_anchors)

    def generate_cell_anchors(
        self,
        sizes: List[float] = (32, 64, 128, 256, 512),
        aspect_ratios: List[float] = (0.5, 1, 2),
        angles: List[float] = (-90, -60, -30, 0, 30, 60, 90),
    ) -> torch.Tensor:
        """Generates anchors which are centered at (0, 0).

        Generate a tensor storing canonical anchor boxes, which are all anchor boxes of different
        sizes, aspect_ratios, angles centered at (0, 0). We can later build the set of anchors for a
         full feature map by shifting and tiling these tensors (see :meth:`_grid_anchors`).

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios) * len(angles), 5) storing anchor boxes
            in (x_ctr, y_ctr, w, h, angle) format.
        """
        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # h / w = ar -> h = w * ar
                # h * w = area
                # w * w * ar = area -> w = sqrt(area / ar)
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                anchors.extend([0, 0, w, h, a] for a in angles)
        return torch.tensor(anchors)

    def _grid_anchors(self, grid_sizes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self._strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, stride, self._offset, base_anchors.device)
            zeros = torch.zeros_like(shift_x)
            shifts = torch.stack((shift_x, shift_y, zeros, zeros, zeros), dim=1)

            anchors.append((shifts.view(-1, 1, 5) + base_anchors.view(1, -1, 5)).reshape(-1, 5))

        return anchors

    def forward(self, features: List[torch.Tensor]) -> List[RotatedBoxes]:
        """
        Args:
            features: List of backbone feature maps on which to generate anchors.

        Returns:
            A list of Boxes containing all the anchors for each feature map (i.e. the cell anchors
            repeated over all locations in the feature map). The number of anchors of each feature
            map is Hi x Wi x num_cell_anchors, where Hi, Wi are resolution of the feature map
            divided by anchor stride.
        """
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
        return [RotatedBoxes(x) for x in anchors_over_all_feature_maps]
