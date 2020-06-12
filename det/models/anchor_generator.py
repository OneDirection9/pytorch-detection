from __future__ import absolute_import, division, print_function

from typing import Iterator, List, Optional, Union

import torch
from torch import nn

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
        self._num_features = len(strides)
        sizes = _broadcast_params(sizes, self._num_features, 'sizes')
        aspect_ratios = _broadcast_params(aspect_ratios, self._num_features, 'aspect_ratios')
