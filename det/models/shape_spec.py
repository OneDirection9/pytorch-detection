from __future__ import absolute_import, division, print_function

from collections import namedtuple
from typing import Optional

__all__ = ['ShapeSpec']


class ShapeSpec(namedtuple('_ShapeSpec', ['channels', 'height', 'width', 'stride'])):
    """A simple structure that contains basic shape specification about a tensor.

    It is often used as the auxiliary inputs/outputs of models, to obtain the shape inference
    ability among PyTorch modules.

    Attributes:
        channels (int):
        height (int):
        width (int):
        stride (int):
    """

    def __new__(
        cls,
        *,
        channels: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> 'ShapeSpec':
        return super(ShapeSpec, cls).__new__(cls, channels, height, width, stride)
