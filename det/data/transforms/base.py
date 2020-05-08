from __future__ import absolute_import, division, print_function

import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Union

import numpy as np
from foundation.registry import Registry
from foundation.transforms import Transform

__all__ = ['TransformGenRegistry', 'TransformGen']


class TransformGenRegistry(Registry):
    """Registry of transform generators."""
    pass


class TransformGen(object, metaclass=ABCMeta):
    """A wrapper that creates a :class:`Transform` based on the given image and annotations.

    It creates a :class:`Transform` based on the given image and optionally the annotations,
    sometimes with randomness. The transform can then be used to transform images or other data
    (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class
    is that the image itself and annotations are sufficient to instantiate a transform.
    When this assumption is not true, you need to create the transforms by your own.

    A list of `TransformGen` can be applied with :func:`apply_transform_gens`.
    """

    def __init__(self) -> None:
        """Rewrites it to avoid raise AssertionError in :meth:`__repr__` due to *args, **kwargs."""
        pass

    @abstractmethod
    def get_transform(self, image: np.ndarray, annotations: Optional[Any] = None) -> Transform:
        """Gets a :class:`Transform` based on the given image.

        Args:
            image: Array of shape HxWxC or HxW.
            annotations: Annotations of image.
        """
        pass

    @staticmethod
    def _rand_range(
        low=1.0,
        high: Optional[float] = None,
        size: Optional[int] = None,
    ) -> Union[np.ndarray, float]:
        """Uniforms float random number between low and high.

        See :func:`np.random.uniform`.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self) -> str:
        """Produces something like:
        MyPipeline(field1={self._field1}, field2={self._field2})
        """
        try:
            sig = inspect.signature(self.__init__)
            items = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"

                attr_name = '_{}'.format(name)
                assert hasattr(self, attr_name), (
                    'Attribute {} not found! '
                    'Default __repr__ only works if attributes match the constructor. '
                    'The matched attribute name for parameter name `a` is `_a`'.format(attr_name)
                )
                attr = getattr(self, attr_name)
                items.append('{}={!r}'.format(name, attr))
            return '{}({})'.format(self.__class__.__name__, ', '.join(items))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__
