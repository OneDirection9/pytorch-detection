# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import inspect
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from foundation.transforms import Transform, TransformList

__all__ = ['Augmentation', 'apply_augmentations']


class Augmentation(object, metaclass=ABCMeta):
    """A wrapper that creates a :class:`Transform` based on the given image.

    It creates a :class:`Transform` based on the given image, sometimes with randomness.
    The transform can then be used to transform images or other data
    (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class
    is that the image itself is sufficient to instantiate a transform.
    When this assumption is not true, you need to create the transforms by your own.

    A list of `Augmentation` can be applied with :func:`apply_augmentations`.
    """

    def __init__(self) -> None:
        """Rewrites it to avoid raise AssertionError in :meth:`__repr__` due to *args, **kwargs."""
        pass

    @abstractmethod
    def get_transform(self, image: np.ndarray) -> Transform:
        """Gets a :class:`Transform` based on the given image.

        Args:
            image: Array of shape HxWxC or HxW.
        """
        pass

    @staticmethod
    def _rand_range(
        low=1.0, high: Optional[float] = None, size: Optional[int] = None
    ) -> Union[np.ndarray, float]:  # yapf:disable
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
        MyPipeline(field1={self.field1}, field2={self.field2})
        """
        try:
            sig = inspect.signature(self.__init__)
            items = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"

                assert hasattr(self, name), (
                    'Attribute {} not found! '
                    'Default __repr__ only works if attributes match the constructor.'.format(name)
                )
                attr = getattr(self, name)
                items.append('{}={!r}'.format(name, attr))
            return '{}({})'.format(self.__class__.__name__, ', '.join(items))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


def apply_augmentations(
    augmentations: List[Union[Augmentation, Transform]], image: np.ndarray
) -> Tuple[np.ndarray, TransformList]:  # yapf: disable
    """Applies a list of :class:`Augmentation` or :class:`Transform` on the input image, and
    returns the transformed image and a list of transforms.

    We cannot simply create and return all transforms without applying it to the image, because
    a subsequent transform may need the output of the previous one.

    Args:
        augmentations: List of :class:`Augmentation` or :class:`Transform` instance to be applied.
        image: Array of shape HxW or HxWx3.

    Returns:
        ndarray: The transformed image.
        TransformList: Contain the transforms that's used to other data.
    """
    for aug in augmentations:
        if not isinstance(aug, (Augmentation, Transform)):
            raise TypeError('Expected Augmentation or Transform. But got {}'.format(type(aug)))

    tfms = []
    for aug in augmentations:
        tfm = aug.get_transform(image) if isinstance(aug, Augmentation) else aug
        if not isinstance(tfm, Transform):
            raise TypeError(
                'Augmentation {} must return an instance of Transform. Got {}'.format(aug, tfm)
            )
        image = tfm.apply_image(image)
        tfms.append(tfm)
    return image, TransformList(tfms)
