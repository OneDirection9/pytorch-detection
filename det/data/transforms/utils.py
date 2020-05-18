from __future__ import absolute_import, division, print_function

from typing import List, Tuple, Union

import numpy as np
from foundation.transforms import Transform, TransformList

from .transform_gen import TransformGen

__all__ = ['apply_transforms']


def apply_transforms(
    transforms: List[Union[Transform, TransformGen]],
    image: np.ndarray,
) -> Tuple[np.ndarray, TransformList]:
    """Applies a list of :class:`Transform` or :class:`TransformGen` on the input image, and
    returns the transformed image and a list of transforms.

    We cannot simply create and return all transforms without applying it to the image, because
    a subsequent transform may need the output of the previous one.

    Args:
        transforms: List of :class:`Transform` or :class:`TransformGen` instance to be applied.
        image: Array of shape HxW or HxWx3.

    Returns:
        ndarray: The transformed image.
        TransformList: Contain the transforms that's used to other data.
    """
    for g in transforms:
        if not isinstance(g, (Transform, TransformGen)):
            raise TypeError('Expected Transform or TransformGen. But got {}'.format(type(g)))

    tfms = []
    for g in transforms:
        tfm = g.get_transform(image) if isinstance(g, TransformGen) else g
        if not isinstance(tfm, Transform):
            raise TypeError(
                'TransformGen {} must return an instance of Transform! Got {} instead'.format(
                    g, tfm
                )
            )
        image = tfm.apply_image(image)
        tfms.append(tfm)
    return image, TransformList(tfms)
