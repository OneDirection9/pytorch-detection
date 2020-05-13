from __future__ import absolute_import, division, print_function

from typing import Any, Dict, List, Optional

import numpy as np
from foundation.transforms import Transform

from .base import TransformGen


def apply_transform_gens(
    transform_gens: List[TransformGen],
    image: np.ndarray,
    annotations: Optional[Dict[str, Any]] = None
):
    """
    Apply a list of :class:`TransformGen` on the input image, and
    returns the transformed image and a list of transforms.

    We cannot simply create and return all transforms without
    applying it to the image, because a subsequent transform may
    need the output of the previous one.

    Args:
        transform_gens (list): list of :class:`TransformGen` instance to
            be applied.
        image (ndarray): uint8 or floating point images with 1 or 3 channels.

    Returns:
        ndarray: the transformed image
        TransformList: contain the transforms that's used.
    """
    for g in transform_gens:
        assert isinstance(g, TransformGen), g

    tfms = []
    for g in transform_gens:
        tfm = g.get_transform(image, annotations)
        assert isinstance(
            tfm, Transform
        ), 'TransformGen {} must return an instance of Transform! Got {} instead'.format(g, tfm)
        image = tfm.apply_image(image)
    return tfms
