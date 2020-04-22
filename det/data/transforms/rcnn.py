from __future__ import absolute_import, division, print_function

import random

import cv2
import numpy as np

from foundation import transforms as tfr
from foundation.backends import torch as fdn_pt
from ..registry import TransformStash

__all__ = ['FasterRCNNTrainTransform', 'FasterRCNNValTransform']


@TransformStash.register('FasterRCNNTrainTransform')
class FasterRCNNTrainTransform(object):
    """Default Faster-RCNN training transform.

    Args:
        short (int, tuple[2], optional): Resize image shorter edge to `short`, or resize
            randomly within given range. Default: 600
        max_size (int, optional): The threshold that the longer size of new image should
            not great than. Default: 1000
        size_divisor (int, optional): Height and width of new image are rounded to
            multiple of it. Default: 32
        flip_p (float, optional): Horizontal flip probability [0, 1]. Default: 0.5
        mean (tuple[3], optional): Mean values used for input normalization. Default
            values are the mean pixel value from ImageNet in *R G B* order. Default:
            (123.675, 116.28, 103.53)
        std (tuple[3], optional): Std values used for input normalization. Default values
            are std pixel value from ImageNet in *R G B* order. Default:
            (58.395, 57.12, 57.375)
    """

    def __init__(
        self,
        short=600,
        max_size=1000,
        size_divisor=32,
        flip_p=0.5,
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375)
    ):
        self._short = short
        self._max_size = max_size
        self._size_divisor = size_divisor
        self._flip_p = flip_p
        self._mean = np.array(mean, dtype=np.float32)
        self._std = np.array(std, dtype=np.float32)

    def __call__(self, item):
        """Applies transform to image and target.

        Args:
            item (dict): Image information and annotations.

        Returns:
            Transformed item.
        """
        ori_h, ori_w = item['ori_shape']
        image, bboxes = item.pop('image'), item.pop('bboxes')

        # Resize shorter edge
        if isinstance(self._short, (tuple, list)):
            short = random.randint(self._short[0], self._short[1])
        else:
            short = self._short

        image = tfr.image.resize_short_within(
            image, short, self._max_size, self._size_divisor, cv2.INTER_LINEAR
        )
        h, w, _ = image.shape
        bboxes = tfr.bbox.resize(bboxes, (ori_h, ori_w), (h, w))

        # Random horizontal flip
        image, flipped = tfr.image.random_flip(image, px=self._flip_p, py=0)
        bboxes = tfr.bbox.flip(bboxes, (h, w), flip_x=flipped[0])

        # Normalize image
        image = (image - self._mean) / self._std
        # [H, W, C] -> [C, H, W]
        image = np.ascontiguousarray(image.transpose(2, 0, 1))

        item['image'] = image
        item['bboxes'] = bboxes

        # Convert to tensor
        item = fdn_pt.utils.to_tensor(item)

        return item


@TransformStash.register('FasterRCNNValTransform')
class FasterRCNNValTransform(object):
    """Default Faster-RCNN validation transform.

    Args:
        short (int, optional): Resize image shorter edge to `short`. Default: 600
        max_size (int, optional): The threshold that the longer size of new image should
            not great than. Default: 1000
        size_divisor (int, optional): Height and width of new image are rounded to
            multiple of it. Default: 32
        mean (tuple[3], optional): Mean values used for input normalization. Default:
            (123.675, 116.28, 103.53)
        std (tuple[3], optional): Std values used for input normalization. Default:
            (58.395, 57.12, 57.375)
    """

    def __init__(
        self,
        short=600,
        max_size=1000,
        size_divisor=32,
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375)
    ):
        self._short = short
        self._max_size = max_size
        self._size_divisor = size_divisor
        self._mean = mean
        self._std = std

    def __call__(self, item):
        """Applies transform to image and target.

        Args:
            item (dict): Image information and annotations, the annotations are optional.

        Returns:
            Transformed item.
        """
        ori_h, ori_w = item['ori_shape']
        image, bboxes = item.pop('image'), item.pop('bboxes', None)

        # Resize shorter edge
        image = tfr.image.resize_short_within(
            image, self._short, self._max_size, self._size_divisor, cv2.INTER_LINEAR
        )
        h, w, _ = image.shape
        if bboxes is not None:
            bboxes = tfr.bbox.resize(bboxes, (ori_h, ori_w), (h, w))

        # Normalize image
        image = (image - self._mean) / self._std
        # [H, W, C] -> [C, H, W]
        image = np.ascontiguousarray(image.transpose(2, 0, 1))

        item['image'] = image
        if bboxes is not None:
            item['bboxes'] = bboxes

        # Convert to tensor
        item = fdn_pt.utils.to_tensor(item)

        return item
