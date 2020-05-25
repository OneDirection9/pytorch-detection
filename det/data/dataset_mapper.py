# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import copy
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from foundation.registry import Registry
from foundation.transforms import Transform

from . import transforms as T, utils
from .transforms import TransformGen

logger = logging.getLogger(__name__)


class DatasetMapperRegistry(Registry):
    """Registry of mappers."""
    pass


class DatasetMapper(object, metaclass=ABCMeta):
    """The base class of dataset mapper.

    A callable which takes an example and map it into a format used by downstream modules.
    :class:`MapDataset` uses the instance of it as argument, i.e. map_func.
    """

    def __init__(
        self,
        transforms: Optional[List[Union[Transform, TransformGen]]] = None,
        keypoint_hflip_indices: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            transforms: List of :class:`Transform` or :class`TransformGen`.
            keypoint_hflip_indices: A vector of size=#keypoints, storing the horizontally-flipped
                keypoint indices.
        """
        self.transforms = transforms
        self.keypoint_hflip_indices = keypoint_hflip_indices

    @abstractmethod
    def __call__(self, example: Dict[str, Any]) -> Optional[Any]:
        pass


@DatasetMapperRegistry.register('DictMapper')
class DictMapper(DatasetMapper):

    def __init__(
        self,
        image_format: Optional[str] = 'BGR',
        mask_format: Optional[str] = 'polygon',
        training: bool = True,
        transforms: Optional[List[Union[Transform, TransformGen]]] = None,
        keypoint_hflip_indices: Optional[np.ndarray] = None,
    ) -> None:
        super(DictMapper, self).__init__(transforms, keypoint_hflip_indices)

        if mask_format not in ('polygon', 'bitmask'):
            raise ValueError('mask_format should be polygon or bitmask. Got {}'.format(mask_format))

        # TODO: better logic to handle RandomCrop
        if transforms is not None:
            for t in transforms[1:]:
                if isinstance(t, T.RandomCrop):
                    raise ValueError('RandomCrop can only be used once and as the first transform')
            self.crop_gen = isinstance(transforms[0], T.RandomCrop)

        self.image_format = image_format
        self.mask_format = mask_format
        self.training = training

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        example = copy.deepcopy(example)  # deepcopy to avoid modifying raw data

        image = utils.read_image(example['file_name'], self.image_format)
        utils.check_image_size(example, image)

        if 'annotations' not in example:
            image, transforms = T.apply_transforms(self.transforms, image)
        else:
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.transforms[0],
                    image,
                    example['annotations'],
                )
                image = crop_tfm.apply_image(image)

        if self.transforms is not None:
            if 'annotations' in example and isinstance(self.transforms[0], T.RandomCrop):
                # Crop around an instance if there are instances in the image.
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.transforms[0],
                    image,
                    example['annotations'],
                )
                image = crop_tfm.apply_image(image)
                image, transforms = T.apply_transforms(self.transforms[1:], image)
                transforms = crop_tfm + transforms
            else:
                image, transforms = T.apply_transforms(self.transforms, image)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        example['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose((2, 0, 1))))

        if not self.training:
            example.pop('annotations', None)
            example.pop('sem_seg_file_name', None)
            return example

        if self.transforms is None:
            return example

        return example

    def __repr__(self):
        return self.transform_gens
