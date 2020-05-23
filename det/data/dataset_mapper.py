# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import copy
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
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

        self.image_format = image_format
        self.mask_format = mask_format
        self.training = training

    def __call__(self, example: Dict[str, Any]) -> Optional[Any]:
        example = copy.deepcopy(example)  # deepcopy to avoid modifying raw data

        image = utils.read_image(example['file_name'], self.image_format)
        utils.check_image_size(example, image)

        if 'annotations' in example:
            # Crop around an instance if there are instances in the image.
            new_transforms = []
            for tfm in self.transforms:
                if isinstance(tfm, T.RandomCrop):
                    new_transforms.append(
                        utils.gen_crop_transform_with_instance(tfm, image, example['annotations'])
                    )
                else:
                    new_transforms.append(tfm)
            image, transforms = T.apply_transforms(new_transforms, image)
        else:
            image, transforms = T.apply_transforms(self.transforms, image)

        from ipdb import set_trace
        set_trace()

        return example

    def __repr__(self):
        return self.transform_gens
