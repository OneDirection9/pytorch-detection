# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
"""
This file contains the default mapping that's applied to dataset dict.
"""
from __future__ import absolute_import, division, print_function

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from foundation.registry import Registry
from foundation.transforms import Transform

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
        image_format=1,
        mask_on=1,
        mask_format=1,
        keypoint_on=1,
        transforms: Optional[List[Union[Transform, TransformGen]]] = None,
        keypoint_hflip_indices: Optional[np.ndarray] = None,
    ) -> None:
        super(DictMapper, self).__init__(transforms, keypoint_hflip_indices)

        self.image_format = image_format
        self.mask_on = mask_on
        self.mask_format = mask_format
        self.keypoint_on = keypoint_on

    def __call__(self, example: Dict[str, Any]) -> Optional[Any]:
        pass

    def __repr__(self):
        return self.transform_gens
