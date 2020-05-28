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
from PIL import Image

from . import transforms as T, utils

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
        transforms: Optional[List[Union[T.Transform, T.TransformGen]]] = None,
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
    """Reading images and transforming it alongside with annotations."""

    def __init__(
        self,
        transforms: List[Union[T.Transform, T.TransformGen]],
        image_format: Optional[str] = 'BGR',
        mask_format: Optional[str] = 'polygon',
        training: bool = True,
        keypoint_hflip_indices: Optional[np.ndarray] = None,
    ) -> None:
        super(DictMapper, self).__init__(transforms, keypoint_hflip_indices)

        if transforms is None:
            raise ValueError('None transforms available')
        if mask_format not in ('polygon', 'bitmask'):
            raise ValueError('mask_format should be polygon or bitmask. Got {}'.format(mask_format))

        # TODO: better logic to handle RandomCrop
        for t in transforms[1:]:
            if isinstance(t, T.RandomCrop):
                raise ValueError('RandomCrop can only be used as the first transform')
        self.use_crop = isinstance(transforms[0], T.RandomCrop)

        self.image_format = image_format
        self.mask_format = mask_format
        self.training = training

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        example = copy.deepcopy(example)  # deepcopy to avoid modifying raw data

        image = utils.read_image(example['file_name'], self.image_format)
        utils.check_image_size(example, image)

        if 'annotations' in example and self.use_crop:
            crop_tfm = utils.gen_crop_transform_with_instance(
                self.transforms[0], image, example['annotations']
            )
            image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transforms(self.transforms[1:], image)
            transforms = crop_tfm + transforms
        else:
            image, transforms = T.apply_transforms(self.transforms, image)

        image_shape = image.shape[:2]

        # PyTorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        example['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose((2, 0, 1))))

        if not self.training:
            # USER: Modify this if you want to keep them for some reason.
            example.pop('annotations', None)
            example.pop('sem_seg_file_name', None)
            return example

        if 'annotations' in example:
            anns = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices
                ) for obj in example.pop('annotations') if obj.get('iscrowd', 0) == 0
            ]
            instances = utils.annotations_to_instances(
                anns, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.use_crop and instances.has('gt_masks'):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            example['instances'] = utils.filter_empty_instances(instances)

        if 'sem_seg_file_name' in example:
            sem_seg_gt = np.asarray(Image.open(example['sem_seg_file_name'], 'rb'), dtype='uint8')
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype('long'))
            example['sem_seg'] = sem_seg_gt

        return example
