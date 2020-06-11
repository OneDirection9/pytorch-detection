# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from foundation.registry import Registry
from PIL import Image

from . import transforms as T, utils

__all__ = [
    'MapperRegistry',
    'Mapper',
    'MapperList',
    'ImageLoader',
    'ToInstances',
    'TransformApply',
]


class MapperRegistry(Registry):
    """Registry of mappers.

    Similar to pipeline, a mapper is used to process a single example on-the-fly and may with
    randomness.
    """
    pass


Mapper = Callable


class MapperList(object):
    """Maintains a list of mappers which will be applied in sequence.

    Attributes:
        mappers (list[Mapper]):
    """

    def __init__(self, mappers: List[Mapper]) -> None:
        """
        Args:
            mappers: List of mappers which are executed one by one.
        """
        for m in mappers:
            if not isinstance(m, Mapper):
                raise TypeError('Expected a Mapper. Got {}'.format(type(m)))

        self.mappers = mappers

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for m in self.mappers:
            example = m(example)
            if example is None:
                return None
        return example

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for m in self.mappers:
            format_string += '\n'
            format_string += '    {0}'.format(m)
        format_string += '\n)'
        return format_string


@MapperRegistry.register('ImageLoader')
class ImageLoader(object):
    """Loading image from file path."""

    def __init__(self, image_format: str = 'BGR') -> None:
        self.image_format = image_format

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        image = utils.read_image(example['file_name'], self.image_format)
        utils.check_image_size(example, image)
        example['image'] = image

        if 'sem_seg_file_name' in example:
            example['sem_seg'] = np.asarray(Image.open(example['sem_seg_file_name']), dtype='uint8')

        return example

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(image_format={0})'.format(self.image_format)


@MapperRegistry.register('ToInstances')
class ToInstances(object):
    """Converting example to format acceptable by downstream modules."""

    def __init__(self, mask_format: str = 'polygon', use_crop: bool = False) -> None:
        """
        Args:
            mask_format: Mask format.
            use_crop: Whether used :class:`RandomCrop` to annotations.
        """
        self.mask_format = mask_format
        self.use_crop = use_crop

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        image_size = example['image'].shape[:2]

        # PyTorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        example['image'] = torch.as_tensor(
            np.ascontiguousarray(example['image'].transpose((2, 0, 1)))
        )

        if 'annotations' in example:
            instances = utils.annotations_to_instances(
                example.pop('annotations'), image_size, self.mask_format
            )
            if self.use_crop and instances.has('gt_masks'):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            example['instances'] = utils.filter_empty_instances(instances)

        if 'sem_seg' in example:
            example['sem_seg'] = torch.as_tensor(example['sem_seg'].astype('long'))
        return example

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(mask_format={0}, use_crop={1})'.format(
            self.mask_format, self.use_crop
        )


class TransformApply(object):
    """A wrapper that instantiate a transform with given arguments and apply it to an example."""

    def __init__(
        self,
        transform_cls: Type[Union[T.Transform, T.TransformGen]],
        *args: Any,
        random_apply: Optional[float] = None,
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            transform_cls: Can be either `Transform` or `TransformGen`.
            random_apply: If not None, wrap the transform by :class:`RandomApply`.
            keypoint_hflip_indices: A vector of size=#keypoints, storing the horizontally-flipped
                keypoint indices.
            args: Positional arguments to instantiate the transform.
            kwargs: Keyword arguments to instantiate the transform.
        """
        transform = transform_cls(*args, **kwargs)
        if random_apply is not None:
            transform = T.RandomApply(transform, random_apply)

        self.transform = transform
        self.keypoint_hflip_indices = keypoint_hflip_indices

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(self.transform, T.RandomCrop) and 'annotations' in example:
            # Crop around an instance if there are instances in the image.
            transform = utils.gen_crop_transform_with_instance(
                self.transform,
                example['image'],
                example['annotations'],
            )
        elif isinstance(self.transform, T.TransformGen):
            transform = self.transform.get_transform(example['image'])
        else:
            transform = self.transform

        example['image'] = transform.apply_image(example['image'])
        image_shape = example['image'].shape[:2]  # H, W

        if 'annotations' in example:
            example['annotations'] = [
                utils.transform_instance_annotations(
                    obj, transform, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                ) for obj in example.pop('annotations') if obj.get('iscrowd', 0) == 0
            ]

        if 'sem_seg' in example:
            example['sem_seg'] = transform.apply_segmentation(example['sem_seg'])

        return example

    def __repr__(self) -> str:
        return 'Apply(transform={0})'.format(self.transform)


# Register wrappers presetting transform_cls
MapperRegistry.register_partial('ResizeShortestEdge', T.ResizeShortestEdge)(TransformApply)
MapperRegistry.register_partial('RandomHFlip', T.RandomHFlip)(TransformApply)
