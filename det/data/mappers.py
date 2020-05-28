from __future__ import absolute_import, division, print_function

import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from foundation.registry import Registry
from PIL import Image
from pycocotools import mask as mask_util

from ..structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    polygons_to_bitmask,
)
from . import transforms as T, utils


class MapperRegistry(Registry):
    """Registry of mappers."""
    pass


Mapper = Callable


class MapperList(object):
    """Maintains a list of Mappers which will be applied in sequence.

    Attributes:
        mappers (list[Callable]):
    """

    def __init__(self, mappers: List[Callable]) -> None:
        """
        Args:
            mappers: List of callable objects which are executed one by one.
        """
        for ppl in mappers:
            if not isinstance(ppl, Callable):
                raise TypeError('Expected a callable object. Got {}'.format(type(ppl)))

        self.mappers = mappers

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for ppl in self.mappers:
            example = ppl(example)
            if example is None:
                return None
        return example

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for ppl in self.mappers:
            format_string += '\n'
            format_string += '    {0}'.format(ppl)
        format_string += '\n)'
        return format_string

    __str__ = __repr__


@MapperRegistry.register('ImageLoader')
class ImageLoader(object):

    def __init__(self, image_format='BGR'):
        self.image_format = image_format

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image = utils.read_image(example['file_name'], self.image_format)
        utils.check_image_size(example, image)
        example['image'] = image

        if 'sem_seg_file_name' in example:
            example['sem_seg'] = np.asarray(
                Image.open(example['sem_seg_file_name'], 'rb'), dtype='uint8'
            )
        return example


class TransformApply(object):
    """A wrapper that instantiate a transform with given arguments and apply it to an example."""

    def __init__(
        self,
        transform_cls: Type[Union[T.Transform, T.TransformGen]],
        *args: Any,
        random_apply: Optional[float] = None,
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        **kwargs: Any
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

    @classmethod
    def transform_segmentation_annotations(cls, segmentation, transforms, image_size):
        # each instance contains one or more polygons
        if isinstance(segmentation, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segmentation]
            segmentation = [p.reshape(-1) for p in transforms.apply_polygons(polygons)]
            return segmentation
        elif isinstance(segmentation, dict):
            # RLE
            if isinstance(segmentation['counts'], list):
                # Uncompressed RLE -> encoded RLE
                segmentation = mask_util.frPyObjects(
                    segmentation, segmentation['size'][0], segmentation['size'][1]
                )
            segmentation = mask_util.decode(segmentation)
            segmentation = transforms.apply_segmentation(segmentation)
            assert tuple(segmentation.shape[:2]) == image_size
            return segmentation
        elif isinstance(segmentation, np.ndarray):
            # masks
            segmentation = transforms.apply_segmentation(segmentation)
            assert tuple(segmentation.shape[:2]) == image_size
        else:
            raise TypeError('')
        return segmentation

    @classmethod
    def transform_keypoints_annotations(cls, keypoints, transforms, keypoint_hflip_indices=None):
        keypoints = np.asarray(keypoints, dtype='float64').reshape(-1, 3)
        keypoints[:, :2] = transforms.apply_coords(keypoints[:, :2])

        # This assumes that HFlipTransform is the only one that does flip
        if isinstance(transforms, T.TransformList):
            do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        else:
            do_hflip = isinstance(transforms, T.HFlipTransform)

        if do_hflip:
            assert keypoint_hflip_indices is not None
            keypoints = keypoints[keypoint_hflip_indices, :]

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        # TODO may need to reset visibility for cropped keypoints,
        # but it does not matter for our existing algorithms
        keypoints[keypoints[:, 2] == 0] = 0

        return keypoints

    @classmethod
    def transform_instance_annotations(
        cls, annotation, transform, image_size, *, keypoint_hflip_indices=None
    ):
        bbox = BoxMode.convert(annotation['bbox'], annotation['bbox_mode'], BoxMode.XYXY_ABS)
        # Note the bbox is 1d (per-instance bounding box)
        annotation['bbox'] = transform.apply_box(bbox)[0]
        annotation['bbox_mode'] = BoxMode.XYXY_ABS

        if 'segmentation' in annotation:
            annotation['segmentation'] = cls.transform_segmentation_annotations(
                annotation['segmentation'],
                transform,
                image_size,
            )

        if 'keypoints' in annotation:
            annotation['keypoints'] = cls.transform_keypoints_annotations(
                annotation['keypoints'], transform, keypoint_hflip_indices
            )

        return annotation

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if isinstance(self.transform, T.TransformGen):
            transform = self.transform.get_transform(example['image'])
        else:
            transform = self.transform
        example['image'] = transform.apply_image(example['image'])
        image_shape = example['image'].shape

        if 'annotations' in example:
            example['annotations'] = [
                self.transform_instance_annotations(
                    obj, transform, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                ) for obj in example.pop('annotations') if obj.get('iscrowd', 0) == 0
            ]

        if 'sem_seg' in example:
            example['sem_seg'] = transform.apply_segmentation(example['sem_seg'])

        return example

    def __repr__(self):
        return repr(self.transform)


MapperRegistry.register('ResizeShortestEdge')(
    functools.partial(TransformApply, T.ResizeShortestEdge)
)

MapperRegistry.register('RandomHFlip')(functools.partial(TransformApply, T.RandomHFlip))

# def transform_wrapper(
#     transform_cls: Type[Union[T.Transform, T.TransformGen]],
#     *args: Any,
#     random_apply: Optional[float] = None,
#     apply_cls: Type[object] = TransformApply,
#     **kwargs: Any,
# ) -> object:
#     """A wrapper that instantiate a transform with given arguments and apply it with given class.
#
#     Args:
#         transform_cls: Can be either `Transform` or `TransformGen`.
#         args: Positional arguments to instantiate the transform.
#         random_apply: If not None, wrap the transform by :class:`RandomApply`.
#         apply_cls: A class that take the transform to instantiate and implement :meth:`__call__`.
#         kwargs: Keyword arguments to instantiate the transform.
#
#     Returns:
#         Instance of apply_cls.
#     """
#     transform = transform_cls(*args, **kwargs)
#     if random_apply is not None:
#         transform = T.RandomApply(transform, random_apply)
#     return apply_cls(transform)


@MapperRegistry.register('ToInstances')
class ToInstances(object):

    def __init__(self, mask_format='polygon', use_crop=False):
        self.mask_format = mask_format
        self.use_crop = use_crop

    @classmethod
    def annotations_to_instances(cls, anns, image_size, mask_format):
        target = Instances(image_size)

        boxes = [BoxMode.convert(obj['bbox'], obj['bbox_mode'], BoxMode.XYXY_ABS) for obj in anns]
        boxes = target.gt_boxes = Boxes(boxes)
        boxes.clip(image_size)

        classes = [obj['category_id'] for obj in anns]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes

        if len(anns) and 'segmentation' in anns[0]:
            segms = [obj['segmentation'] for obj in anns]
            if mask_format == 'polygon':
                masks = PolygonMasks(segms)
            elif mask_format == 'bitmask':
                masks = []
                for segm in segms:
                    if isinstance(segm, list):
                        # polygon
                        masks.append(polygons_to_bitmask(segm, *image_size))
                    elif isinstance(segm, dict):
                        # RLE
                        masks.append(mask_util.decode(segm))
                    elif isinstance(segm, dict):
                        assert segm.ndim == 2, 'Expect segmentation of 2 dimensions. got {}'.format(
                            segm.ndim
                        )
                        masks.append(segm)
                    else:
                        raise ValueError(
                            "Cannot convert segmentation of type '{}' to BitMasks!"
                            'Supported types are: polygons as list[list[float] or ndarray],'
                            ' COCO-style RLE as a dict, or a full-image segmentation mask '
                            'as a 2D ndarray.'.format(type(segm))
                        )
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
                )
            else:
                raise ValueError('Expect polygon or bitmask. Got {}'.format(mask_format))

            target.gt_masks = masks

        if len(anns) and 'keypoints' in anns[0]:
            target.gt_keypoints = Keypoints([obj.get('keypoints', []) for obj in anns])

        return target

    @classmethod
    def filter_empty_instances(
        cls,
        instances: Instances,
        by_box: bool = True,
        by_mask: bool = True,
        box_threshold: float = 1e-5
    ) -> Instances:
        """Filters out empty instances in an `Instances` object.

        Args:
            instances:
            by_box: Whether to filter out instances with empty boxes
            by_mask: Whether to filter out instances with empty masks
            box_threshold: Minimum width and height to be considered non-empty

        Returns:
            Instances: The filtered instances.
        """
        assert by_box or by_mask
        r = []
        if by_box:
            r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
        if instances.has('gt_masks') and by_mask:
            r.append(instances.gt_masks.nonempty())

        # TODO: can also filter visible keypoints

        if not r:
            return instances
        m = r[0]
        for x in r[1:]:
            m = m & x
        return instances[m]

    def __call__(self, example):
        image_size = example['image'].shape[:2]

        # PyTorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        example['image'] = torch.as_tensor(
            np.ascontiguousarray(example['image'].transpose((2, 0, 1)))
        )

        if 'annotations' in example:
            instances = self.annotations_to_instances(
                example.pop('annotations'), image_size, self.mask_format
            )
            if self.use_crop and instances.has('gt_masks'):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            example['instances'] = self.filter_empty_instances(instances)

        if 'sem_seg' in example:
            example['sem_seg'] = torch.as_tensor(example['sem_seg'].astype('long'))
        return example
