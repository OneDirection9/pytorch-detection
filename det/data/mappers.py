from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from foundation.registry import Registry
from PIL import Image, ImageOps
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
from . import transforms as T

__all__ = [
    'MapperRegistry',
    'Mapper',
    'MapperList',
    'ImageLoader',
    'ToInstances',
]


class MapperRegistry(Registry):
    """Registry of mappers.

    Similar to pipeline, a mapper is also used to process a single example. But the mappers are
    on-the-fly and may with randomness.
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

    @classmethod
    def read_image(cls, file_name: str, format: Optional[str] = None) -> np.ndarray:
        """Reads an image into the given format.

        Will apply rotation and flipping if the image has such exif information.

        Args:
            file_name: Image file path
            format: One of the supported image modes in PIL, or 'BGR'.

        Returns:
            image: An HWC image in the given format, which is 0-255, uint8 for supported image modes
                in PIL or 'BGR'.
        """
        image = Image.open(file_name)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == 'BGR':
                conversion_format = 'RGB'
            image = image.convert(conversion_format)
        image = np.asarray(image)
        # PIL squeezes out the channel dimension for 'L', so make it HWC
        if format == 'L':
            image = np.expand_dims(image, -1)

        # Handle formats not supported by PIL
        if format == 'BGR':
            # flip channels if needed
            image = image[:, :, ::-1]
        return image

    @classmethod
    def check_image_size(cls, example: Dict[str, Any], image: np.ndarray) -> None:
        """Checks image size between loaded array and annotation."""
        h, w = image.shape[:2]

        if 'width' in example or 'height' in example:
            image_wh = (w, h)
            expected_wh = (example['width'], example['height'])
            if not image_wh == expected_wh:
                raise ValueError(
                    'Mismatched (W, H){}. Got {}, expect {}'.format(
                        ' for image' + example['file_name'] if 'file_name' in example else '',
                        image_wh, expected_wh
                    )
                )

        if 'width' not in example:
            example['width'] = w
        if 'height' not in example:
            example['height'] = h

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        image = self.read_image(example['file_name'], self.image_format)
        self.check_image_size(example, image)
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

    @classmethod
    def annotations_to_instances(
        cls,
        anns: List[Dict[str, Any]],
        image_size: Tuple[int, int],
        mask_format: str = 'polygon'
    ) -> Instances:
        """Creates an :class:`Instances` object ysed by the models from instance annotations.

        Args:
            anns: List of instance annotations in one image, each element for one instance.
            image_size: The height and width of image.
            mask_format: Mask format.

        Returns:
            Instances:
                It will contain fields "gt_boxes", "gt_classes", "gt_masks", "gt_keypoints", if they can
                be obtained from `anns`. This is the format that builtin models expect.
        """
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
                    elif isinstance(segm, np.ndarray):
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

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
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
    def transform_instance_annotations(
        cls,
        annotation: Dict[str, Any],
        transforms: Union[T.Transform, T.TransformList],
        image_size: Tuple[int, int],
        *,
        keypoint_hflip_indices: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Applies transforms to box, segmentation and keypoints annotations of a single instance.

        It will use :meth:`transforms.apply_box` for the box, :meth:`transforms.apply_segmentation`
        for segmentation RLE, and :meth:`transforms.apply_coords` for segmentation polygons &
        keypoints.

        Args:
            annotation: Dictionary of instance annotations for a single instance. It will be
                modified in-place.
            transforms: :class:`TransformList` or :class:`Transform`.
            image_size: The height and width of the transformed image
            keypoint_hflip_indices: A vector of size=#keypoints, storing the horizontally-flipped
                keypoint indices.

        Returns:
            dict: The same input dict with fields 'bbox', 'segmentation', 'keypoints' transformed
                according to transforms. The 'bbox_mode' field will be set to XYXY_ABS.
        """
        bbox = BoxMode.convert(annotation['bbox'], annotation['bbox_mode'], BoxMode.XYXY_ABS)
        # Note the bbox is 1d (per-instance bounding box)
        annotation['bbox'] = transforms.apply_box(bbox)[0]
        annotation['bbox_mode'] = BoxMode.XYXY_ABS

        if 'segmentation' in annotation:
            # each instance contains one or more polygons
            segm = annotation['segmentation']
            if isinstance(segm, list):
                # polygons
                polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                annotation['segmentation'] = [
                    p.reshape(-1) for p in transforms.apply_polygons(polygons)
                ]
            elif isinstance(segm, dict):
                # RLE
                if isinstance(segm['counts'], list):
                    # Uncompressed RLE -> encoded RLE
                    segm = mask_util.frPyObjects(segm, segm['size'][0], segm['size'][1])
                mask = mask_util.decode(segm)
                mask = transforms.apply_segmentation(mask)
                assert tuple(mask.shape[:2]) == image_size
                annotation['segmentation'] = mask
            elif isinstance(segm, np.ndarray):
                # masks
                mask = transforms.apply_segmentation(segm)
                assert tuple(mask.shape[:2]) == image_size
                annotation['segmentation'] = mask
            else:
                raise TypeError(
                    "Cannot transform segmentation of type '{}'!"
                    'Supported types are: polygons as list[list[float] or ndarray],'
                    ' COCO-style RLE as a dict, 2D ndarray.'.format(type(segm))
                )

        if 'keypoints' in annotation:
            # (Nx3,) -> (N, 3)
            keypoints = np.asarray(annotation['keypoints'], dtype='float64').reshape(-1, 3)
            keypoints[:, :2] = transforms.apply_coords(keypoints[:, :2])

            # This assumes that HFlipTransform is the only one that does flip
            if isinstance(transforms, T.TransformList):
                do_hflip = sum(
                    isinstance(t, T.HFlipTransform) for t in transforms.transforms
                ) % 2 == 1
            else:
                do_hflip = isinstance(transforms, T.HFlipTransform)

            if do_hflip:
                assert keypoint_hflip_indices is not None
                keypoints = keypoints[keypoint_hflip_indices, :]

            # Maintain COCO convention that if visibility == 0, then x, y = 0
            # TODO may need to reset visibility for cropped keypoints,
            # but it does not matter for our existing algorithms
            keypoints[keypoints[:, 2] == 0] = 0
            annotation['keypoints'] = keypoints

        return annotation

    @classmethod
    def gen_crop_transform_with_instance(
        cls, random_crop: T.RandomCrop, image: np.ndarray, annotations: List[Dict[str, Any]]
    ) -> T.CropTransform:
        """Generates a :class:`CropTransform` so that the cropping region contains the center of the
        given instance.

        Args:
            random_crop: :class:`RandomCrop` instance.
            image:
            annotations: Annotations in the format of a list of dictionaries.
        """
        image_size = image.shape[:2]
        crop_size = random_crop.get_crop_size(image_size)
        crop_size = np.asarray(crop_size, dtype=np.int32)

        ann = np.random.choice(annotations)
        crop_size = np.asarray(crop_size, dtype=np.int32)
        bbox = BoxMode.convert(ann['bbox'], ann['bbox_mode'], BoxMode.XYXY_ABS)
        center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
        assert (
            image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
        ), 'The annotation bounding box is outside of the image!'
        assert (
            image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
        ), 'Crop size is larger than image size!'

        min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
        max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
        max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

        y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
        x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
        return T.CropTransform(x0, y0, crop_size[1], crop_size[0])

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(self.transform, T.RandomCrop) and 'annotations' in example:
            # Crop around an instance if there are instances in the image.
            transform = self.gen_crop_transform_with_instance(
                self.transform,
                example['image'],
                example['annotations'],
            )
        elif isinstance(self.transform, T.TransformGen):
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

    def __repr__(self) -> str:
        return 'Apply(transform={0})'.format(self.transform)


MapperRegistry.register_partial('ResizeShortestEdge', TransformApply, T.ResizeShortestEdge)
MapperRegistry.register_partial('RandomHFlip', TransformApply, T.RandomHFlip)
