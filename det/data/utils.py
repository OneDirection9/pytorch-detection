# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from pycocotools import mask as mask_util

from det.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    polygons_to_bitmask,
)
from . import transforms as T
from .datasets import VisionDataset

__all__ = [
    'convert_pil_to_numpy',
    'convert_image_to_rgb',
    'read_image',
    'check_image_size',
    'transform_instance_annotations',
    'gen_crop_transform_with_instance',
    'annotations_to_instances',
    'filter_empty_instances',
    'check_metadata_consistency',
    'create_keypoint_hflip_indices',
]

logger = logging.getLogger(__name__)

# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]


def convert_pil_to_numpy(image: Image, format: str) -> np.ndarray:
    """Converts PIL image to numpy array of target format

    Args:
        image: The PIL image.
        format: The format of output image. See :func:`read_image`.

    Returns:
        See :func:`read_image`.
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ['BGR', 'YUV-BT.601']:
            conversion_format = 'RGB'
        image = image.convert(conversion_format)

    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == 'L':
        image = np.expand_dims(image, -1)
    # handle formats not supported by PIL
    elif format == 'BGR':
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == 'YUV-BT.601':
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def convert_image_to_rgb(image: Union[np.ndarray, torch.Tensor], format: str) -> np.ndarray:
    """Converts an image from given format to RGB.

    Args:
        image: An HxWxC image.
        format: The format of input image, see :func:`read_image`.

    Returns:
        HxWx3 RGB image in 0-255 range, can be either float or uint8.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if format == 'BGR':
        image = image[:, :, [2, 1, 0]]
    elif format == 'YUV-BT.601':
        image = image.dot(image, np.array(_M_YUV2RGB).T)
        image = image * 255.0
    else:
        if format == 'L':
            image = image[:, :, 0]
        image = image.astype(np.uint8)
        image = np.asarray(Image.fromarray(image, mode=format).convert('RGB'))
    return image


def read_image(file_name: str, format: str = None) -> np.ndarray:
    """Reads an image into the given format.

    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name: Image file path.
        format: One the supported image modes in PIL, or "BGR", or "YUV-BT.601".

    Returns:
        An HxWxC image in the given format, which is 0-255, uint8 for supported image modes in PIL
        or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    return convert_pil_to_numpy(image, format)


def check_image_size(dataset_dict: Dict[str, Any], image: np.ndarray) -> None:
    """Checks image size between loaded image and annotation."""
    h, w = image.shape[:2]

    if 'width' in dataset_dict or 'height' in dataset_dict:
        image_wh = (w, h)
        expected_wh = (dataset_dict['width'], dataset_dict['height'])
        if not image_wh == expected_wh:
            raise ValueError(
                'Mismatched (W, H){}. Got {}, expect {}'.format(
                    ' for image' + dataset_dict['file_name'] if 'file_name' in dataset_dict else '',
                    image_wh, expected_wh
                )
            )

    if 'width' not in dataset_dict:
        dataset_dict['width'] = w
    if 'height' not in dataset_dict:
        dataset_dict['height'] = h


def transform_instance_annotations(
    annotation: Dict[str, Any],
    transforms: Union[T.Transform, T.TransformList],
    image_size: Tuple[int, int],
    *,
    keypoint_hflip_indices: Optional[np.ndarray] = None,
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
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation['bbox'], annotation['bbox_mode'], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(bbox)[0].clip(min=0)
    annotation['bbox'] = np.minimum(bbox, list(image_size + image_size)[::-1])
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
        keypoints_xy = transforms.apply_coords(keypoints[:, :2])

        # Set all out-of-boundary points to "unlabeled"
        inside = (keypoints_xy >= np.array([0, 0])) & (keypoints_xy <= np.array(image_size[::-1]))
        inside = inside.all(axis=1)
        keypoints[:, :2] = keypoints_xy
        keypoints[:, 2][~inside] = 0

        # This assumes that HFlipTransform is the only one that does flip
        if isinstance(transforms, T.TransformList):
            do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        else:
            do_hflip = isinstance(transforms, T.HFlipTransform)

        # If flipped, swap each keypoint with its opposite-handed equivalent
        if do_hflip:
            assert keypoint_hflip_indices is not None
            keypoints = keypoints[keypoint_hflip_indices, :]

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        keypoints[keypoints[:, 2] == 0] = 0
        annotation['keypoints'] = keypoints

    return annotation


def annotations_to_instances(
    anns: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    mask_format: str = 'polygon',
) -> Instances:
    """Creates an :class:`Instances` object used by the models from instance annotations.

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
    target.gt_boxes = Boxes(boxes)

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


def filter_empty_instances(
    instances: Instances,
    by_box: bool = True,
    by_mask: bool = True,
    box_threshold: float = 1e-5,
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


def gen_crop_transform_with_instance(
    crop_size: Tuple[int, int], image_size: Tuple[int, int], instance: Dict[str, Any]
) -> T.CropTransform:
    """Generates a :class:`CropTransform` so that the cropping region contains the center of the
    given instance.

    Args:
        crop_size: Height and width of cropped size.
        image_size: Height and width of image.
        instance: An annotation dict of one instance, in Detectron2's dataset format.
    """
    crop_size = np.asarray(crop_size, dtype=np.int32)
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), 'Crop size is larger than image size!'

    bbox = BoxMode.convert(instance['bbox'], instance['bbox_mode'], BoxMode.XYXY_ABS)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), 'The annotation bounding box is outside of the image!'

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y1 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x1 = np.random.randint(min_yx[1], max_yx[1] + 1)
    return T.CropTransform(x1, y1, crop_size[1], crop_size[0])


def check_metadata_consistency(name: str, vision_datasets: List[VisionDataset]) -> None:
    """Checks that the datasets have consistent metadata.

    Args:
        name: A metadata key.
        vision_datasets: List of datasets.

    Raises:
        AttributeError: If the `name` does not exist in the metadata
        ValueError: If the given datasets do not have the same metadata values defined by `name`.
    """
    if len(vision_datasets) == 0:
        raise ValueError('None vision dataset is available')

    entries_per_dataset = [ds.metadata.get(name) for ds in vision_datasets]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'.".format(
                    name,
                    type(vision_datasets[idx]).__name__, entry
                )
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'.".format(
                    name,
                    type(vision_datasets[0]).__name__, entries_per_dataset[0]
                )
            )
            raise ValueError("Datasets have different '{}'!".format(name))


def create_keypoint_hflip_indices(vision_datasets: List[VisionDataset]) -> np.ndarray:
    """
    Args:
        vision_datasets: List of datasets.

    Returns:
        np.ndarray[int]: A vector of size=#keypoints, storing the horizontally-flipped keypoint
            indices.
    """
    check_metadata_consistency('keypoint_names', vision_datasets)
    check_metadata_consistency('keypoint_flip_map', vision_datasets)

    metadata = vision_datasets[0].metadata
    names = metadata.get('keypoint_names')
    hflip_map = dict(metadata.get('keypoint_flip_map'))
    hflip_map.update({v: k for k, v in hflip_map.items()})

    flipped_names = [i if i not in hflip_map else hflip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return np.asarray(flip_indices)
