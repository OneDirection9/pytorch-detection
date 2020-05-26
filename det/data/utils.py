# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
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
from .datasets import VisionDataset

__all__ = [
    'read_image',
    'check_image_size',
    'check_metadata_consistency',
    'gen_crop_transform_with_instance',
    'create_keypoint_hflip_indices',
    'transform_instance_annotations',
    'annotations_to_instances',
    'filter_empty_instances',
]

logger = logging.getLogger(__name__)


def read_image(file_name: str, format: Optional[str] = None) -> np.ndarray:
    """Read an image into the given format.

    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name: Image file path
        format: One of the supported image modes in PIL, or 'BGR'.

    Returns:
        image: An HWC image in the given format, which is 0-255, uint8 for supported image modes in
            PIL or 'BGR'.
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


def check_image_size(example: Dict[str, Any], image: np.ndarray) -> None:
    """Checks image size between loaded array and annotation."""
    h, w = image.shape[:2]

    if 'width' in example or 'height' in example:
        image_wh = (w, h)
        expected_wh = (example['width'], example['height'])
        if not image_wh == expected_wh:
            raise ValueError(
                'Mismatched (W, H){}. Got {}, expect {}'.format(
                    ' for image' + example['file_name'] if 'file_name' in example else '', image_wh,
                    expected_wh
                )
            )

    if 'width' not in example:
        example['width'] = w
    if 'height' not in example:
        example['height'] = h


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


def gen_crop_transform_with_instance(
    random_crop: T.RandomCrop, image: np.ndarray, annotations: List[Dict[str, Any]]
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


def transform_instance_annotations(
    annotation: Dict[str, Any],
    transforms: T.TransformList,
    image_size: Tuple[int, int],
    *,
    keypoint_hflip_indices: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Applies transforms to box, segmentation and keypoints annotations of a single instance.

    It will use :meth:`transforms.apply_box` for the box, :meth:`transforms.apply_segmentation` for
    segmentation RLE, and :meth:`transforms.apply_coords` for segmentation polygons & keypoints.


    Args:
        annotation: Dictionary of instance annotations for a single instance. It will be modified
            in-palce.
        transforms: :class:`TransformList`.
        image_size: The height and width of the transformed image
        keypoint_hflip_indices: see :func:`create_keypoint_hflip_indices`.

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

    if 'keypoints' in annotation:
        # (Nx3,) -> (N, 3)
        keypoints = np.asarray(annotation['keypoints'], dtype='float64').reshape(-1, 3)
        keypoints[:, :2] = transforms.apply_coords(keypoints[:, :2])

        # This assumes that HFlipTransform is the only one taht does flip
        do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1
        if do_hflip:
            assert keypoint_hflip_indices is not None
            keypoints = keypoints[keypoint_hflip_indices, :]

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        # TODO may need to reset visibility for cropped keypoints,
        # but it does not matter for our existing algorithms
        keypoints[keypoints[:, 2] == 0] = 0
        annotation['keypoints'] = keypoints

    return annotation


def annotations_to_instances(
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


def filter_empty_instances(
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
