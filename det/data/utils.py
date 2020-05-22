# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from foundation.transforms import CropTransform
from PIL import Image, ImageOps

from ..structures import BoxMode
from .datasets import VisionDataset
from .transforms import RandomCrop

__all__ = [
    'read_image',
    'check_image_size',
    'check_metadata_consistency',
    'gen_crop_transform_with_instance',
    'create_keypoint_hflip_indices',
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
    random_crop: RandomCrop, image: np.ndarray, annotations: List[Dict[str, Any]]
) -> CropTransform:
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
    return CropTransform(x0, y0, crop_size[1], crop_size[0])


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
