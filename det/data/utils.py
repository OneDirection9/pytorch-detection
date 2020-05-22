# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import logging
from typing import Any, Dict, List

import numpy as np
from foundation.transforms import CropTransform

from ..structures import BoxMode
from .datasets import VisionDataset
from .transforms import RandomCrop

__all__ = [
    'check_metadata_consistency',
    'gen_crop_transform_with_instance',
    'create_keypoint_hflip_indices',
]

logger = logging.getLogger(__name__)


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

    Returns:

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
