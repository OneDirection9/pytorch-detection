# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import logging
from typing import List

import numpy as np

from .datasets import VisionDataset

__all__ = [
    'check_metadata_consistency',
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
