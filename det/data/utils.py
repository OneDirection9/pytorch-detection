from __future__ import absolute_import, division, print_function

import logging
from typing import List

from .datasets import VisionDataset

__all__ = ['check_metadata_consistency']

logger = logging.getLogger(__name__)


def check_metadata_consistency(name: str, datasets: List[VisionDataset]) -> None:
    """Checks that the datasets have consistent metadata.

    Args:
        name: A metadata key.
        datasets: List of datasets.

    Raises:
        AttributeError: If the `name` does not exist in the metadata
        ValueError: If the given datasets do not have the same metadata values defined by `name`.
    """
    if len(datasets) == 0:
        return

    entries_per_dataset = [ds.metadata.get(name) for ds in datasets]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'.".format(
                    name,
                    type(datasets[idx]).__name__, entry
                )
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'.".format(
                    name,
                    type(datasets[0]).__name__, entries_per_dataset[0]
                )
            )
            raise ValueError("Datasets have different '{}'!".format(name))
