from __future__ import absolute_import, division, print_function

import logging
from typing import List

from .datasets import VisionDataset

logger = logging.getLogger(__name__)


def check_metadata_consistency(name: str, datasets: List[VisionDataset]) -> None:
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
