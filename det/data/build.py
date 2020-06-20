# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import itertools
import logging
import operator
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from tabulate import tabulate
from termcolor import colored
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from det.utils.comm import get_world_size
from det.utils.env import seed_all_rng
from . import utils
from .common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from .dataset_mapper import DatasetMapper
from .datasets import VisionDataset, build_vision_datasets
from .samplers import InferenceSampler, TrainingSampler

logger = logging.getLogger(__name__)

__all__ = [
    'print_instances_class_histogram',
    'get_detection_dataset_dicts',
    'build_batch_data_loader',
    'build_detection_train_loader',
    'build_detection_test_loader',
]


class Processing(object):
    """Filtering out images with only crowd annotations and too few number of keypoints."""

    def __init__(self, filter_empty: bool = True, min_keypoints: int = 1) -> None:
        """
        Args:
            filter_empty: Whether to filter out images without instance annotations.
            min_keypoints: Filter out images with few keypoints than `min_keypoints`. Set to 0 to do
                nothing.
        """
        self.filter_empty = filter_empty
        self.min_keypoints = min_keypoints

    @staticmethod
    def filter_images_with_only_crowd_annotations(
        dataset_dicts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filters out images with none annotations or only crowd annotations.

        Filter out images without non-crowd annotations. A common training-time pre-processing on
        COCO dataset.

        Args:
            dataset_dicts: Annotations in Detectron2's Dataset format.

        Returns:
            The same format, but filtered.
        """
        num_before = len(dataset_dicts)

        def is_valid(anns: List[Dict[str, Any]]) -> bool:
            for ann in anns:
                if ann.get('iscrowd', 0) == 0:
                    return True
            return False

        dataset_dicts = [x for x in dataset_dicts if is_valid(x['annotations'])]
        num_after = len(dataset_dicts)
        logger.info(
            'Removed {} images with no usable annotations. {} images left.'.format(
                num_before - num_after, num_after
            )
        )
        return dataset_dicts

    @staticmethod
    def filter_images_with_few_keypoints(
        dataset_dicts: List[Dict[str, Any]], min_keypoints: int = 0
    ) -> List[Dict[str, Any]]:
        """Filters out images with too few number of keypoints.

        Args:
            dataset_dicts: Annotations in Detectron2's Dataset format.
            min_keypoints: See :class:`Processing`.

        Returns:
            The same format, but filtered.
        """
        num_before = len(dataset_dicts)

        def visible_keypoints_in_image(anns: List[Dict[str, Any]]) -> int:
            return sum(
                (np.array(ann['keypoints'][2::3]) > 0).sum() for ann in anns if 'keypoints' in ann
            )

        dataset_dicts = [
            x for x in dataset_dicts if visible_keypoints_in_image(x['annotations']) > min_keypoints
        ]
        num_after = len(dataset_dicts)
        logger.info(
            'Removed {} images with fewer than {} keypoints. {} images left.'.format(
                num_before - num_after, min_keypoints, num_after
            )
        )
        return dataset_dicts

    def __call__(self, dataset_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        has_instances = 'annotations' in dataset_dicts[0]

        if self.filter_empty and has_instances and 'sem_seg_file_name' not in dataset_dicts:
            dataset_dicts = self.filter_images_with_only_crowd_annotations(dataset_dicts)

        if self.min_keypoints > 0 and has_instances:
            dataset_dicts = self.filter_images_with_few_keypoints(dataset_dicts, self.min_keypoints)

        return dataset_dicts

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(filter_empty={0}, min_keypoints={1})'.format(
            self.filter_empty, self.min_keypoints
        )


def print_instances_class_histogram(
    dataset_dicts: List[Dict[str, Any]], class_names: List[str]
) -> None:
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry['annotations']
        classes = [x['category_id'] for x in annos if not x.get('iscrowd', 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + '..'
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(['total', total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=['category', '#instances'] * (N_COLS // 2),
        tablefmt='pipe',
        numalign='left',
        stralign='center',
    )
    logger.info(
        'Distribution of instances among all {} categories:\n'.format(num_classes) +
        colored(table, 'cyan')
    )


def get_detection_dataset_dicts(
    datasets: List[VisionDataset],
    processing_fn: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Loads and processes data for instance detection/segmentation and semantic segmentation.

    Args:
        datasets: List of vision datasets.
        processing_fn: A callable that takes dataset dicts as input and returns processed ones.

    Returns:
        List of dataset dicts.
    """
    dataset_dicts = [ds.get_items() for ds in datasets]
    for dicts, ds in zip(dataset_dicts, datasets):
        assert len(dicts), "Datasets '{}' is empty!\n{}".format(type(ds).__name__, ds)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    if processing_fn is not None:
        logger.info("Using '{}' to process dataset dicts".format(processing_fn))
        dataset_dicts = processing_fn(dataset_dicts)

    has_instances = 'annotations' in dataset_dicts[0]
    if has_instances:
        try:
            utils.check_metadata_consistency('thing_classes', datasets)
            print_instances_class_histogram(dataset_dicts, datasets[0].metadata.thing_classes)
        except AttributeError:  # class names are not available for this dataset
            pass

    return dataset_dicts


def build_batch_data_loader(
    dataset: Dataset,
    sampler: Sampler,
    total_batch_size: int,
    *,
    aspect_ratio_grouping: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """Builds a batched dataloader for training.

    Args:
        dataset: Map-style PyTorch dataset. Can be indexed.
        sampler: A sampler that produces indices
        total_batch_size: Total batch size across GPUs.
        aspect_ratio_grouping: Whether to group images with similar aspect ratio for efficiency.
            When enabled, it requires each element in dataset be a dict with keys "width" and
            "height".
        num_workers: Number of parallel data loading workers

    Returns:
        Length of each list is the batch size of the current GPU. Each element in the list comes
        from the dataset.
    """
    world_size = get_world_size()
    assert total_batch_size > 0 and total_batch_size % world_size == 0, \
        'Total batch size ({}) must be divisible by the number of GPUs ({}).'.format(
            total_batch_size, world_size
        )

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        dataloader = AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )
    return dataloader


def build_detection_train_loader(
    cfg: Dict[str, Any], processing_fn: Callable = None, mapper: Callable = None
) -> DataLoader:
    """Builds a training dataloader from config.

    A data loader is created by the following steps:
    1. Build vision datasets from config, then load and process a list of dicts.
    3. Coordinate a random shuffle order shared among all processes (all GPUs).
    4. Each process spawn another few workers to process the dicts. Each worker will:
        * Map each metadata dict into another format to be consumed by the model.
        * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg: Config should be a dictionary and looks something like:
            {'datasets': ...,
             'processing_params': ...
             'mapper_params': ...}
        processing_fn: A callable that takes dataset dicts as input and returns processed ones. By
            default it will be `Processing(**cfg.get('processing_params', {}))`.
        mapper: A callable that takes a sample (dict) from dataset and returns the format to be
            consumed by the model. By default it will be `DatasetMapper(mapper_params)`.
    """
    vision_datasets = build_vision_datasets(cfg['datasets'])
    if processing_fn is None:
        # Instantiate Processing using processing_params from config by default
        processing_fn = Processing(**cfg.get('processing_params', {}))
    # Loads and processes dataset dicts.
    dataset_dicts = get_detection_dataset_dicts(vision_datasets, processing_fn)

    dataset = DatasetFromList(dataset_dicts, False)
    if mapper is None:
        # Instantiate mapper
        mapper_params = cfg.get('mapper_params', {})
        mapper = DatasetMapper(**mapper_params, training=True, vision_datasets=vision_datasets)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg['dataloader']['sampler']
    if sampler_name == 'TrainingSampler':
        sampler = TrainingSampler(dataset)
    else:
        raise ValueError('Unknown training sampler: {}'.format(sampler_name))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg['dataloader']['images_per_batch'],
        aspect_ratio_grouping=cfg['dataloader']['aspect_ratio_grouping'],
        num_workers=cfg['dataloader']['num_workers'],
    )


def build_detection_test_loader(
    cfg: Dict[str, Any], processing_fn: Callable = None, mapper: Callable = None
) -> DataLoader:
    """Similar to :func:`build_detection_train_loader`."""
    vision_datasets = build_vision_datasets(cfg['datasets'])
    if processing_fn is None:
        processing_fn = Processing(filter_empty=False, min_keypoints=0)
    # Loads and processes dataset dicts.
    dataset_dicts = get_detection_dataset_dicts(vision_datasets, processing_fn)

    dataset = DatasetFromList(dataset_dicts, False)
    if mapper is None:
        # Instantiate mapper
        mapper_params = cfg.get('mapper_params', {})
        mapper = DatasetMapper(**mapper_params, training=False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(dataset)
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg['dataloader']['num_workers'],
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """A batch collator that does nothing."""
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
