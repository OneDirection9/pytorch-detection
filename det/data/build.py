# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import itertools
import logging
import operator
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from tabulate import tabulate
from termcolor import colored
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from det.config import CfgNode
from det.utils.comm import get_world_size
from det.utils.env import seed_all_rng
from . import utils
from .common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from .dataset_mapper import DatasetMapper
from .datasets import VisionDataset, build_vision_datasets
from .samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler

logger = logging.getLogger(__name__)

__all__ = [
    'print_instances_class_histogram',
    'get_detection_dataset_dicts',
    'build_batch_data_loader',
    'build_detection_train_loader',
    'build_detection_test_loader',
]


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


def filter_images_with_few_keypoints(
    dataset_dicts: List[Dict[str, Any]], min_keypoints: int = 0
) -> List[Dict[str, Any]]:  # yapf: disable
    """Filters out images with too few number of keypoints.

    Args:
        dataset_dicts: Annotations in Detectron2's Dataset format.
        min_keypoints: Filter out images with few keypoints than `min_keypoints`. Set to 0 to do
            nothing.

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


def print_instances_class_histogram(
    dataset_dicts: List[Dict[str, Any]], class_names: List[str]
) -> None:
    """
    Args:
        dataset_dicts: List of dataset dicts.
        class_names: List of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        anns = entry['annotations']
        classes = [x['category_id'] for x in anns if not x.get('iscrowd', 0)]
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
    datasets: List[VisionDataset], filter_empty: bool = True, min_keypoints: int = 0
) -> List[Dict[str, Any]]:  # yapf: disable
    """Loads and processes data for instance detection/segmentation and semantic segmentation.

    Args:
        datasets: List of vision datasets.
        filter_empty: Whether to filter out images without instance annotations.
        min_keypoints: Filter out images with few keypoints than `min_keypoints`. Set to 0 to do
            nothing.

    Returns:
        List of dataset dicts.
    """
    dataset_dicts = [ds.get_items() for ds in datasets]
    for dicts, ds in zip(dataset_dicts, datasets):
        assert len(dicts), "Datasets '{}' is empty!\n{}".format(type(ds).__name__, ds)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = 'annotations' in dataset_dicts[0]
    # Keep images without instance-level GT if the dataset has semantic labels.
    if filter_empty and has_instances and 'sem_seg_file_name' not in dataset_dicts[0]:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)

    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    has_instances = 'annotations' in dataset_dicts[0]
    if has_instances:
        try:
            class_names = datasets[0].metadata.thing_classes
            utils.check_metadata_consistency('thing_classes', datasets)
            print_instances_class_histogram(dataset_dicts, class_names)
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


def build_detection_train_loader(cfg: CfgNode, mapper: Callable = None) -> DataLoader:
    """Builds a training dataloader from config.

    A data loader is created by the following steps:
    1. Build vision datasets from config, then load and process a list of dicts.
    3. Coordinate a random shuffle order shared among all processes (all GPUs).
    4. Each process spawn another few workers to process the dicts. Each worker will:
        * Map each metadata dict into another format to be consumed by the model.
        * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg: The config.
        mapper: A callable that takes a sample (dict) from dataset and returns the format to be
            consumed by the model. By default it will be `DatasetMapper.from_config(cfg, True)`.
    """
    dataset_dicts = get_detection_dataset_dicts(
        build_vision_datasets(cfg.DATASETS.TRAIN),
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON else 0
    )

    dataset = DatasetFromList(dataset_dicts, copy=False)
    if mapper is None:
        mapper = DatasetMapper.from_config(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger.info('Using training sampler {}'.format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == 'TrainingSampler':
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == 'RepeatFactorTrainingSampler':
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError('Unknown training sampler: {}'.format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_detection_test_loader(cfg: CfgNode, mapper: Callable = None) -> DataLoader:
    """Similar to :func:`build_detection_train_loader`."""
    dataset_dicts = get_detection_dataset_dicts(
        build_vision_datasets(cfg.DATASETS.TEST),
        filter_empty=False,
        min_keypoints=0,
    )

    dataset = DatasetFromList(dataset_dicts, False)
    if mapper is None:
        mapper = DatasetMapper.from_config(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """A batch collator that does nothing."""
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
