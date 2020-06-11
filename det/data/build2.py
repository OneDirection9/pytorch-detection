from __future__ import absolute_import, division, print_function

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np

from .common import DatasetFromList, MapDataset
from .datasets import VisionDataset, build_vision_datasets
from .utils import check_metadata_consistency

logger = logging.getLogger(__name__)


class Processing(object):
    """Filtering out images with only crowd annotations and too few number of keypoints."""

    def __init__(self, filter_empty: bool = False, min_keypoints: int = 0) -> None:
        """
        The processing will do nothing by default value (i.e. filter_empty=False, min_keypoints=0).

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
        return self.__class__.__name__ + '(filter_empty={0}, min_keypoints={1}'.format(
            self.filter_empty, self.min_keypoints
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
        logger.info('Using {} to process dataset dicts'.format(processing_fn))
        dataset_dicts = processing_fn(dataset_dicts)

    has_instances = 'annotations' in dataset_dicts[0]
    if has_instances:
        try:
            check_metadata_consistency('thing_classes', datasets)
            # TODO: add print_instances_class_histogram
        except AttributeError:  # class names are not available for this dataset
            pass

    return dataset_dicts


def build_train_loader(
    cfg: Dict[str, Any],
    processing_cls: Type = Processing,
    mapper_cls: Type = None,
):
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
        processing_cls: Type of processing that will be instantiate with parameters from config.
        mapper_cls:

    Returns:

    """
    vision_datasets = build_vision_datasets(cfg['datasets'])
    # Instantiate processing class.
    processing_params = cfg.get('processing_params', {})
    processing_fn = processing_cls(**processing_params)
    # Loads and processes dataset dicts.
    dataset_dicts = get_detection_dataset_dicts(vision_datasets, processing_fn)

    dataset = DatasetFromList(dataset_dicts, False)
    # Instantiate mapper
    mapper_params = cfg.get('mapper_params', {})
    dataset = MapDataset(dataset, mapper_cls(**mapper_params))

    return dataset
