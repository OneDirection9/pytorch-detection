"""
The workflow of data is:

1. Build raw dataset:
    1.1. Get examples from one or more :class:`VisionDataset`. Build :class:`Metadata` if needed.
    1.2. Pass each example through zero or more :class:`Pipeline` to get a new list of examples.
    1.3. Pass new examples to :class:`DatasetFromList` to get a dataset.
2. Build final dataset:
    2.1. Build :class:`BaseMapper` and build :class:`Transform` or :class:`TransformGen` if needed.
    2.2. Pass mapper to :class:`MapDataset` to get final dataset.
3. Build dataloader
"""
from __future__ import absolute_import, division, print_function

import copy
import itertools
import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from foundation.registry import build
from torch.utils.data import BatchSampler, DataLoader, Dataset

from ..utils.comm import get_world_size
from ..utils.env import seed_all_rng
from . import utils
from .common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from .dataset_mapper import DatasetMapper, DatasetMapperRegistry
from .datasets import MetadataRegistry, VisionDataset, VisionDatasetRegistry
from .mappers import MapperList, MapperRegistry
from .pipelines import PipelineRegistry
from .samplers import InferenceSampler, SamplerRegistry
from .transforms import Transform, TransformGen, TransformGenRegistry, TransformRegistry

__all__ = [
    'build_vision_datasets',
    'build_pipelines',
    'build_transforms',
    'build_dataset_mapper',
    'get_dataset_examples',
    'build_pytorch_dataset',
    'build_train_dataloader',
    'build_test_dataloader',
]

logger = logging.getLogger(__name__)

_SingleCfg = Dict[str, Any]
_MultiCfg = List[Dict[str, Any]]
_CfgType = Union[_SingleCfg, _MultiCfg]


def build_vision_datasets(ds_cfg: _CfgType) -> List[VisionDataset]:
    """Builds vision datasets from config.

    Args:
        ds_cfg: Dataset config that should be a dictionary or a list of dictionaries, each looks
            something like:
            {'name': 'COCOInstance',
             'json_file': './data/MSCOCO/annotations/instances_train2014.json',
             'image_root': './data/MSCOCO/train2014/',
             'metadata': {'name': 'COCOInstanceMetadata'}}

    Returns:
        List of vision datasets.
    """
    if isinstance(ds_cfg, dict):
        ds_cfg = [ds_cfg]

    if len(ds_cfg) == 0:
        raise ValueError('None vision dataset is available')

    # Build metadata
    for cfg in ds_cfg:
        if 'metadata' in cfg:
            cfg['metadata'] = build(MetadataRegistry, cfg['metadata'])

    # Build vision datasets
    vision_datasets = [build(VisionDatasetRegistry, cfg) for cfg in ds_cfg]
    return vision_datasets


def build_pipelines(ppl_cfg: _CfgType) -> List[Callable]:
    """Builds pipelines from config.

    Args:
        ppl_cfg: Pipeline config that should be a dictionary or a list of dictionaries, each looks
            something like:
            {'name': 'FewKeypointsFilter',
             'min_keypoints_per_image': 1}

    Returns:
        List of pipelines.
    """
    if isinstance(ppl_cfg, dict):
        ppl_cfg = [ppl_cfg]

    # Build pipelines
    pipelines = [build(PipelineRegistry, cfg) for cfg in ppl_cfg]
    return pipelines


def build_transforms(tfm_cfg: _CfgType) -> List[Union[Transform, TransformGen]]:
    """Builds transforms from config.

    Args:
        tfm_cfg: Transform configs that should be a dictionary or a list of dictionaries,
            each looks something like:
            {'name': 'RandomHFlip',
             'prob': 0.5}

    Returns:
        List of transforms, either :class:`Transform` or :class:`TransformGen`.
    """
    if isinstance(tfm_cfg, dict):
        tfm_cfg = [tfm_cfg]

    def _build(_cfg: _SingleCfg) -> Union[Transform, TransformGen]:
        # We may pass a Transform or a TransformGen, so we need to discuss it.
        name = _cfg['name']
        if TransformRegistry.contains(name) and TransformGenRegistry.contains(name):
            raise ValueError(
                'Both TransformRegistry and TransformGenRegistry contain {}. '
                'We cannot inference which one do you want to use. You can rename one of them'
                .format(name)
            )

        if TransformRegistry.contains(name):
            return build(TransformRegistry, _cfg)
        elif TransformGenRegistry.contains(name):
            return build(TransformGenRegistry, _cfg)
        else:
            raise KeyError(
                '{} registered in neither TransformRegistry nor TransformGenRegistry'.format(name)
            )

    transforms = []
    for cfg in tfm_cfg:
        if cfg['name'] == 'RandomApply':
            # Currently only RandomApply contains a nested transform
            cfg['transform'] = _build(cfg['transform'])
        transforms.append(_build(cfg))

    return transforms


def build_dataset_mapper(
    mapper_cfg: _SingleCfg, has_keypoints: bool, vision_datasets: List[VisionDataset]
) -> DatasetMapper:
    """Builds dataset mapper.

    Args:
        mapper_cfg: Dataset mapper config that should be a dictionary, and looks something like:
            {'name': 'DictMapper',
             'transforms': [{'name': 'RandomHFlip',
                             'prob': 0.5},
                            ...]}
        has_keypoints: Where has keypoints in annotations. If True, the keypoint_hflip_indices will
            be created.
        vision_datasets: List of vision datasets.

    Returns:
        The dataset mapper.
    """
    if 'transforms' in mapper_cfg:
        mapper_cfg['transforms'] = build_transforms(mapper_cfg['transforms'])
        logger.info('Transforms: {}'.format(mapper_cfg['transforms']))

    if has_keypoints:
        mapper_cfg['keypoint_hflip_indices'] = utils.create_keypoint_hflip_indices(vision_datasets)

    # Build dataset mapper
    dataset_mapper = build(DatasetMapperRegistry, mapper_cfg)
    return dataset_mapper


def get_dataset_examples(
    datasets: List[VisionDataset],
    pipelines: Optional[List[Callable]] = None,
) -> List[Dict[str, Any]]:
    """Gets dataset examples.

    Args:
        datasets: List of vision datasets.
        pipelines: List of pipelines.

    Returns:
         List of examples.
    """
    examples = [ds.get_examples() for ds in datasets]
    for examples_per_ds, ds in zip(examples, datasets):
        if len(examples_per_ds) == 0:
            raise ValueError('{} is empty:\n{}'.format(type(ds).__name__, ds))

    examples = list(itertools.chain.from_iterable(examples))

    if pipelines is not None:
        for ppl in pipelines:
            num_before = len(examples)

            processed = []
            for eg in examples:
                eg = ppl(eg)
                if eg is not None:
                    processed.append(eg)
            examples = processed

            num_after = len(examples)

            logger.info("Pipeline - '{}' done!".format(ppl))
            if num_after != num_before:
                logger.info('Removed {} examples with {}'.format(num_before - num_after, ppl))

            if len(examples) == 0:
                raise ValueError('None examples left!')

    return examples


def build_pytorch_dataset(data_cfg: _SingleCfg) -> Dataset:
    """Builds PyTorch dataset from config.

    Args:
        data_cfg: Data config that should be a dictionary, looks something like:
            {'datasets': [{'name': 'COCOInstance',
                           'json_file': './data/MSCOCO/annotations/instances_train2014.json',
                           'metadata': {'name': 'COCOInstanceMetadata'}}],
             'pipelines': [{'name': 'FewKeypointsFilter', 'min_keypoints_per_image': 0},
                           {'name': 'FormatConverter'}],
             'dataset_mapper': {'name': 'DictMapper',
                                'transforms': [{'name': 'RandomApply',
                                                'transform': {'name': 'RandomHFlip'}},
                                               ...]}}
    """
    data_cfg = copy.deepcopy(data_cfg)
    vision_datasets = build_vision_datasets(data_cfg['datasets'])
    if 'pipelines' in data_cfg:
        pipelines = build_pipelines(data_cfg['pipelines'])
    else:
        pipelines = None
    # examples is a list[dict], where each dict is a record for an image. Example of examples:
    # ['file_name': './data/MSCOCO/train2017/000000000036.jpg'
    #  'height': 640,
    #  'width': 481,
    #  'image_id': 36,
    #  'annotations: [{'iscrowd': 0,
    #                  'bbox': [167.58, 162.89, 478.19, 628.08],
    #                  'segmentation': [345.28, 220.68, ...],
    #                  'keypoints': [250.5, 244.5, 2, ...],
    #                  'category_id': 0},
    #                 ...]
    #  ...]
    examples = get_dataset_examples(vision_datasets, pipelines)

    # Check metadata across multiple datasets
    has_instances = 'annotations' in examples[0]
    if has_instances:
        try:
            utils.check_metadata_consistency('thing_classes', vision_datasets)
            # TODO: add print_instances_class_histogram
        except AttributeError:  # class names are not available for this dataset
            pass

    dataset = DatasetFromList(examples, copy=False, serialization=True)

    if 'dataset_mapper' in data_cfg:
        has_keypoints = has_instances and 'keypoints' in examples[0]['annotations'][0]
        dataset_mapper = build_dataset_mapper(
            data_cfg['dataset_mapper'], has_keypoints, vision_datasets
        )
        dataset = MapDataset(dataset, dataset_mapper)

    return dataset


def build_mapper(mapper_cfg, has_keypoints, vision_datasets):
    res = []
    for m in mapper_cfg:
        if m['name'] == 'RandomHFlip' and has_keypoints:
            m['keypoint_hflip_indices'] = utils.create_keypoint_hflip_indices(vision_datasets)
        res.append(build(MapperRegistry, m))
    return MapperList(res)


def build_pytorch_dataset2(data_cfg: _SingleCfg) -> Dataset:
    """Builds PyTorch dataset from config.

    Args:
        data_cfg: Data config that should be a dictionary, looks something like:
            {'datasets': [{'name': 'COCOInstance',
                           'json_file': './data/MSCOCO/annotations/instances_train2014.json',
                           'metadata': {'name': 'COCOInstanceMetadata'}}],
             'pipelines': [{'name': 'FewKeypointsFilter', 'min_keypoints_per_image': 0},
                           {'name': 'FormatConverter'}],
             'dataset_mapper': {'name': 'DictMapper',
                                'transforms': [{'name': 'RandomApply',
                                                'transform': {'name': 'RandomHFlip'}},
                                               ...]}}
    """
    vision_datasets = build_vision_datasets(data_cfg['datasets'])
    if 'pipelines' in data_cfg:
        pipelines = build_pipelines(data_cfg['pipelines'])
    else:
        pipelines = None
    # examples is a list[dict], where each dict is a record for an image. Example of examples:
    # ['file_name': './data/MSCOCO/train2017/000000000036.jpg'
    #  'height': 640,
    #  'width': 481,
    #  'image_id': 36,
    #  'annotations: [{'iscrowd': 0,
    #                  'bbox': [167.58, 162.89, 478.19, 628.08],
    #                  'segmentation': [345.28, 220.68, ...],
    #                  'keypoints': [250.5, 244.5, 2, ...],
    #                  'category_id': 0},
    #                 ...]
    #  ...]
    examples = get_dataset_examples(vision_datasets, pipelines)

    # Check metadata across multiple datasets
    has_instances = 'annotations' in examples[0]
    if has_instances:
        try:
            utils.check_metadata_consistency('thing_classes', vision_datasets)
            # TODO: add print_instances_class_histogram
        except AttributeError:  # class names are not available for this dataset
            pass

    dataset = DatasetFromList(examples, copy=False, serialization=True)

    if 'mappers' in data_cfg:
        has_keypoints = has_instances and 'keypoints' in examples[0]['annotations'][0]
        dataset_mapper = build_mapper(data_cfg['mappers'], has_keypoints, vision_datasets)
        dataset = MapDataset(dataset, dataset_mapper)

    return dataset


def build_train_dataloader(cfg: _SingleCfg) -> DataLoader:
    """Builds training dataloader from all config.

    Args:
        cfg: Config which loads from a .yaml file.
    """
    cfg = copy.deepcopy(cfg)
    dataset = build_pytorch_dataset(cfg['data']['train'])
    return dataset

    dl_cfg = cfg['dataloader']['train']

    # Build sampler
    sampler_cfg = dl_cfg['sampler']
    sampler_cfg['data_source'] = dataset
    sampler = build(SamplerRegistry, sampler_cfg)

    world_size = get_world_size()
    # images_per_batch: Number of images per batch across all machines
    batch_size = dl_cfg['images_per_batch'] // world_size

    num_workers = dl_cfg['num_workers']
    aspect_ratio_grouping = dl_cfg['aspect_ratio_grouping']

    if aspect_ratio_grouping:
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
        data_loader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

    return data_loader


def build_test_dataloader(cfg: _SingleCfg) -> DataLoader:
    """Builds test dataloader from all config.

    Args:
        cfg: Config which loads from a .yaml file.
    """
    dataset = build_pytorch_dataset(cfg['data']['test'])

    dl_cfg = cfg['dataloader']['test']

    sampler = InferenceSampler(dataset)
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers
    batch_sampler = BatchSampler(sampler, 1, drop_last=False)

    num_workers = dl_cfg['num_workers']
    data_loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """A batch collator that does nothing."""
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
