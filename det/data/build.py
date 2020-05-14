"""
The workflow of data is:

1. Build raw dataset:
    1.1. Get examples from one or more :class:`VisionDataset`. Build :class:`Metadata` if needed.
    1.2. Pass each example through zero or more :class:`Pipeline` to get a new list of examples.
    1.3. Pass new examples to :class:`DatasetFromList` to get a dataset.
2. Build final dataset:
    2.1. Build :class:`BaseMapper` and build :class:`TransformGen` if needed.
    2.2. Pass mapper to :class:`MapDataset` to get final dataset.
"""
from __future__ import absolute_import, division, print_function

import itertools
import logging
from typing import Any, Dict, List, Optional, Union

from foundation.registry import build

from .common import DatasetFromList, MapDataset
from .dataset_mapper import DatasetMapper, DatasetMapperRegistry
from .datasets import MetadataRegistry, VisionDataset, VisionDatasetRegistry
from .pipelines import Pipeline, PipelineRegistry
from .transforms import TransformGen, TransformGenRegistry, TransformRegistry
from .utils import check_metadata_consistency

__all__ = [
    'build_vision_datasets',
    'build_pipelines',
    'get_dataset_examples',
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
        raise ValueError('None dataset is available')

    # Build metadata
    for cfg in ds_cfg:
        if 'metadata' in cfg:
            cfg['metadata'] = build(MetadataRegistry, cfg['metadata'])

    # Build datasets
    datasets = [build(VisionDatasetRegistry, cfg) for cfg in ds_cfg]
    return datasets


def build_pipelines(ppl_cfg: _CfgType) -> List[Pipeline]:
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


def build_transform_gens(tfm_cfg: _CfgType) -> List[TransformGen]:
    """Builds transform generators from config.

    Args:
        tfm_cfg: Transform generator config that should be a dictionary or a list of dictionaries,
            each looks something like:
            {'name': 'RandomHFlip',
             'prob': 0.5}

    Returns:
        List of transform generators.
    """
    if isinstance(tfm_cfg, dict):
        tfm_cfg = [tfm_cfg]

    tfm_gens = []
    for cfg in tfm_cfg:
        if cfg['name'] == 'RandomApply':
            # In RandomApply, we may pass a TransformGen or a Transform, so we need to discuss it.
            name = cfg['transform']['name']
            if TransformRegistry.contains(name) and TransformGenRegistry.contains(name):
                raise ValueError(
                    'Both TransformRegistry and TransformGenRegistry contain {}. '
                    'We cannot inference which one do you want to use. You can rename one of them'
                    .format(name)
                )

            if TransformRegistry.contains(name):
                cfg['transform'] = build(TransformRegistry, cfg['transform'])
            elif TransformGenRegistry.contains(name):
                cfg['transform'] = build(TransformGenRegistry, cfg['transform'])
            else:
                raise KeyError(
                    '{} is not registered in TransformGenRegistry or TransformRegistry'
                    .format(name)
                )
        tfm_gens.append(build(TransformGenRegistry, cfg))

    return tfm_gens


def build_dataset_mapper(mapper_cfg: _SingleCfg) -> DatasetMapper:
    """Builds dataset mapper from config.

    Args:
        mapper_cfg: Mapper config and we can have one and only one mapper. And transform_gen will
            also be build, if provided.

    Returns:
        A mapper.
    """
    if 'transform_gens' in mapper_cfg:
        mapper_cfg['transform_gens'] = build_transform_gens(mapper_cfg['transform_gens'])
    mapper = build(DatasetMapperRegistry, mapper_cfg)
    return mapper


def get_dataset_examples(
    datasets: List[VisionDataset],
    pipelines: Optional[List[Pipeline]] = None,
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

            logger.info('Pipeline - {} done!'.format(ppl))
            if num_after != num_before:
                logger.info('Removed {} examples with {}'.format(num_before - num_after, ppl))

            if len(examples) == 0:
                raise ValueError('None examples left!')

    return examples


def build_train_dataloader(cfg: Dict[str, Any]):
    """Builds train dataloader from all config.

    Args:
        cfg: Config that loaded from .yaml file.
    """
    _cfg = cfg['data']['train']

    datasets = build_vision_datasets(_cfg['datasets'])
    if 'pipelines' in _cfg:
        pipelines = build_pipelines(_cfg['pipelines'])
    else:
        pipelines = None
    examples = get_dataset_examples(datasets, pipelines)

    # Check metadata across multiple datasets
    has_instances = 'annotations' in examples[0]
    if has_instances:
        try:
            check_metadata_consistency('thing_classes', datasets)
        except AttributeError:  # class names are not available for this dataset
            pass

    dataset = DatasetFromList(examples, copy=True, serialization=True)

    if 'mapper' in _cfg:
        mapper = build_dataset_mapper(_cfg['mapper'])
        dataset = MapDataset(dataset, mapper)

    return dataset
