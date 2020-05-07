"""
The workflow of data is:

1. Get examples from one or more :class:`VisionDataset`.
2. Passes each example through zero or more :class:`Pipeline`.
"""
from __future__ import absolute_import, division, print_function

import itertools
import logging
from typing import Any, Dict, List, Optional

from foundation.registry import build

from .datasets import MetadataStash, VisionDataset, VisionDatasetStash
from .pipelines import Pipeline, PipelineRegistry
from .utils import check_metadata_consistency

__all__ = [
    'build_vision_datasets',
    'build_pipelines',
    'get_dataset_examples',
]

logger = logging.getLogger(__name__)

_CfgType = List[Dict[str, Any]]


def build_vision_datasets(ds_cfg: _CfgType) -> List[VisionDataset]:
    """Builds vision datasets from config.

    Args:
        ds_cfg: Dataset config that should be list of dicts, each looks something like:
            {'name': 'COCOInstance',
             'json_file': './data/MSCOCO/annotations/instances_train2014.json',
             'image_root': './data/MSCOCO/train2014/',
             'metadata': {'name': 'COCOInstanceMetadata'}}

    Returns:
        List of datasets.
    """
    if isinstance(ds_cfg, dict):
        ds_cfg = [ds_cfg]

    if len(ds_cfg) == 0:
        raise ValueError('None dataset is available')

    # Build metadata
    for cfg in ds_cfg:
        if 'metadata' in cfg:
            cfg['metadata'] = build(MetadataStash, cfg['metadata'])

    # Build datasets
    datasets = [build(VisionDatasetStash, cfg) for cfg in ds_cfg]
    return datasets


def build_pipelines(ppl_cfg: _CfgType) -> List[Pipeline]:
    """Builds pipelines from config.

    Args:
        ppl_cfg: Pipeline config that should be list of dicts, each looks something like:
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


def get_dataset_examples(ds_cfg: _CfgType, ppl_cfg: Optional[_CfgType] = None) -> List[Dict]:
    """Gets dataset examples.

    Args:
        ds_cfg: See :func:`build_vision_datasets`.
        ppl_cfg: See :func:`build_pipelines`.

    Returns:
         List of examples.
    """
    datasets = build_vision_datasets(ds_cfg)
    examples = [ds.get_examples() for ds in datasets]
    for examples_per_ds, ds in zip(examples, datasets):
        if len(examples_per_ds) == 0:
            raise ValueError('{} is empty:\n{}'.format(type(ds).__name__, ds))

    examples = list(itertools.chain.from_iterable(examples))

    if ppl_cfg is not None:
        pipelines = build_pipelines(ppl_cfg)

        for ppl in pipelines:
            num_before = len(examples)

            processed = []
            for eg in examples:
                eg = ppl(eg)
                if eg is not None:
                    processed.append(eg)
            examples = processed

            num_after = len(examples)

            logger.info('Pipeline {} done'.format(ppl))
            if num_after != num_before:
                logger.info('Removed {} examples with {}'.format(num_before - num_after, ppl))

            if len(examples) == 0:
                raise ValueError('No examples left!')

    has_instances = 'annotations' in examples[0]
    if has_instances:
        try:
            check_metadata_consistency('thing_classes', datasets)
        except AttributeError:  # class names are not available for this dataset
            pass

    return examples
