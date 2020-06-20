from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

from foundation.registry import Registry, build

from .metadata import Metadata

__all__ = [
    'VisionDataset',
    'VisionDatasetRegistry',
    'build_vision_datasets',
]


class VisionDataset(object, metaclass=ABCMeta):
    """Base vision dataset class.

    This is not a typical PyTorch format dataset class. It is intended for storing metadata and
    producing a list of samples for future usage, e.g. filtering samples without valid annotations,
    calculating image aspect ratio for grouping, and so on. Then we can use :class:`DatasetFromList`
    to build a PyTorch format dataset class.

    Examples:

    .. code-block:: python

        items = vision_dataset_instance.get_items()
        items = processing_fn(items)
        dataset = DatasetFromList(items)
    """

    def __init__(self, metadata: Optional[Metadata] = None) -> None:
        if metadata is None:
            metadata = Metadata()
        elif isinstance(metadata, Metadata):
            # Copy to avoid modify original metadata
            metadata = Metadata(**metadata.as_dict())
        else:
            raise TypeError(
                'metadata should be instance of Metadata or None. Got {}'.format(type(metadata))
            )
        self._metadata = metadata

    @property
    def metadata(self) -> Metadata:
        """Metadata which is useful in evaluation, visualization or logging."""
        return self._metadata

    @abstractmethod
    def get_items(self) -> List[Dict[str, Any]]:
        """Returns a list of dictionaries with paths and annotations."""
        pass

    def __repr__(self):
        return '{}(metadata={})'.format(self.__class__.__name__, self.metadata)

    __str__ = __repr__


class VisionDatasetRegistry(Registry):
    """Registry of vision datasets.

    A vision dataset should have get_items method returning list of samples in dictionary, and
    metadata attribute. See :class:`VisionDataset`.
    """
    pass


def build_vision_datasets(cfg: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[VisionDataset]:
    """Builds vision datasets from config.

    Args:
        cfg: Dataset config that should be a dictionary or a list of dictionaries, each looks
            something like:
            {'name': 'COCOInstance',
             'json_file': './data/MSCOCO/annotations/instances_train2014.json',
             'image_root': './data/MSCOCO/train2014/'}

    Returns:
        List of vision datasets.
    """
    if isinstance(cfg, dict):
        cfg = [cfg]

    if len(cfg) == 0:
        raise ValueError('None vision dataset is available')

    vision_datasets = [build(VisionDatasetRegistry, c) for c in cfg]
    return vision_datasets
