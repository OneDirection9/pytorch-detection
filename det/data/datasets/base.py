from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

from foundation.registry import Registry

from .metadata import Metadata

__all__ = ['VisionDatasetRegistry', 'VisionDataset']


class VisionDatasetRegistry(Registry):
    """Registry of vision datasets."""
    pass


class VisionDataset(object, metaclass=ABCMeta):
    """Base vision dataset class.

    This is not a typical PyTorch format dataset class. It is intended for storing metadata and
    producing a list of examples for future usage, e.g. filtering examples without valid
    annotations, calculating image aspect ratio for grouping, and so on. Then we can use
    :class:`DatasetFromList` to build a PyTorch format dataset class.

    Examples:

    .. code-block:: python

        examples = vision_dataset_instance.get_examples()
        # some operations
        dataset = DatasetFromList(examples)
    """

    @property
    @abstractmethod
    def metadata(self) -> Metadata:
        """Metadata which is useful in evaluation, visualization or logging."""
        pass

    @abstractmethod
    def get_examples(self) -> List[Dict[str, Any]]:
        """Returns a list of dictionaries with paths and annotations."""
        pass

    def __repr__(self):
        return '{}(metadata={})'.format(self.__class__.__name__, self.metadata)

    __str__ = __repr__
