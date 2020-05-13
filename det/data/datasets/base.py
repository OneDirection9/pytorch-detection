from __future__ import absolute_import, division, print_function

import copy
import logging
import types
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

from foundation.registry import Registry

__all__ = [
    'MetadataStash',
    'Metadata',
    'VisionDatasetStash',
    'VisionDataset',
]

logger = logging.getLogger(__name__)


class MetadataStash(Registry):
    """Registry of metadata."""
    pass


class Metadata(types.SimpleNamespace):
    """A class that supports simple attribute setter/getter.

    It is intended for storing metadata and as a attribute of a :class:`VisionDataset`.

    Examples:

    .. code-block:: python

        # somewhere when you load the data:
        metadata_instance.thing_classes = ['person', 'dog']

        # somewhere when you print statistics or visualize:
        classes = metadata_instance.thing_classes
    """

    # TODO: Need to make sure the mapping is noncyclic. An counterexample: {'a': 'b', 'b': 'a'}
    #   when call like metadata_instance.a, will raise
    #   RecursionError: maximum recursion depth exceeded while calling a Python object
    _RENAMED = {}

    def __getattr__(self, key: str) -> Any:
        if key in self._RENAMED:
            logger.warning("Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]))
            return getattr(self, self._RENAMED[key])

        raise AttributeError(
            "Attribute '{}' does not exist in the metadata. Available keys are {}.".format(
                key, list(self.__dict__.keys())
            )
        )

    def __setattr__(self, key: str, val: Any) -> None:
        if key in self._RENAMED:
            logger.warning("Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]))
            setattr(self, self._RENAMED[key], val)

        # Ensure that metadata of the same name stays consistent
        try:
            old_val = getattr(self, key)
            if old_val != val:
                raise ValueError(
                    "Attribute '{}' cannot be set to a different value!\n{}!={}".format(
                        key, old_val, val
                    )
                )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self) -> Dict[str, Any]:
        """Returns all the metadata as a dict.

        Note that modifications to the returned dict will not reflect on the Metadata object.
        """
        return copy.copy(self.__dict__)

    def set(self, **kwargs: Any) -> 'Metadata':
        """Sets multiple metadata with kwargs."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def get(self, key: str, default=None) -> Any:
        """Accesses an attribute and return its value if exists. Otherwise return default."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class VisionDatasetStash(Registry):
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
