from __future__ import absolute_import, division, print_function

from foundation.registry import Registry

__all__ = ['MetadataStash', 'DatasetStash', 'TransformStash']


class MetadataStash(Registry):
    """Registry for metadata."""
    pass


class DatasetStash(Registry):
    """Registry for datasets."""
    pass


class TransformStash(Registry):
    """Registry for transforms."""
    pass
