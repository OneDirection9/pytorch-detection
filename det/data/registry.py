from __future__ import absolute_import, division, print_function

from foundation.utils import Registry

__all__ = ['DatasetStash', 'TransformStash']


class DatasetStash(Registry):
    """Registry for datasets."""
    pass


class TransformStash(Registry):
    """Registry for preset transforms."""
    pass
