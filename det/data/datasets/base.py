from __future__ import absolute_import, division, print_function

from foundation.registry import Registry
from torch.utils.data import Dataset

__all__ = ['VisionDatasetStash', 'VisionDataset']


class VisionDatasetStash(Registry):
    """Registry for vision datasets."""
    pass


class VisionDataset(Dataset):
    """Base vision dataset."""

    @property
    def metadata(self):
        """Metadata which is useful in evaluation, visualization or logging."""
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
