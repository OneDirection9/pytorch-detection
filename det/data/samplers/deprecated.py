from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from torch.utils.data import Sampler

__all__ = ['GroupSampler']


class GroupSampler(Sampler):
    """Samples data from dataset that have similar aspect ratios.

    Args:
        data_source (Dataset): dataset used for sampling.
        samples_per_gpu (int, optional): number of samples in each GPU. Default: 1
    """

    def __init__(self, data_source, samples_per_gpu=1):
        super(GroupSampler, self).__init__(data_source)

        self._data_source = data_source
        self._samples_per_gpu = samples_per_gpu

        self._flags = getattr(data_source, 'aspect_ratio_flag')
        self._range = np.arange(self._samples_per_gpu).reshape(1, -1)

        num_samples, group_ceil_size = 0, []
        unique, counts = np.unique(self._flags, return_counts=True)
        for group, size in zip(unique, counts):
            ceil_size = int(np.ceil(size / samples_per_gpu)) * samples_per_gpu
            num_samples += ceil_size
            group_ceil_size.append((group, ceil_size))

        self._num_samples = num_samples
        self._group_ceil_size = group_ceil_size
        self._num_batches = num_samples // samples_per_gpu

    def __iter__(self):
        indices = []
        # Add extra samples for each group to make it evenly divisible
        for group, ceil_size in self._group_ceil_size:
            if ceil_size == 0:
                continue
            index = np.where(self._flags == group)[0]
            npr.shuffle(index)
            num_extra = ceil_size - len(index)
            index = np.concatenate((index, index[:num_extra]))
            indices.append(index)

        indices = np.concatenate(indices)
        assert len(indices) == self._num_samples

        # Permutate indices by group to make samples in each group has same flag
        index = npr.permutation(self._num_batches).reshape(-1, 1) * self._samples_per_gpu
        index = np.tile(index, (1, self._samples_per_gpu)) + self._range
        index = index.reshape(-1)
        indices = indices[index]
        return iter(indices.tolist())

    def __len__(self):
        return self._num_samples
