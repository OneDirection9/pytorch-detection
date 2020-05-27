# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import itertools
from typing import Iterator, Optional, Union

import torch
from foundation.registry import Registry
from torch.utils.data import Dataset, Sampler

from ...utils import comm

__all__ = ['SamplerRegistry', 'TrainingSampler', 'InferenceSampler']


class SamplerRegistry(Registry):
    """Registry of samplers."""
    pass


@SamplerRegistry.register('TrainingSampler')
class TrainingSampler(Sampler):
    """Producing an infinite stream of indices.

    In training, we only care about the "infinite stream" of training data. So this sampler produces
    an infinite stream of indices and all workers cooperate to correctly shuffle the indices and
    sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]` where
    `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False).
    """

    def __init__(
        self,
        data_source: Union[list, Dataset],
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> None:
        """
        Args:
            data_source: :class:`Dataset` instance or a list that has all elements and implements
                :meth:`__len__` which is the total number of data of the underlying dataset to
                sample from.
            shuffle: Whether to shuffle the indices or not.
            seed: The initial seed of the shuffle. Must be the same across all workers. If None,
                will use a random seed shared among workers (require synchronization among all
                workers).
        """
        size = len(data_source)
        assert size > 0
        self._size = size

        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self) -> Iterator[int]:
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


@SamplerRegistry.register('InferenceSampler')
class InferenceSampler(Sampler):
    """Produces indices for inference.

    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, data_source: Union[list, Dataset]) -> None:
        """
        Args:
            data_source: :class:`Dataset` instance or a list that has all elements and implements
                :meth:`__len__` which is the total number of data of the underlying dataset to
                sample from.
        """
        size = len(data_source)
        assert size > 0
        self._size = size
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self) -> Iterator[int]:
        yield from self._local_indices

    def __len__(self) -> int:
        return len(self._local_indices)
