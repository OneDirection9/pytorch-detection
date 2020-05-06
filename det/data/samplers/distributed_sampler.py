# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng han
from __future__ import absolute_import, division, print_function

import itertools
from typing import Iterator, Optional

import torch
from torch.utils.data.sampler import Sampler

from ...utils import comm

__all__ = ['TrainingSampler', 'InferenceSampler']


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data. So this sampler produces
    an infinite stream of indices and all workers cooperate to correctly shuffle the indices and
    sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]` where
    `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False).
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None) -> None:
        """
        Args:
            size: The total number of data of the underlying dataset to sample from.
            shuffle: Whether to shuffle the indices or not.
            seed: The initial seed of the shuffle. Must be the same across all workers. If None,
                will use a random seed shared among workers (require synchronization among all
                workers).
        """
        self._size = size
        assert size > 0
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


class InferenceSampler(Sampler):
    """Produces indices for inference.

    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int) -> None:
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
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
