from __future__ import absolute_import, division, print_function

import copy
import logging
import pickle
import random

import numpy as np
from torch.utils.data import Dataset

__all__ = ['DatasetFromList']

logger = logging.getLogger(__name__)


class MapDataset(Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = map_func

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    'Failed to apply `_map_func` for idx: {}, retry count: {}'.format(
                        idx, retry_count
                    )
                )


class DatasetFromList(Dataset):
    """A class that wrap a list to a torch Dataset. It produces elements of the list as data."""

    def __init__(self, lst: list, copy: bool = True, serialization: bool = True) -> None:
        """
        Args:
            lst: A list which contains elements to produce.
            copy: Whether to deepcopy the element when producing it, so that the result can be
                modified in place without affecting the source in the list.
            serialization: Whether to hold memory using serialized objects, when enabled, data
                loader workers can use shared RAM from master process instead of making a copy.
        """
        self._lst = lst
        self._copy = copy
        self._serialization = serialization

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialization:
            logger.info(
                'Serializing {} elements to byte tensors and concatenating them all ...'.format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info('Serialized dataset takes {:.2f} MiB'.format(len(self._lst) / 1024 ** 2))

    def __len__(self):
        if self._serialization:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        if self._serialization:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            v = memoryview(self._lst[start_addr:end_addr])
            item = pickle.loads(v)
        else:
            item = self._lst[idx]

        if self._copy:
            return copy.deepcopy(item)
        else:
            return item
