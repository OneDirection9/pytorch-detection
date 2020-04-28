from __future__ import absolute_import, division, print_function

import copy
import logging
import pickle

import numpy as np
import torch.utils.data as data

__all__ = ['DatasetFromList']

logger = logging.getLogger(__name__)


class DatasetFromList(data.Dataset):
    """Wrap a list to a torch Dataset. It produces elements of the list as data."""

    def __init__(self, lst: list, copy: bool = True, serialize: bool = True) -> None:
        """
        Args:
            lst: A list which contains elements to produce.
            copy: Whether to deepcopy the element when producing it, so that the result can be
                modified in place without affecting the source in the list.
            serialize: Whether to hold memory using serialized objects, when enabled, data loader
                workers can use shared RAM from master process instead of making a copy.
        """
        self._lst = lst
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger.info(
                'Serializing {} elements to byte tensors and concatenating them all ...'.format(
                    len(self._lst)
                )
            )
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info('Serialized dataset takes {:.2f} MiB'.format(len(self._lst) / 1024**2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    def __getitem__(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            return pickle.loads(bytes)
        elif self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]
