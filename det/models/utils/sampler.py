from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from .constant import *

__all__ = ['BalancedPositiveNegativeSampler']


class BalancedPositiveNegativeSampler(object):
    """Samples batches, ensuring that they contain a fixed proportion of positive.

    Args:
        batch_size_per_image (int): Number of elements to be selected per image.
        positive_fraction (float): percentage of positive elements per batch.
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        self._batch_size_per_image = batch_size_per_image
        self._positive_fraction = positive_fraction

    def __call__(self, label):
        """

        Args:
            label (Tensor[N]): Label vector with values:
                * IGNORED: ignored
                * NEGATIVE: negative class
                * otherwise: positive class

        Returns:
            pos_idx, neg_idx (Tensor): The total length of both is `batch_size_per_image`
                or fewer.
        """
        positive = torch.nonzero((label != IGNORED) & (label != NEGATIVE)).squeeze(1)
        negative = torch.nonzero(label == NEGATIVE).squeeze(1)

        num_pos = int(self._batch_size_per_image * self._positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = self._batch_size_per_image - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        # randomly select positive and negative examples
        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]
        return pos_idx, neg_idx
