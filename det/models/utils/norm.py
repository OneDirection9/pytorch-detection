# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified: Zhipeng Han
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import Function

from foundation.backends.torch.utils import comm

__all__ = ['NaiveSyncBatchNorm2d', 'get_norm']


class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_outputs):
        dist.all_reduce(grad_outputs, async_op=False)
        return grad_outputs


class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if comm.get_world_size() == 1 or not self.training:
            return super(NaiveSyncBatchNorm2d, self).forward(input)

        assert input.shape[0] > 0, 'SyncBatchNorm does not support empty inputs'
        C = input.shape[1]

        # E[(f - E[f])^2] = E[f^2 - 2fE[f] + E[f]^2) = E[f^2] - E[f]^2
        mean = torch.mean(input, dim=[0, 2, 3])
        mean_sqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat((mean, mean_sqr), dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, mean_sqr = torch.split(vec, C)
        var = mean_sqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):
        out_channels (int):

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "SyncBN": NaiveSyncBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)
