from __future__ import absolute_import, division, print_function

from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['FrozenBatchNorm2d', 'get_norm']


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super(FrozenBatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps

        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            # (x - m)/v * alpha + beta -> scale * x + bias - scale * m
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def __repr__(self) -> str:
        return '{}(num_features={}, eps={})'.format(
            self.__class__.__name__, self.num_features, self.eps
        )

    __str__ = __repr__

    @classmethod
    def convert_frozen_batchnorm(cls, module: nn.Module) -> nn.Module:
        """Converts BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = (nn.BatchNorm2d, nn.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def get_norm(norm: Union[str, nn.Module], out_channels: int) -> Optional[nn.Module]:
    """
    Args:
        norm: Either one of BN, SyncBN, FrozenBN, GN; or a callable that takes a channel number and
            returns the normalization layer as a nn.Module.
        out_channels:

    Returns:
        nn.Module or None: The normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None

        norm = {
            'BN': nn.BatchNorm2d,
            # FIXME: In PyTorch<=1.5, `nn.SyncBatchNorm` has incorrect gradient when the batch size
            #     on each worker is different.
            'SyncBN': nn.SyncBatchNorm,
            'FrozenBN': FrozenBatchNorm2d,
            'GN': lambda channels: nn.GroupNorm(32, channels)
        }[norm]
    return norm(out_channels)
