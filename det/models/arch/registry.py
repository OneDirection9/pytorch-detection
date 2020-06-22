from __future__ import absolute_import, division, print_function

from typing import Any, Dict, Union

import torch
from foundation.registry import Registry, build
from torch import nn

__all__ = ['ArchRegistry', 'build_model']


class ArchRegistry(Registry):
    """Registry of architectures."""
    pass


def build_model(cfg: Dict[str, Any], device: Union[int, str] = 'cuda') -> nn.Module:
    """Builds a meta-architecture, i.e. the whole model, from config."""
    model = build(ArchRegistry, cfg)
    model.to(torch.device(device))
    return model
