# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

from foundation.common.config import CfgNode

__all__ = ['CfgNode', 'get_cfg']


def get_cfg() -> CfgNode:
    """Gets a copy of default config."""
    from .defaults import _C

    return _C.clone()
