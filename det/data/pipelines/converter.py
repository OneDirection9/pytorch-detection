from __future__ import absolute_import, division, print_function

from typing import Any, Dict, List, Optional

import numpy as np

from ...structures import BoxMode
from .base import Pipeline, PipelineRegistry

__all__ = ['FormatConverter']


@PipelineRegistry.register('FormatConverter')
class FormatConverter(Pipeline):
    """A class that converts format acceptable by :class:`Transform`."""

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = example['annotations']
        for ann in annotations:
            bbox = np.asarray(ann['bbox']).reshape(-1, 4)
            bbox = BoxMode.convert(bbox, ann['bbox_mode'], BoxMode.XYXY_ABS)
            ann['bbox'] = bbox
            ann['bbox_mode'] = BoxMode.XYXY_ABS

        return example
