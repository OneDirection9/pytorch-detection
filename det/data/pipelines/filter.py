from __future__ import absolute_import, division, print_function

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Pipeline, PipelineRegistry

__all__ = ['CrowdFilter', 'FewKeypointsFilter']


@PipelineRegistry.register('CrowdFilter')
class CrowdFilter(Pipeline):
    """A pipeline that filter out crowd annotations.

    Returns None if none annotations or images with only crowd annotations.
    """

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = example['annotations']
        res = []
        for ann in annotations:
            if ann.get('iscrowd', 0) == 0:
                res.append(ann)

        if len(res) == 0:  # none annotations or only crowd annotations
            return None
        else:
            example['annotations'] = res
            return example


@PipelineRegistry.register('FewKeypointsFilter')
class FewKeypointsFilter(Pipeline):
    """A pipeline that filter out images with too few number of keypoints."""

    def __init__(self, min_keypoints_per_image: int) -> None:
        """
        Args:
            min_keypoints_per_image: The image with visible keypoints less than
                `min_keypoints_per_image` will be filtered out.
        """
        self._min_keypoints_per_image = min_keypoints_per_image

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = example['annotations']
        # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility.
        num_keypoints = sum(
            (np.array(ann['keypoints'][2::3]) > 0).sum()
            for ann in annotations if 'keypoints' in ann
        )  # yapf: disable

        if num_keypoints < self._min_keypoints_per_image:
            return None
        else:
            return example
