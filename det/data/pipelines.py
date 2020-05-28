# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserve
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pycocotools.mask as mask_util
from foundation.registry import Registry

from ..structures import BoxMode

__all__ = [
    'PipelineRegistry',
    'Pipeline',
    'PipelineList',
    'CrowdFilter',
    'FewKeypointsFilter',
    'FormatConverter',
    'AnnotationPopup',
]


class PipelineRegistry(Registry):
    """Registry of pipelines.

    A pipeline, which can be either a function or an object implementing :meth:`__call__`, that
    takes a single example produced by the :class:`VisionDataset` as input and returns processed
    example or None. When returning None, the example should be ignored.

    Typical pipeline use cases are filtering out invalid annotations, converting loaded examples to
    the format accepted by downstream modules, and so on, which are only need to do once during the
    whole workflow.

    Notes:
        Don't do memory heavy work in pipelines, such as loading images. If so, the memory cost is
        expensive. Because the examples returning by pipelines are usually passed to
        :class:`DatasetFromList` to get a PyTorch format class.
    """
    pass


Pipeline = Callable


class PipelineList(object):
    """Maintains a list of pipelines which will be applied in sequence.

    Attributes:
        pipelines (list[Pipeline]):
    """

    def __init__(self, pipelines: List[Pipeline]) -> None:
        """
        Args:
            pipelines: List of pipelines which are executed one by one.
        """
        for ppl in pipelines:
            if not isinstance(ppl, Pipeline):
                raise TypeError('Expected a Pipeline. Got {}'.format(type(ppl)))

        self.pipelines = pipelines

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for ppl in self.pipelines:
            example = ppl(example)
            if example is None:
                return None
        return example

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for ppl in self.pipelines:
            format_string += '\n'
            format_string += '    {0}'.format(ppl)
        format_string += '\n)'
        return format_string


@PipelineRegistry.register('CrowdFilter')
class CrowdFilter(object):
    """Filtering out crowd annotations.

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

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PipelineRegistry.register('FewKeypointsFilter')
class FewKeypointsFilter(object):
    """Filtering out images with too few number of keypoints."""

    def __init__(self, min_keypoints_per_image: int = 0) -> None:
        """
        Args:
            min_keypoints_per_image: The image with visible keypoints less than
                `min_keypoints_per_image` will be filtered out.
        """
        self.min_keypoints_per_image = min_keypoints_per_image

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = example['annotations']
        # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility.
        num_keypoints = sum(
            (np.array(ann['keypoints'][2::3]) > 0).sum()
            for ann in annotations if 'keypoints' in ann
        )  # yapf: disable

        if num_keypoints < self.min_keypoints_per_image:
            return None
        else:
            return example

    def __repr__(self):
        return self.__class__.__name__ + '(min_keypoints_per_image={0})'.format(
            self.min_keypoints_per_image
        )


@PipelineRegistry.register('FormatConverter')
class FormatConverter(object):
    """Converting annotations to the format acceptable by :class:`Transform` inplace.

    This class do following converts:
    1. Convert bounding box to XYXY_ABS format.
    2. Convert polygons to list of np.ndarray of shape Nx2, uncompressed RLE/RLE to 2D np.ndarray.
    3. Convert keypoints to np.ndarray of shape Nx3.
    """

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = example['annotations']
        for ann in annotations:
            bbox = BoxMode.convert(ann['bbox'], ann['bbox_mode'], BoxMode.XYXY_ABS)
            ann['bbox'] = bbox
            ann['bbox_mode'] = BoxMode.XYXY_ABS

            if 'segmentation' in ann:
                segm = ann['segmentation']
                if isinstance(segm, list):
                    # polygons
                    ann['segmentation'] = [np.asarray(p).flatten().reshape(-1, 2) for p in segm]
                elif isinstance(segm, dict):
                    if isinstance(segm['counts'], list):
                        # uncompressed RLE -> encoded RLE
                        segm = mask_util.frPyObjects(segm, segm['size'][0], segm['size'][1])
                    # encoded RLE -> binary masks
                    ann['segmentation'] = mask_util.decode(segm)
                else:
                    raise TypeError(
                        'Expected segmentation in polygons as List[List[float] or np.ndarray] or '
                        'COCO-style RLE as a dict. Got {}'.format(type(segm))
                    )

            if 'keypoints' in ann:
                ann['keypoints'] = np.asarray(ann['keypoints'], dtype='float64').reshape(-1, 3)

        return example

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PipelineRegistry.register('AnnotationPopup')
class AnnotationPopup(object):
    """Popping up unused annotations."""

    def __init__(self, mask_on: bool = False, keypoint_on: bool = False) -> None:
        """
        Args:
            mask_on: If False, remove segmentation annotations.
            keypoint_on: If False, remove keypoints annotations.
        """
        self.mask_on = mask_on
        self.keypoint_on = keypoint_on

    def __call__(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = example['annotations']
        for ann in annotations:
            if not self.mask_on:
                ann.pop('segmentation', None)
            if not self.keypoint_on:
                ann.pop('keypoints', None)
        return example

    def __repr__(self):
        return self.__class__.__name__ + '(mask_on={0}, keypoint_on={1})'.format(
            self.mask_on, self.keypoint_on
        )
